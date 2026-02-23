
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from openai import APIError, AuthenticationError, RateLimitError
import tempfile, os, json
from loguru import logger

from ..services.cache import sha256_bytes, read_quiz, save_quiz
from ..services.pdf import build_bullets_from_pdf
from ..services.llm import llm
from ..services.parse import parse_quiz
from ..auth import user_id_from_auth_header
from ..services.db import insert_quiz, upsert_document, upload_pdf_to_storage, find_document_id_by_hash, new_uuid
from ..settings import settings

router = APIRouter()

@router.post("/quiz")
async def quiz(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    num_questions: int = Form(18)
):
    if num_questions < 10: num_questions = 10
    if num_questions > 40: num_questions = 40

    raw = await file.read()
    if not raw: raise HTTPException(400, "Empty file.")
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "Only PDF supported.")
    if len(raw) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF too large. Max {settings.MAX_UPLOAD_MB} MB.")

    content_hash = sha256_bytes(raw)
    cached = read_quiz(content_hash)
    if cached:
        payload = dict(cached)
        payload["id"] = new_uuid()
        return payload

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw); tmp_path = tmp.name

    try:
        joined, _ = await build_bullets_from_pdf(tmp_path, content_hash)

        sys = (
            "Return only valid JSON with no extra text. "
            "Schema: {\"questions\":[{\"question\":\"...\",\"choices\":[\"A\",\"B\",\"C\",\"D\"],"
            "\"answer_index\":0,\"explanation\":\"...\",\"source\":\"Slide X\"}]}."
        )
        raw_quiz = await llm(
            [
                {"role":"system","content": sys},
                {"role":"user","content": f"Create {num_questions} MCQs from these bullets:\n{joined[:12000]}"},
            ],
            max_tokens=2000, temperature=0.2
        )

        try:
            quiz_obj = parse_quiz(raw_quiz)
        except Exception:
            repaired = await llm(
                [
                    {"role":"system","content": sys},
                    {"role":"user","content": "Repair strictly to schema (4 choices each):\n" + raw_quiz}
                ],
                max_tokens=2000
            )
            quiz_obj = parse_quiz(repaired)

        payload = {"id": new_uuid(), "title": title, "num_questions": len(quiz_obj["questions"]),
                   "quiz_json": json.dumps(quiz_obj, ensure_ascii=False)}
        save_quiz(content_hash, payload)

        # Save to Supabase if logged in
        try:
            user_id = user_id_from_auth_header(request.headers.get("Authorization"))
            if user_id:
                # Ensure there is a Document row (and PDF in Storage) so quizzes can link to it.
                doc_uuid = find_document_id_by_hash(user_id=user_id, content_hash=content_hash)
                if not doc_uuid:
                    doc_uuid = new_uuid()
                    pdf_path = upload_pdf_to_storage(user_id=user_id, doc_id=doc_uuid, raw_pdf=raw, filename=file.filename)
                    upsert_document(
                        user_id=user_id,
                        doc_id=doc_uuid,
                        title=title,
                        summary="",
                        cards_json=json.dumps({"cards": []}, ensure_ascii=False),
                        guide_json=None,
                        pdf_path=pdf_path,
                        content_hash=content_hash,
                    )
                insert_quiz(
                    user_id=user_id,
                    doc_id=doc_uuid,
                    title=title,
                    quiz_json=payload["quiz_json"],
                    num_questions=payload["num_questions"],
                )
        except Exception as e:
            logger.warning(f"[quiz] persist error: {e}")

        return payload

    except AuthenticationError:
        raise HTTPException(401, "OpenAI auth failed.")
    except RateLimitError:
        raise HTTPException(429, "OpenAI quota/rate limit exceeded.")
    except APIError as e:
        raise HTTPException(502, f"OpenAI API error: {getattr(e, 'message', str(e))}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        try: os.remove(tmp_path)
        except: pass
