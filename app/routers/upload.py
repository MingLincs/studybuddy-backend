from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from openai import APIError, AuthenticationError, RateLimitError
import tempfile
import os
import json
import asyncio
from loguru import logger

from ..services.cache import sha256_bytes
from ..services.pdf import build_bullets_from_pdf
from ..services.llm import llm
from ..services.parse import parse_cards
from ..services.db import upsert_document, upload_pdf_to_storage, new_uuid
from ..services.concept_engine import update_class_graph
from ..services.knowledge_graph import extract_knowledge_graph
from ..auth import user_id_from_auth_header

router = APIRouter()


# --------------------------------------------------
# CONCEPT PROMPT
# --------------------------------------------------

CONCEPT_PROMPT = """
You are building structured study concepts from a textbook chapter.

Return ONLY valid JSON:

{
  "concepts": [
    {
      "name": "Concept Name",
      "importance": "core|important|advanced",
      "difficulty": "easy|medium|hard",
      "simple": "Short intuitive explanation",
      "detailed": "Deeper explanation with structure and reasoning",
      "technical": "More formal or technical description",
      "example": "Clear real-world example",
      "common_mistake": "Typical misunderstanding students make"
    }
  ]
}

Rules:
- Generate 6–10 high-quality concepts
- Concepts must be meaningful
- Detailed explanations must be rich (4–6 sentences)
- Examples must be concrete
- Common mistakes must be realistic
- Stay faithful to the provided text
"""


# --------------------------------------------------
# UPLOAD ENDPOINT
# --------------------------------------------------

@router.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("Comprehensive Study Notes"),
    class_id: str | None = Form(None),
    make_summary: str = Form("1"),
    make_cards: str = Form("1"),
    make_guide: str = Form("1"),
    word_target: int = Form(3000),
):

    # ----------------------------
    # Validate file
    # ----------------------------

    raw = await file.read()

    if not raw:
        raise HTTPException(400, "Empty file.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported.")

    # ----------------------------
    # Auth
    # ----------------------------

    user_id = user_id_from_auth_header(request.headers.get("Authorization"))
    if not user_id:
        raise HTTPException(401, "Login required.")

    if not class_id:
        raise HTTPException(400, "class_id required.")

    # ----------------------------
    # Flags
    # ----------------------------

    to_bool = lambda v: str(v).lower() in ("1", "true", "yes", "on")
    want_summary = to_bool(make_summary)
    want_cards = to_bool(make_cards)
    want_guide = to_bool(make_guide)

    if not (want_summary or want_cards or want_guide):
        raise HTTPException(400, "Select at least one option.")

    doc_id = new_uuid()
    content_hash = sha256_bytes(raw)

    # ----------------------------
    # Temp save PDF
    # ----------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        # Extract text
        joined, _ = await build_bullets_from_pdf(tmp_path, content_hash)

        tasks = {}

        # ----------------------------
        # Summary
        # ----------------------------

        if want_summary:
            tasks["summary"] = llm(
                [
                    {
                        "role": "system",
                        "content": (
    f"Write detailed structured study notes in markdown (~{word_target} words). "
    "Use headings/subheadings, bullets, and clear spacing.\n\n"
    "MATH FORMATTING RULES:\n"
    "- Any equation must be written in LaTeX.\n"
    "- Inline math must use $...$.\n"
    "- Display equations must use $$...$$ on their own lines.\n"
    "- Use standard LaTeX like \\frac{}, \\sqrt{}, ^{}, _{}, \\sum, \\int.\n"
    "- Do NOT put equations in code blocks.\n"
),
                    },
                    {"role": "user", "content": joined[:18000]},
                ],
                max_tokens=4000,
                temperature=0.2,
            )

        # ----------------------------
        # Flashcards
        # ----------------------------
        # NOTE: We generate flashcards AFTER we extract the knowledge graph.
        # This makes cards higher signal and prevents empty output when
        # the model returns JSON with code fences.

        # ----------------------------
        # Concepts / Knowledge Graph
        # ----------------------------

        if want_guide:
            tasks["graph"] = extract_knowledge_graph(joined, max_nodes=12)

        # ----------------------------
        # Run AI tasks in parallel
        # ----------------------------

        results = {}
        if tasks:
            keys = list(tasks.keys())
            values = await asyncio.gather(*[tasks[k] for k in keys])
            results = dict(zip(keys, values))

        summary = results.get("summary", "") or ""

        graph = results.get("graph", {})
        if isinstance(graph, Exception):
            graph = {}
        if not isinstance(graph, dict):
            graph = {}

        # ----------------------------
        # Knowledge graph JSON (for concept map + storage)
        # ----------------------------

        guide_json = "{}"
        if want_guide and graph.get("concepts"):
            guide_json = json.dumps(graph, ensure_ascii=False)

        # ----------------------------
        # Generate flashcards (from extracted concepts when available)
        # ----------------------------

        cards_json = json.dumps({"cards": []})
        if want_cards:
            try:
                concepts = graph.get("concepts") if isinstance(graph, dict) else None
                if isinstance(concepts, list) and concepts:
                    concept_lines = "\n".join(
                        f"- {c.get('name')}: {c.get('simple', '')}" for c in concepts[:12]
                    )
                    cards_resp = await llm(
                        [
                            {
                                "role": "system",
                                "content": 'Return ONLY valid JSON: {"cards":[{"type":"definition|qa|concept","front":"...","back":"..."}]}. Make cards exam-focused.',
                            },
                            {
                                "role": "user",
                                "content": "Create 20–30 high-quality flashcards from these core concepts:\n\n" + concept_lines,
                            },
                        ],
                        max_tokens=2000,
                        temperature=0.2,
                    )
                    parsed_cards = parse_cards(cards_resp)
                    cards_json = json.dumps(parsed_cards, ensure_ascii=False)
                else:
                    cards_resp = await llm(
                        [
                            {
                                "role": "system",
                                "content": 'Return ONLY valid JSON: {"cards":[{"type":"definition|qa|concept","front":"...","back":"..."}]}.',
                            },
                            {
                                "role": "user",
                                "content": "Create 20–30 high-quality flashcards from this content:\n\n" + joined[:15000],
                            },
                        ],
                        max_tokens=2000,
                        temperature=0.2,
                    )
                    parsed_cards = parse_cards(cards_resp)
                    cards_json = json.dumps(parsed_cards, ensure_ascii=False)
            except Exception:
                cards_json = json.dumps({"cards": []})

        # ----------------------------
        # Save PDF
        # ----------------------------

        pdf_path = upload_pdf_to_storage(
            user_id=user_id,
            doc_id=doc_id,
            raw_pdf=raw,
            filename=file.filename,
        )

        # ----------------------------
        # Save Document
        # ----------------------------

        upsert_document(
            user_id=user_id,
            class_id=class_id,
            doc_id=doc_id,
            title=title,
            summary=summary,
            cards_json=cards_json,
            guide_json=guide_json,
            pdf_path=pdf_path,
            content_hash=content_hash,
        )

        # ----------------------------
        # 🔥 Update AI Concept Graph
        # ----------------------------

        if want_guide and guide_json != "{}":
            try:
                await update_class_graph(
                    class_id=class_id,
                    doc_id=doc_id,
                    guide_json=guide_json,
                )
            except Exception as e:
                logger.warning(f"[graph] update_class_graph failed: {e}")

        # ----------------------------
        # Return response
        # ----------------------------

        return {
            "id": doc_id,
            "summary": summary,
            "cards_json": cards_json,
            "guide_json": guide_json,
        }

    except AuthenticationError:
        raise HTTPException(401, "OpenAI auth failed.")
    except RateLimitError:
        raise HTTPException(429, "Rate limit exceeded.")
    except APIError as e:
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass