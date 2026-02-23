from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import PyPDF2
import io
import json

from ..auth import user_id_from_auth_header
from ..supabase import supabase
from ..services.db import new_uuid
from ..services.llm import llm

router = APIRouter(prefix="/syllabus", tags=["syllabus"])


class SyllabusData(BaseModel):
    class_name: str
    subject_area: Optional[str] = None
    instructor: Optional[str] = None
    description: Optional[str] = None
    assignments: List[dict] = []
    grading_policy: Optional[str] = None
    course_schedule: Optional[str] = None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            s = s.replace("json", "", 1).strip() if s.lstrip().startswith("json") else s.strip()
    return s.strip()


async def parse_syllabus_with_openai(text: str) -> dict:
    """Use OpenAI (via app/services/llm.py) to parse syllabus content -> JSON"""
    prompt = f"""Parse this syllabus and extract structured information.
Return ONLY valid JSON. No markdown, no backticks, no commentary.

Syllabus text:
{text[:12000]}

Extract fields:
1. class_name (course title/name)
2. subject_area (e.g., "Computer Science", "Biology", etc.)
3. instructor (professor name if present)
4. description (brief course description)
5. assignments: list of objects with:
   - title (string)
   - due_date (ISO 8601 date string if present, else null)
   - points (number if present, else null)
   - description (string if present, else null)
6. grading_policy (grading breakdown text if present)
7. course_schedule (high-level schedule text if present)

JSON only:"""

    try:
        # Your llm() helper uses OpenAI chat.completions under the hood
        raw = await llm(
            [
                {
                    "role": "system",
                    "content": "You extract structured data. Output STRICT JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        raw = _strip_code_fences(raw)

        # Try direct JSON parse
        return json.loads(raw)

    except json.JSONDecodeError:
        # Attempt salvage: find first '{' and last '}'
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to parse syllabus JSON from model output")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse syllabus: {str(e)}")


@router.post("/upload")
async def upload_syllabus(
    class_id: str,
    file: UploadFile = File(...),
    user_id: Optional[str] = Depends(user_id_from_auth_header),
):
    """Upload and parse a syllabus PDF for a class"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    # Verify class ownership
    check = (
        supabase.table("classes")
        .select("id,name")
        .eq("id", class_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if not check.data:
        raise HTTPException(status_code=404, detail="Class not found")

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # Parse with OpenAI
    parsed_data = await parse_syllabus_with_openai(text)

    # Update class with syllabus data
    supabase.table("classes").update(
        {
            "has_syllabus": True,
            "subject_area": parsed_data.get("subject_area"),
            "instructor": parsed_data.get("instructor"),
            "description": parsed_data.get("description"),
            "grading_policy": parsed_data.get("grading_policy"),
        }
    ).eq("id", class_id).execute()

    # Create assignments (if table exists)
    assignments = parsed_data.get("assignments", []) or []
    created = 0
    for assignment in assignments:
        try:
            supabase.table("assignments").insert(
                {
                    "id": new_uuid(),
                    "class_id": class_id,
                    "user_id": user_id,
                    "title": (assignment.get("title") or "Untitled")[:200],
                    "description": assignment.get("description"),
                    "due_date": assignment.get("due_date"),
                    "points": assignment.get("points"),
                    "completed": False,
                }
            ).execute()
            created += 1
        except Exception:
            # If assignments table doesn't exist yet or insert fails, skip safely
            pass

    return {
        "success": True,
        "parsed_data": parsed_data,
        "assignments_created": created,
    }


@router.get("/preview/{class_id}")
def get_syllabus_data(
    class_id: str,
    user_id: Optional[str] = Depends(user_id_from_auth_header),
):
    """Get parsed syllabus data for a class"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    result = (
        supabase.table("classes")
        .select("id,name,subject_area,instructor,description,grading_policy,has_syllabus")
        .eq("id", class_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Class not found")

    return result.data