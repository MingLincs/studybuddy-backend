from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date, time, timezone
from zoneinfo import ZoneInfo
import re
import json

from icalendar import Calendar

from ..auth import user_id_from_auth_header
from ..supabase import supabase
from ..services.db import new_uuid
from ..services.llm import llm  # optional fallback

router = APIRouter(prefix="/calendar", tags=["calendar"])

LOCAL_TZ = ZoneInfo("America/Chicago")

# Matches codes like: INFOTC-4400, CMP_SC-4540
COURSE_CODE_RE = re.compile(r"([A-Z]{2,}(?:_[A-Z]{2,})?-\d{3,4})", re.IGNORECASE)

# Heuristics to keep only assignments (tune as you like)
ASSIGNMENT_KEYWORDS = [
    "assignment", "homework", "hw", "quiz", "exam", "midterm", "final",
    "project", "lab", "paper", "essay", "discussion", "reading",
    "due", "submission", "submit", "checkpoint"
]
NON_ASSIGNMENT_KEYWORDS = [
    "office hour", "lecture", "class meeting", "zoom", "recitation",
    "review session", "lab meeting", "seminar", "holiday"
]


# ---------------- helpers ----------------

def _safe_str(x) -> str:
    return "" if x is None else str(x).strip()


def _normalize(s: str) -> str:
    return " ".join((s or "").lower().split())


def _is_assignment_like(summary: str, description: str, categories: str) -> bool:
    """
    Only keep events that look like assignments. Canvas feeds may include
    lectures/office hours/etc. We filter aggressively.
    """
    blob = _normalize(f"{summary} {description} {categories}")

    # If it contains explicit non-assignment cues, drop it
    for k in NON_ASSIGNMENT_KEYWORDS:
        if k in blob:
            return False

    # If it contains assignment cues, keep it
    for k in ASSIGNMENT_KEYWORDS:
        if k in blob:
            return True

    # Canvas sometimes labels categories as "assignment"
    if "assignment" in blob:
        return True

    return False


def _dt_to_iso(dt_value) -> Optional[str]:
    """
    Convert DTSTART into ISO for TIMESTAMPTZ.
    - If dt_value is a DATE (all-day): store as NOON in America/Chicago -> UTC
      to avoid showing the previous day in local time.
    - If dt_value is datetime:
        - if naive: assume LOCAL_TZ
        - convert to UTC ISO
    """
    if dt_value is None:
        return None

    # All-day date (VALUE=DATE)
    if isinstance(dt_value, date) and not isinstance(dt_value, datetime):
        local_noon = datetime.combine(dt_value, time(12, 0), tzinfo=LOCAL_TZ)
        return local_noon.astimezone(timezone.utc).isoformat()

    # Datetime
    if isinstance(dt_value, datetime):
        if dt_value.tzinfo is None:
            dt_value = dt_value.replace(tzinfo=LOCAL_TZ)
        return dt_value.astimezone(timezone.utc).isoformat()

    return None


def _extract_course_code_anywhere(summary: str, description: str) -> Optional[str]:
    blob = f"{summary}\n{description}".upper()
    m = COURSE_CODE_RE.search(blob)
    return m.group(1).upper() if m else None


def _build_class_code_map(classes: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    code_to_id: COURSECODE -> class_id
    id_to_name: class_id -> class_name
    Includes INFOTC <-> INFOC aliasing (common typo mismatch).
    """
    code_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}

    for c in classes:
        cid = c.get("id")
        name = (c.get("name") or "").strip()
        if not cid or not name:
            continue

        id_to_name[cid] = name

        m = COURSE_CODE_RE.search(name.upper())
        if not m:
            continue

        code = m.group(1).upper().strip()
        code_to_id[code] = cid

        # Alias INFOC <-> INFOTC
        if code.startswith("INFOTC-"):
            code_to_id[code.replace("INFOTC-", "INFOC-")] = cid
        if code.startswith("INFOC-"):
            code_to_id[code.replace("INFOC-", "INFOTC-")] = cid

    return code_to_id, id_to_name


def _already_exists(user_id: str, class_id: str, title: str, due_date_iso: Optional[str]) -> bool:
    q = (
        supabase.table("assignments")
        .select("id")
        .eq("user_id", user_id)
        .eq("class_id", class_id)
        .eq("title", title)
    )
    if due_date_iso:
        q = q.eq("due_date", due_date_iso)
    res = q.limit(1).execute()
    return bool(res.data)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()
    return s.strip()


async def _llm_match_events_to_classes(
    events: List[Dict[str, Any]],
    classes: List[Dict[str, Any]],
) -> Dict[int, Optional[str]]:
    """
    Optional: If we can't match by course code, let the LLM pick the best class.
    Returns mapping local_idx -> class_id or None.
    """
    class_list = [{"id": c["id"], "name": c["name"]} for c in classes]
    event_list = [
        {
            "idx": i,
            "summary": (e.get("summary") or "")[:160],
            "description": (e.get("description") or "")[:240],
        }
        for i, e in enumerate(events)
    ]

    prompt = f"""
Match Canvas calendar assignment items to classes.

Return ONLY valid JSON.

Classes:
{json.dumps(class_list)}

Items:
{json.dumps(event_list)}

Output format:
{{"matches":[{{"idx":0,"class_id":"... or null"}}, ...]}}
"""

    raw = await llm(
        [
            {"role": "system", "content": "Return STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        temperature=0.0,
    )
    raw = _strip_code_fences(raw)
    obj = json.loads(raw)

    out: Dict[int, Optional[str]] = {}
    for m in obj.get("matches", []):
        idx = m.get("idx")
        cid = m.get("class_id")
        if isinstance(idx, int):
            out[idx] = cid if cid else None
    return out


# ---------------- routes ----------------

@router.post("/import")      # frontend calls this
@router.post("/import-ics")  # optional alias
async def import_canvas_ics(
    file: UploadFile = File(...),
    user_id: Optional[str] = Depends(user_id_from_auth_header),
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    if not (file.filename or "").lower().endswith(".ics"):
        raise HTTPException(status_code=400, detail="Only .ics files are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        cal = Calendar.from_ical(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse .ics: {str(e)}")

    # Load user's classes
    classes_res = (
        supabase.table("classes")
        .select("id,name")
        .eq("user_id", user_id)
        .execute()
    )
    classes = classes_res.data or []
    if not classes:
        return {
            "success": True,
            "assignments_created": 0,
            "assignments_skipped": 0,
            "matched_classes": [],
            "debug": {"message": "No classes found"},
        }

    code_to_id, id_to_name = _build_class_code_map(classes)

    total_vevents = 0
    filtered_out = 0

    # Collect ONLY assignment-like VEVENTs
    items: List[Dict[str, Any]] = []
    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue

        total_vevents += 1

        summary = _safe_str(comp.get("summary"))
        description = _safe_str(comp.get("description"))
        categories = _safe_str(comp.get("categories"))

        if not _is_assignment_like(summary, description, categories):
            filtered_out += 1
            continue

        dtstart = comp.get("dtstart")
        dtstart_val = dtstart.dt if dtstart else None
        due_date_iso = _dt_to_iso(dtstart_val)

        title = summary or "Untitled"

        code = _extract_course_code_anywhere(summary, description)

        items.append(
            {
                "summary": summary,
                "description": description,
                "title": title,
                "due_date": due_date_iso,
                "course_code": code,
            }
        )

    # First pass: course code match
    matched_class_names = set()
    created = 0
    skipped = 0

    no_code_or_nomatch: List[Dict[str, Any]] = []
    class_ids_for_items: List[Optional[str]] = []

    for it in items:
        cid = None
        code = it.get("course_code")
        if code:
            cid = code_to_id.get(code.upper().strip())
        class_ids_for_items.append(cid)
        if not cid:
            no_code_or_nomatch.append(it)

    # Optional LLM fallback for the ones without a code match
    llm_used = False
    llm_error = None
    if no_code_or_nomatch:
        try:
            llm_used = True
            llm_map = await _llm_match_events_to_classes(no_code_or_nomatch, classes)
            # apply back in order
            j = 0
            for i in range(len(items)):
                if class_ids_for_items[i] is None:
                    suggested = llm_map.get(j)
                    class_ids_for_items[i] = suggested
                    j += 1
        except Exception as e:
            llm_error = str(e)

    insert_errors_sample: List[str] = []
    skipped_no_match = 0
    skipped_duplicate = 0

    for it, class_id in zip(items, class_ids_for_items):
        if not class_id:
            skipped += 1
            skipped_no_match += 1
            continue

        cname = id_to_name.get(class_id)
        if cname:
            matched_class_names.add(cname)

        title = (it.get("title") or "Untitled")[:200]
        due_date_iso = it.get("due_date")
        description = (it.get("description") or "").strip()

        if _already_exists(user_id, class_id, title, due_date_iso):
            skipped += 1
            skipped_duplicate += 1
            continue

        payload = {
            "id": new_uuid(),
            "class_id": class_id,
            "user_id": user_id,
            "title": title,
            "description": description or None,
            "due_date": due_date_iso,
            "points": None,
            "completed": False,
            "source": "canvas_ics",
        }

        try:
            supabase.table("assignments").insert(payload).execute()
            created += 1
        except Exception as ex:
            skipped += 1
            if len(insert_errors_sample) < 5:
                insert_errors_sample.append(str(ex))

    return {
        "success": True,
        "assignments_created": created,
        "assignments_skipped": skipped,
        "matched_classes": sorted(list(matched_class_names)),
        "debug": {
            "total_vevents": total_vevents,
            "filtered_out_non_assignments": filtered_out,
            "assignment_like_items": len(items),
            "skipped_no_match": skipped_no_match,
            "skipped_duplicate": skipped_duplicate,
            "llm_used": llm_used,
            "llm_error": llm_error,
            "insert_errors_sample": insert_errors_sample,
            "timezone_fix": "DATE-only DTSTART stored as noon America/Chicago",
        },
    }


@router.get("/assignments")
def list_assignments(
    class_id: Optional[str] = None,
    user_id: Optional[str] = Depends(user_id_from_auth_header),
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    q = (
        supabase.table("assignments")
        .select("*")
        .eq("user_id", user_id)
        .order("due_date", desc=False)
    )
    if class_id:
        q = q.eq("class_id", class_id)

    res = q.execute()
    return res.data or []