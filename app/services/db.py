from __future__ import annotations

import uuid
from typing import Optional, Any

from supabase import create_client, Client
from ..settings import settings


_supabase: Client | None = None


# ------------------------------------------------
# Supabase Client
# ------------------------------------------------

def supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("Supabase is not configured.")
        _supabase = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY,
        )
    return _supabase


def _bucket() -> str:
    return (settings.SUPABASE_STORAGE_BUCKET or "documents").strip()


# ------------------------------------------------
# Storage
# ------------------------------------------------

def upload_pdf_to_storage(
    *,
    user_id: str,
    doc_id: str,
    raw_pdf: bytes,
    filename: str,
) -> str:
    sb = supabase()

    safe_name = (filename or "document.pdf").replace("\\", "_").replace("/", "_")
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    # Store per-user so objects are isolated
    object_path = f"{user_id}/{doc_id}/{safe_name}"

    sb.storage.from_(_bucket()).upload(
        object_path,
        raw_pdf,
        {
            "content-type": "application/pdf",
            "upsert": "true",
        },
    )

    return object_path


def create_signed_download_url(
    *,
    object_path: str,
    ttl_seconds: Optional[int] = None,
) -> str:
    sb = supabase()
    ttl = int(ttl_seconds or settings.SUPABASE_SIGNED_URL_TTL_SECONDS or 600)

    res = sb.storage.from_(_bucket()).create_signed_url(object_path, ttl)
    data = getattr(res, "data", None) or res

    signed = data.get("signedURL") or data.get("signedUrl") or data.get("signed_url")
    if not signed:
        raise RuntimeError("Failed to create signed URL")

    if signed.startswith("http"):
        return signed

    return f"{settings.SUPABASE_URL}{signed}"


def delete_storage_object(*, object_path: str) -> None:
    sb = supabase()
    sb.storage.from_(_bucket()).remove([object_path])


# ------------------------------------------------
# Documents
# ------------------------------------------------

def find_document_id_by_hash(
    *,
    user_id: str,
    content_hash: str,
) -> Optional[str]:
    """Used to avoid duplicate uploads for a given user."""
    sb = supabase()

    try:
        r = (
            sb.table("documents")
            .select("id")
            .eq("user_id", user_id)
            .eq("content_hash", content_hash)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        rows = getattr(r, "data", None) or []
        if rows:
            return rows[0].get("id")
    except Exception:
        return None

    return None


def upsert_document(
    *,
    user_id: str,
    doc_id: str,
    class_id: str | None = None,
    title: str,
    summary: str | None = None,
    cards_json: str | None = None,
    guide_json: str | None = None,
    pdf_path: str | None = None,
    content_hash: str | None = None,
) -> None:
    sb = supabase()

    row: dict[str, Any] = {
        "id": doc_id,
        "user_id": user_id,
        "class_id": class_id,
        "title": title,
        "summary": summary,
        "cards_json": cards_json,
        "guide_json": guide_json,
        "pdf_path": pdf_path,
        "content_hash": content_hash,
    }

    # remove None values
    row = {k: v for k, v in row.items() if v is not None}

    sb.table("documents").upsert(row, on_conflict="id").execute()


# ------------------------------------------------
# Quizzes
# ------------------------------------------------

def insert_quiz(
    *,
    user_id: str,
    doc_id: str,
    class_id: str | None = None,
    title: str,
    quiz_json: str,
    num_questions: int,
) -> None:
    sb = supabase()

    sb.table("quizzes").insert(
        {
            "doc_id": doc_id,
            "user_id": user_id,
            "class_id": class_id,
            "title": title,
            "quiz_json": quiz_json,
            "num_questions": num_questions,
        }
    ).execute()


# ------------------------------------------------
# Utility
# ------------------------------------------------

def new_uuid() -> str:
    return str(uuid.uuid4())
