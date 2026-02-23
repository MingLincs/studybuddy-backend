from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ..auth import user_id_from_auth_header
from ..supabase import supabase

router = APIRouter(prefix="/classes", tags=["documents"])

class DocumentCreate(BaseModel):
    title: str
    storage_path: str | None = None
    raw_text_hash: str | None = None

@router.post("/{class_id}/documents")
def create_document(class_id: str, payload: DocumentCreate, user_id: str = Depends(user_id_from_auth_header)):
    # 1) create document row
    doc_res = supabase.table("documents").insert({
        "class_id": class_id,
        "user_id": user_id,
        "title": payload.title,
        "storage_path": payload.storage_path,
        "raw_text_hash": payload.raw_text_hash,
    }).execute()
    doc = doc_res.data[0]

    # 2) enqueue graph job
    job_res = supabase.table("graph_jobs").insert({
        "class_id": class_id,
        "document_id": doc["id"],
        "user_id": user_id,
        "status": "queued",
        "stage": "extract",
        "payload": {}
    }).execute()

    return {"document": doc, "job": job_res.data[0]}