from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..auth import user_id_from_auth_header
from ..supabase import supabase

router = APIRouter(prefix="/classes", tags=["concepts"])

class MergeRequest(BaseModel):
    keep_concept_id: str
    merge_concept_id: str
    reason: str | None = None

@router.post("/{class_id}/concepts/merge")
def merge_concepts(class_id: str, payload: MergeRequest, user_id: str = Depends(user_id_from_auth_header)):
    if payload.keep_concept_id == payload.merge_concept_id:
        raise HTTPException(400, "Can't merge concept into itself")

    # mark merge_concept as merged_into keep_concept
    supabase.table("concepts").update({"merged_into": payload.keep_concept_id}).eq("id", payload.merge_concept_id).execute()

    # move aliases
    aliases = supabase.table("concept_aliases").select("*").eq("concept_id", payload.merge_concept_id).execute().data
    for a in aliases:
        supabase.table("concept_aliases").update({"concept_id": payload.keep_concept_id}).eq("id", a["id"]).execute()

    # move mentions
    supabase.table("concept_doc_mentions").update({"concept_id": payload.keep_concept_id}) \
        .eq("concept_id", payload.merge_concept_id).execute()

    # note: edges migration can be done more carefully later (avoid duplicate unique collisions)
    # simplest: leave old edges; graph endpoint hides merged nodes anyway

    return {"ok": True}