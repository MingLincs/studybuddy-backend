from fastapi import APIRouter, Depends, HTTPException
from ..auth import user_id_from_auth_header
from ..supabase import supabase

router = APIRouter(prefix="/classes", tags=["class-admin"])


@router.delete("/{class_id}")
def delete_class(class_id: str, user_id: str = Depends(user_id_from_auth_header)):

    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Verify ownership
    cls = (
        supabase.table("classes")
        .select("id")
        .eq("id", class_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if not cls.data:
        raise HTTPException(status_code=404, detail="Class not found")

    # -------------------------
    # Delete in correct order
    # -------------------------

    # 1. Delete concept_doc_mentions
    supabase.table("concept_doc_mentions") \
        .delete() \
        .eq("class_id", class_id) \
        .execute()

    # 2. Delete concept_edges
    supabase.table("concept_edges") \
        .delete() \
        .eq("class_id", class_id) \
        .execute()

    # 3. Delete concepts
    supabase.table("concepts") \
        .delete() \
        .eq("class_id", class_id) \
        .execute()

    # 4. Delete documents
    supabase.table("documents") \
        .delete() \
        .eq("class_id", class_id) \
        .execute()

    # 5. Delete class itself
    supabase.table("classes") \
        .delete() \
        .eq("id", class_id) \
        .execute()

    return {"status": "deleted"}