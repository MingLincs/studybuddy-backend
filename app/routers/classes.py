from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import UUID

from ..auth import user_id_from_auth_header
from ..supabase import supabase
from ..services.db import new_uuid

router = APIRouter(prefix="/classes", tags=["classes"])


# -----------------------
# Models
# -----------------------

class ClassCreate(BaseModel):
    name: str


# -----------------------
# List Classes
# -----------------------

@router.get("")
def list_classes(user_id: Optional[str] = Depends(user_id_from_auth_header)):

    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    res = (
        supabase
        .table("classes")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return res.data or []


# -----------------------
# Create Class
# -----------------------

@router.post("")
def create_class(payload: ClassCreate, user_id: Optional[str] = Depends(user_id_from_auth_header)):

    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Class name required")

    res = (
        supabase
        .table("classes")
        .insert({
            "id": new_uuid(),
            "user_id": user_id,
            "name": name,
        })
        .execute()
    )

    return res.data[0]


# -----------------------
# Delete Class (FULL CASCADE)
# -----------------------

@router.delete("/{class_id}")
def delete_class(class_id: str, user_id: Optional[str] = Depends(user_id_from_auth_header)):

    if not user_id:
        raise HTTPException(status_code=401, detail="Login required")

    try:
        UUID(class_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid class id")

    # Verify ownership first
    check = (
        supabase
        .table("classes")
        .select("id")
        .eq("id", class_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    if not check.data:
        raise HTTPException(status_code=404, detail="Class not found")

    # Delete everything related in correct order
    supabase.table("concept_doc_mentions").delete().eq("class_id", class_id).execute()
    supabase.table("concept_edges").delete().eq("class_id", class_id).execute()
    supabase.table("concepts").delete().eq("class_id", class_id).execute()
    supabase.table("documents").delete().eq("class_id", class_id).execute()
    supabase.table("classes").delete().eq("id", class_id).execute()

    return {"success": True}