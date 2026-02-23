from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import user_id_from_auth_header
from ..supabase import supabase
from ..services.explain import generate_edge_enrichment, get_class_name

router = APIRouter(prefix="/edges", tags=["edges"])


class GenerateRequest(BaseModel):
    force: bool = False


def _require_owner(class_id: str, user_id: str) -> None:
    cls_res = (
        supabase.table("classes")
        .select("id")
        .eq("id", class_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if not cls_res.data:
        raise HTTPException(status_code=404, detail="Class not found")


@router.get("/{edge_id}")
def get_edge_detail(
    edge_id: str,
    user_id: str = Depends(user_id_from_auth_header),
):
    edge_res = (
        supabase.table("concept_edges")
        .select(
            "id, class_id, from_concept_id, to_concept_id, type, label, weight, confidence, definition, example, application"
        )
        .eq("id", edge_id)
        .maybe_single()
        .execute()
    )
    if not edge_res.data:
        raise HTTPException(status_code=404, detail="Edge not found")
    edge = edge_res.data

    _require_owner(edge["class_id"], user_id)

    # Fetch concept names
    ids = [edge["from_concept_id"], edge["to_concept_id"]]
    cres = supabase.table("concepts").select("id, canonical_name, merged_into").in_("id", ids).execute()
    names = {c["id"]: c.get("canonical_name") for c in (cres.data or []) if c.get("merged_into") is None}

    return {
        "edge": {
            "id": edge["id"],
            "class_id": edge["class_id"],
            "from_concept_id": edge["from_concept_id"],
            "to_concept_id": edge["to_concept_id"],
            "from_label": names.get(edge["from_concept_id"], "Unknown"),
            "to_label": names.get(edge["to_concept_id"], "Unknown"),
            "relation_type": edge.get("type"),
            "label": edge.get("label") or edge.get("type"),
            "weight": edge.get("weight"),
            "confidence": edge.get("confidence"),
            "definition": edge.get("definition"),
            "example": edge.get("example"),
            "application": edge.get("application"),
            "has_details": bool(edge.get("definition") or edge.get("example") or edge.get("application")),
        }
    }


@router.post("/{edge_id}/generate")
async def generate_edge_detail(
    edge_id: str,
    body: GenerateRequest,
    user_id: str = Depends(user_id_from_auth_header),
):
    edge_res = (
        supabase.table("concept_edges")
        .select(
            "id, class_id, from_concept_id, to_concept_id, type, label, definition, example, application"
        )
        .eq("id", edge_id)
        .maybe_single()
        .execute()
    )
    if not edge_res.data:
        raise HTTPException(status_code=404, detail="Edge not found")
    edge = edge_res.data

    _require_owner(edge["class_id"], user_id)

    if not body.force and (edge.get("definition") or edge.get("example") or edge.get("application")):
        return {"ok": True, "edge_id": edge_id, "generated": False}

    # Names
    ids = [edge["from_concept_id"], edge["to_concept_id"]]
    cres = supabase.table("concepts").select("id, canonical_name, merged_into").in_("id", ids).execute()
    names = {c["id"]: c.get("canonical_name") for c in (cres.data or []) if c.get("merged_into") is None}

    class_name = get_class_name(edge["class_id"])

    enrich = await generate_edge_enrichment(
        from_name=names.get(edge["from_concept_id"], "Concept A"),
        to_name=names.get(edge["to_concept_id"], "Concept B"),
        relation_type=edge.get("type") or "related",
        class_name=class_name,
    )

    supabase.table("concept_edges").update(
        {
            "label": enrich["label"] or (edge.get("type") or "related"),
            "definition": enrich["definition"],
            "example": enrich["example"],
            "application": enrich["application"],
            "updated_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
    ).eq("id", edge_id).execute()

    return {"ok": True, "edge_id": edge_id, "generated": True, "data": enrich}
