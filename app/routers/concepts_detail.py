from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import user_id_from_auth_header
from ..supabase import supabase
from ..services.explain import generate_concept_enrichment, get_class_name, get_top_concepts

router = APIRouter(prefix="/concepts", tags=["concepts"])


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


@router.get("/{concept_id}")
def get_concept_detail(
    concept_id: str,
    user_id: str = Depends(user_id_from_auth_header),
):
    concept_res = (
        supabase.table("concepts")
        .select(
            "id, class_id, canonical_name, definition, example, application, importance_score, difficulty_level, document_frequency, merged_into"
        )
        .eq("id", concept_id)
        .maybe_single()
        .execute()
    )

    if not concept_res.data:
        raise HTTPException(status_code=404, detail="Concept not found")

    concept = concept_res.data
    if concept.get("merged_into") is not None:
        # If it's merged, redirect caller to canonical node
        raise HTTPException(status_code=409, detail={"merged_into": concept["merged_into"]})

    class_id = concept["class_id"]
    _require_owner(class_id, user_id)

    # Fetch edges connected to this concept
    edges_res = (
        supabase.table("concept_edges")
        .select("id, from_concept_id, to_concept_id, type, label, weight, confidence, definition, example, application")
        .eq("class_id", class_id)
        .or_(f"from_concept_id.eq.{concept_id},to_concept_id.eq.{concept_id}")
        .execute()
    )
    edge_rows = edges_res.data or []

    # Fetch neighbor concept labels
    neighbor_ids = set()
    for e in edge_rows:
        if e["from_concept_id"] != concept_id:
            neighbor_ids.add(e["from_concept_id"])
        if e["to_concept_id"] != concept_id:
            neighbor_ids.add(e["to_concept_id"])

    neighbors = {}
    if neighbor_ids:
        nres = (
            supabase.table("concepts")
            .select("id, canonical_name, merged_into")
            .in_("id", list(neighbor_ids))
            .execute()
        )
        for r in (nres.data or []):
            # Hide merged nodes
            if r.get("merged_into") is None:
                neighbors[r["id"]] = r.get("canonical_name")

    # Group connections by type and direction
    grouped: dict[str, dict[str, list[dict]]] = {}
    for e in edge_rows:
        rel_type = e.get("type") or "related"
        direction = "outgoing" if e["from_concept_id"] == concept_id else "incoming"
        other_id = e["to_concept_id"] if direction == "outgoing" else e["from_concept_id"]
        other_label = neighbors.get(other_id, "Unknown")

        grouped.setdefault(rel_type, {"outgoing": [], "incoming": []})
        grouped[rel_type][direction].append(
            {
                "edge_id": e["id"],
                "other_concept_id": other_id,
                "other_label": other_label,
                "label": e.get("label") or rel_type,
                "weight": e.get("weight"),
                "confidence": e.get("confidence"),
                "has_details": bool(e.get("definition") or e.get("example") or e.get("application")),
            }
        )

    return {
        "concept": {
            "id": concept["id"],
            "class_id": class_id,
            "name": concept.get("canonical_name"),
            "definition": concept.get("definition"),
            "example": concept.get("example"),
            "application": concept.get("application"),
            "importance_score": concept.get("importance_score"),
            "difficulty_level": concept.get("difficulty_level"),
            "document_frequency": concept.get("document_frequency"),
            "has_details": bool(concept.get("definition") or concept.get("example") or concept.get("application")),
        },
        "connections": grouped,
    }


@router.post("/{concept_id}/generate")
async def generate_concept_detail(
    concept_id: str,
    body: GenerateRequest,
    user_id: str = Depends(user_id_from_auth_header),
):
    concept_res = (
        supabase.table("concepts")
        .select("id, class_id, canonical_name, definition, example, application, merged_into")
        .eq("id", concept_id)
        .maybe_single()
        .execute()
    )
    if not concept_res.data:
        raise HTTPException(status_code=404, detail="Concept not found")

    concept = concept_res.data
    if concept.get("merged_into") is not None:
        raise HTTPException(status_code=409, detail={"merged_into": concept["merged_into"]})

    class_id = concept["class_id"]
    _require_owner(class_id, user_id)

    # If already present and not forcing, return existing
    if not body.force and (concept.get("definition") or concept.get("example") or concept.get("application")):
        return {"ok": True, "concept_id": concept_id, "generated": False}

    class_name = get_class_name(class_id)
    top = get_top_concepts(class_id, limit=10)

    enrich = await generate_concept_enrichment(
        concept_name=concept.get("canonical_name") or "",
        class_name=class_name,
        top_context=top,
    )

    supabase.table("concepts").update(
        {
            "definition": enrich["definition"],
            "example": enrich["example"],
            "application": enrich["application"],
            "updated_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
    ).eq("id", concept_id).execute()

    return {"ok": True, "concept_id": concept_id, "generated": True, "data": enrich}
