from fastapi import APIRouter, HTTPException, Request

from ..auth import user_id_from_auth_header
from ..services.db import supabase

router = APIRouter()


@router.get("/classes/{class_id}/concept-map")
def get_concept_map(class_id: str, request: Request):
    user_id = user_id_from_auth_header(request.headers.get("Authorization"))
    if not user_id:
        raise HTTPException(401, "Unauthorized")

    sb = supabase()

    # Verify ownership
    cls = (
        sb.table("classes")
        .select("id")
        .eq("id", class_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if not cls.data:
        raise HTTPException(404, "Class not found")

    # Fetch concepts
    concepts_res = (
        sb.table("concepts")
        .select("id, canonical_name, importance_score, difficulty_level, merged_into")
        .eq("class_id", class_id)
        .is_("merged_into", None)
        .execute()
    )
    concepts = concepts_res.data or []

    # Fetch edges (IMPORTANT: include label/evidence/confidence)
    edges_res = (
        sb.table("concept_edges")
        .select("from_concept_id, to_concept_id, type, label, weight, confidence, evidence")
        .eq("class_id", class_id)
        .execute()
    )
    edges = edges_res.data or []

    # Transform nodes
    nodes = []
    for c in concepts:
        imp = float(c.get("importance_score") or 0.0)
        diff = float(c.get("difficulty_level") or 0.0)

        nodes.append(
            {
                "id": c["id"],
                "label": c["canonical_name"],
                "importance": "core" if imp > 0.8 else "important" if imp > 0.5 else "advanced",
                "difficulty": "hard" if diff > 0.8 else "medium" if diff > 0.5 else "easy",
            }
        )

    # Transform edges
    formatted_edges = []
    for e in edges:
        edge_type = (e.get("type") or "related").strip()
        edge_label = (e.get("label") or "").strip()

        # Backward compat: keep "reason" but make it meaningful.
        # Prefer label; if missing, fall back to type.
        reason = edge_label if edge_label else edge_type

        formatted_edges.append(
            {
                "from": e["from_concept_id"],
                "to": e["to_concept_id"],
                "type": edge_type,          # coarse type (for styling)
                "label": edge_label,        # rich meaning (what you actually want)
                "reason": reason,           # legacy field used by old frontend
                "weight": e.get("weight") or 1,
                "confidence": e.get("confidence"),
                "evidence": e.get("evidence") or [],
            }
        )

    return {"nodes": nodes, "edges": formatted_edges}