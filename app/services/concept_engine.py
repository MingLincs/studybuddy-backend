# app/services/concept_engine.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

from ..supabase import supabase
from .db import new_uuid
from .llm import llm
from .json_utils import safe_json_loads
from .graph_intelligence import reinforce_graph_after_upload


def _now() -> str:
    return datetime.utcnow().isoformat()


def _safe_data(res):
    if res is None:
        return None
    return getattr(res, "data", None)


RELATION_PROMPT = """
You are building a structured knowledge graph for a university course.

Given a list of concepts, determine meaningful relationships.

Return ONLY valid JSON:

{
  "edges": [
    {
      "from": "Concept A",
      "to": "Concept B",
      "type": "prereq|related|part_of|example_of|causes"
    }
  ]
}

Rules:
- Only include strong logical relationships
- Use direction properly (A prereq B means A must be learned first)
- Do not invent nonsense
- Avoid duplicates
- Max 15 edges
"""


async def extract_relationships(concept_names: List[str]) -> List[Dict]:
    if len(concept_names) < 2:
        return []

    content = "\n".join(f"- {c}" for c in concept_names)

    response = await llm(
        [
            {"role": "system", "content": RELATION_PROMPT},
            {"role": "user", "content": f"Concept list:\n{content}"},
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    try:
        parsed = safe_json_loads(response, default={"edges": []})
        edges = parsed.get("edges", []) if isinstance(parsed, dict) else []
        if not isinstance(edges, list):
            return []
        return edges[:15]
    except Exception:
        # one retry with stricter instruction
        try:
            response2 = await llm(
                [
                    {"role": "system", "content": RELATION_PROMPT + "\n\nIMPORTANT: Output raw JSON only. No markdown, no commentary."},
                    {"role": "user", "content": f"Return ONLY JSON.\n\nConcept list:\n{content}"},
                ],
                max_tokens=1500,
                temperature=0.2,
            )
            parsed2 = safe_json_loads(response2, default={"edges": []})
            edges2 = parsed2.get("edges", []) if isinstance(parsed2, dict) else []
            return edges2[:15] if isinstance(edges2, list) else []
        except Exception:
            return []


async def update_class_graph(*, class_id: str, doc_id: str, guide_json: str) -> None:
    """
    1) Upsert concept nodes into concepts table
    2) Store a mention row (doc -> concept)
    3) Use AI to create prereq/part_of/example_of/causes edges
    4) Call graph intelligence layer:
       - reinforce co-occurrence (related)
       - prune weak edges
       - recalc importance (centrality)
    """

    if not class_id or not doc_id:
        return

    parsed = safe_json_loads(guide_json or "{}", default={})
    if not isinstance(parsed, dict):
        return

    concepts = parsed.get("concepts", [])
    if not isinstance(concepts, list) or not concepts:
        return

    # If guide_json already contains edges (preferred), we'll use them.
    precomputed_edges = parsed.get("edges", []) if isinstance(parsed, dict) else []

    # ------------- Upsert Concepts -------------
    concept_ids: List[str] = []
    name_to_id: Dict[str, str] = {}

    for c in concepts:
        name = (c.get("name") or "").strip()
        if not name:
            continue

        # Try to find existing concept in class (case-insensitive)
        res = (
            supabase.table("concepts")
            .select("id, document_frequency")
            .eq("class_id", class_id)
            .ilike("canonical_name", name)
            .maybe_single()
            .execute()
        )
        existing = _safe_data(res)

        if existing:
            cid = existing["id"]
            df = existing.get("document_frequency")
            try:
                df = int(df) if df is not None else 0
            except Exception:
                df = 0

            supabase.table("concepts").update(
                {"document_frequency": df + 1, "updated_at": _now()}
            ).eq("id", cid).execute()
        else:
            cid = new_uuid()
            supabase.table("concepts").insert(
                {
                    "id": cid,
                    "class_id": class_id,
                    "canonical_name": name,
                    "document_frequency": 1,
                    "importance_score": 0.1,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "merged_into": None,
                }
            ).execute()

        concept_ids.append(cid)
        name_to_id[name.strip().lower()] = cid

        # Mention row: keep it minimal to match your schema (avoid missing columns)
        supabase.table("concept_doc_mentions").insert(
            {
                "id": new_uuid(),
                "class_id": class_id,
                "concept_id": cid,
                "document_id": doc_id,
                "created_at": _now(),
                "updated_at": _now(),
            }
        ).execute()

    if not concept_ids:
        return

    # ------------- AI relationship edges -------------
    concept_names = [((c.get("name") or "").strip()) for c in concepts if (c.get("name") or "").strip()]
    edges = precomputed_edges if isinstance(precomputed_edges, list) and precomputed_edges else await extract_relationships(concept_names)

    allowed_types = {"prereq", "related", "part_of", "example_of", "causes"}

    for e in edges:
        from_name = (e.get("from") or "").strip().lower()
        to_name = (e.get("to") or "").strip().lower()
        etype = (e.get("type") or "").strip()

        if not from_name or not to_name:
            continue
        if etype not in allowed_types:
            continue

        from_id = name_to_id.get(from_name)
        to_id = name_to_id.get(to_name)
        if not from_id or not to_id:
            continue
        if from_id == to_id:
            continue

        # If edge exists, increase weight; else insert
        ex = (
            supabase.table("concept_edges")
            .select("id, weight")
            .eq("class_id", class_id)
            .eq("from_concept_id", from_id)
            .eq("to_concept_id", to_id)
            .eq("type", etype)
            .maybe_single()
            .execute()
        )
        row = _safe_data(ex)

        if row:
            w = int(row.get("weight") or 0) + 1
            supabase.table("concept_edges").update(
                {"weight": w, "updated_at": _now()}
            ).eq("id", row["id"]).execute()
        else:
            supabase.table("concept_edges").insert(
                {
                    "id": new_uuid(),
                    "class_id": class_id,
                    "from_concept_id": from_id,
                    "to_concept_id": to_id,
                    "type": etype,
                    "weight": 1,
                    "created_at": _now(),
                    "updated_at": _now(),
                }
            ).execute()

    # ------------- Graph Intelligence Layer -------------
    reinforce_graph_after_upload(class_id=class_id, doc_id=doc_id, concept_ids=concept_ids)