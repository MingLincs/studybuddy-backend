# app/services/graph_intelligence.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from ..supabase import supabase
from .db import new_uuid


def _now() -> str:
    return datetime.utcnow().isoformat()


def _safe_data(res):
    """
    Supabase python returns objects with `.data`.
    Sometimes res can be None if something crashed mid-chain.
    """
    if res is None:
        return None
    return getattr(res, "data", None)


def _normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


@dataclass
class GraphTuning:
    # If an edge weight is below this, delete it
    prune_edge_weight_lt: int = 2

    # Limit how many "related" edges we create per upload (keep it clean)
    max_related_edges_per_upload: int = 25

    # Scaling for degree contribution in importance scoring
    degree_scale: float = 25.0

    # Balance between "doc frequency" and "degree"
    importance_doc_weight: float = 0.60
    importance_degree_weight: float = 0.40


DEFAULT_TUNING = GraphTuning()


# ------------------------------------------------------------
# Core: called after each upload
# ------------------------------------------------------------
def reinforce_graph_after_upload(
    *,
    class_id: str,
    doc_id: str,
    concept_ids: List[str],
    tuning: GraphTuning = DEFAULT_TUNING,
) -> None:
    """
    Makes the graph "self-improving" after each upload:
    1) co-occurrence reinforcement ("related" edges)
    2) prune weak edges
    3) recalc concept importance via centrality
    """
    if not class_id or not doc_id or not concept_ids:
        return

    # 1) reinforce co-occurrence (related edges)
    _reinforce_related_edges(class_id=class_id, concept_ids=concept_ids, tuning=tuning)

    # 2) prune weak edges
    prune_weak_edges(class_id=class_id, tuning=tuning)

    # 3) importance scoring
    recalc_importance(class_id=class_id, tuning=tuning)


# ------------------------------------------------------------
# Co-occurrence edges (stored as edge_type "related")
# Your enum supports: prereq, related, part_of, example_of, causes
# ------------------------------------------------------------
def _reinforce_related_edges(*, class_id: str, concept_ids: List[str], tuning: GraphTuning) -> None:
    # Dedup concept ids
    ids = []
    seen = set()
    for cid in concept_ids:
        if cid and cid not in seen:
            ids.append(cid)
            seen.add(cid)

    if len(ids) < 2:
        return

    # Build candidate pairs (cap to avoid huge spam)
    pairs = list(combinations(ids, 2))
    if len(pairs) > tuning.max_related_edges_per_upload:
        pairs = pairs[: tuning.max_related_edges_per_upload]

    for a, b in pairs:
        _upsert_edge(
            class_id=class_id,
            from_id=a,
            to_id=b,
            edge_type="related",
            delta_weight=1,
        )
        # Also reinforce the reverse direction (so UI can treat it like undirected)
        _upsert_edge(
            class_id=class_id,
            from_id=b,
            to_id=a,
            edge_type="related",
            delta_weight=1,
        )


def _upsert_edge(*, class_id: str, from_id: str, to_id: str, edge_type: str, delta_weight: int) -> None:
    """
    If edge exists: weight += delta_weight
    Else: insert with weight = delta_weight
    """
    res = (
        supabase.table("concept_edges")
        .select("id, weight")
        .eq("class_id", class_id)
        .eq("from_concept_id", from_id)
        .eq("to_concept_id", to_id)
        .eq("type", edge_type)
        .maybe_single()
        .execute()
    )
    row = _safe_data(res)

    if row:
        new_w = int(row.get("weight") or 0) + int(delta_weight)
        supabase.table("concept_edges").update(
            {"weight": new_w, "updated_at": _now()}
        ).eq("id", row["id"]).execute()
        return

    supabase.table("concept_edges").insert(
        {
            "id": new_uuid(),
            "class_id": class_id,
            "from_concept_id": from_id,
            "to_concept_id": to_id,
            "type": edge_type,  # must be valid enum value
            "weight": int(delta_weight),
            "created_at": _now(),
            "updated_at": _now(),
        }
    ).execute()


# ------------------------------------------------------------
# Pruning
# ------------------------------------------------------------
def prune_weak_edges(*, class_id: str, tuning: GraphTuning = DEFAULT_TUNING) -> None:
    """
    Deletes edges that are too weak.
    Keeps the graph from turning into junk after many uploads.
    """
    # Delete only non-prereq weak edges (we keep prereq longer)
    # You can tune this later.
    threshold = tuning.prune_edge_weight_lt

    # We only prune weak *related* edges.
    # Typed edges like part_of / example_of / causes / supports / contrasts are
    # already sparse and meaningful; pruning them aggressively makes the graph
    # lose structure.
    supabase.table("concept_edges").delete() \
        .eq("class_id", class_id) \
        .eq("type", "related") \
        .lt("weight", threshold) \
        .execute()


# ------------------------------------------------------------
# Importance scoring (centrality + document frequency)
# ------------------------------------------------------------
def recalc_importance(*, class_id: str, tuning: GraphTuning = DEFAULT_TUNING) -> None:
    """
    importance_score = 0.6*(doc_frequency normalized) + 0.4*(degree normalized)

    If your schema doesn’t have document_frequency, we fall back safely.
    """
    # Fetch concepts
    cres = (
        supabase.table("concepts")
        .select("id, document_frequency")
        .eq("class_id", class_id)
        .execute()
    )
    concepts = _safe_data(cres) or []
    if not concepts:
        return

    # Normalize doc frequency
    doc_freqs = []
    for c in concepts:
        df = c.get("document_frequency")
        try:
            df = int(df) if df is not None else 1
        except Exception:
            df = 1
        doc_freqs.append(df)

    max_df = max(doc_freqs) if doc_freqs else 1

    # Compute weighted degree for each concept
    # degree = sum(weights of edges touching node)
    for idx, c in enumerate(concepts):
        cid = c["id"]

        eres = (
            supabase.table("concept_edges")
            .select("weight")
            .eq("class_id", class_id)
            .or_(f"from_concept_id.eq.{cid},to_concept_id.eq.{cid}")
            .execute()
        )
        edges = _safe_data(eres) or []
        degree = 0.0
        for e in edges:
            try:
                degree += float(e.get("weight") or 0.0)
            except Exception:
                pass

        df = doc_freqs[idx]
        norm_df = df / max_df if max_df else 0.0
        norm_degree = min(1.0, degree / tuning.degree_scale)

        score = (
            tuning.importance_doc_weight * norm_df
            + tuning.importance_degree_weight * norm_degree
        )

        supabase.table("concepts").update(
            {"importance_score": round(float(score), 4), "updated_at": _now()}
        ).eq("id", cid).execute()