from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..supabase import supabase
from .llm import llm


CONCEPT_ENRICH_PROMPT = """You are helping a student study a university course.

Write a concise but high-quality explanation for a SINGLE concept, using plain language.

Return ONLY valid JSON:
{
  "definition": "...",
  "example": "...",
  "application": "..."
}

Rules:
- definition: 2-5 sentences, clear and precise
- example: a concrete, specific scenario (not abstract)
- application: how/when this shows up in real systems or code (practical)
- Avoid fluff. Avoid disclaimers.
"""

EDGE_ENRICH_PROMPT = """You are building a knowledge graph for a university course.

Explain a SINGLE relationship between two concepts.

Return ONLY valid JSON:
{
  "label": "...",
  "definition": "...",
  "example": "...",
  "application": "..."
}

Rules:
- label: 2-6 words, a human-readable verb phrase describing the relationship
- definition: explain how they connect, 2-6 sentences
- example: concrete scenario showing both concepts
- application: when this relationship matters in practice or code
- Be faithful to typical CS/engineering meanings; do not invent nonsense.
"""


def _safe_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        # Sometimes the model returns extra text; try to extract the first JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}
        return {}


async def generate_concept_enrichment(*, concept_name: str, class_name: Optional[str] = None, top_context: Optional[list[str]] = None) -> Dict[str, str]:
    context_bits = []
    if class_name:
        context_bits.append(f"Class: {class_name}")
    if top_context:
        context_bits.append("Related class concepts: " + ", ".join(top_context[:10]))
    context = "\n".join(context_bits).strip()

    user_msg = f"Concept: {concept_name}"
    if context:
        user_msg += "\n" + context

    raw = await llm(
        [
            {"role": "system", "content": CONCEPT_ENRICH_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=450,
        temperature=0.2,
    )
    data = _safe_json(raw)
    return {
        "definition": (data.get("definition") or "").strip(),
        "example": (data.get("example") or "").strip(),
        "application": (data.get("application") or "").strip(),
    }


async def generate_edge_enrichment(*, from_name: str, to_name: str, relation_type: str, class_name: Optional[str] = None) -> Dict[str, str]:
    context_bits = []
    if class_name:
        context_bits.append(f"Class: {class_name}")
    context = "\n".join(context_bits).strip()

    user_msg = f"From: {from_name}\nTo: {to_name}\nRelation type: {relation_type}"
    if context:
        user_msg += "\n" + context

    raw = await llm(
        [
            {"role": "system", "content": EDGE_ENRICH_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=550,
        temperature=0.2,
    )
    data = _safe_json(raw)
    return {
        "label": (data.get("label") or "").strip(),
        "definition": (data.get("definition") or "").strip(),
        "example": (data.get("example") or "").strip(),
        "application": (data.get("application") or "").strip(),
    }


def get_class_name(class_id: str) -> Optional[str]:
    res = supabase.table("classes").select("name").eq("id", class_id).maybe_single().execute()
    if res and getattr(res, "data", None):
        return res.data.get("name")
    return None


def get_top_concepts(class_id: str, limit: int = 10) -> list[str]:
    res = (
        supabase.table("concepts")
        .select("canonical_name, importance_score")
        .eq("class_id", class_id)
        .is_("merged_into", "null")
        .order("importance_score", desc=True)
        .limit(limit)
        .execute()
    )
    rows = (res.data or []) if res else []
    return [r.get("canonical_name") for r in rows if r.get("canonical_name")]
