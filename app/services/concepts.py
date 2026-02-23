from __future__ import annotations

import json
from typing import Any

from .llm import llm
from .db import supabase, new_uuid


CONCEPT_SYS = """
You are extracting key study concepts from a textbook chapter.
Return ONLY valid JSON:
{
  "concepts": [
    {
      "name": "...",
      "importance": "core|important|advanced",
      "difficulty": "easy|medium|hard",
      "prerequisites": ["..."]
    }
  ]
}
Rules:
- 6 to 12 concepts
- prerequisites should reference other concepts in the list when possible
- keep names short (1-4 words)
- stay faithful to the provided text
"""


# -----------------------------------------
# LLM Extraction
# -----------------------------------------

async def extract_concepts(text: str, max_concepts: int = 10) -> list[dict[str, Any]]:
    raw = await llm(
        [
            {"role": "system", "content": CONCEPT_SYS},
            {"role": "user", "content": text[:20000]},
        ],
        max_tokens=2000,
        temperature=0.2,
    )

    try:
        data = json.loads(raw)
        concepts = data.get("concepts", [])
        out: list[dict[str, Any]] = []

        for c in concepts:
            name = str(c.get("name", "")).strip()
            if not name:
                continue

            out.append(
                {
                    "name": name,
                    "importance": c.get("importance", "important"),
                    "difficulty": c.get("difficulty", "medium"),
                    "prerequisites": c.get("prerequisites", []),
                }
            )

        return out[:max_concepts]
    except Exception:
        return []


# -----------------------------------------
# Graph Engine
# -----------------------------------------

def importance_to_score(level: str) -> float:
    return {
        "core": 0.9,
        "important": 0.6,
        "advanced": 0.3,
    }.get(level, 0.5)


def difficulty_to_score(level: str) -> float:
    return {
        "easy": 0.3,
        "medium": 0.6,
        "hard": 0.9,
    }.get(level, 0.5)


def upsert_concept(class_id: str, c: dict[str, Any]) -> str:
    sb = supabase()

    row = {
        "class_id": class_id,
        "canonical_name": c["name"].strip(),
        "importance_score": importance_to_score(c.get("importance", "important")),
        "difficulty_level": difficulty_to_score(c.get("difficulty", "medium")),
    }

    r = sb.table("concepts").upsert(
        row,
        on_conflict="class_id,canonical_name",
    ).execute()

    data = r.data or []
    if data:
        return data[0]["id"]

    # fallback fetch
    rr = (
        sb.table("concepts")
        .select("id")
        .eq("class_id", class_id)
        .eq("canonical_name", c["name"].strip())
        .limit(1)
        .execute()
    )
    rows = rr.data or []
    return rows[0]["id"] if rows else ""


def add_edge(class_id: str, from_id: str, to_id: str) -> None:
    if not from_id or not to_id or from_id == to_id:
        return

    sb = supabase()

    row = {
        "class_id": class_id,
        "from_concept_id": from_id,
        "to_concept_id": to_id,
        "type": "prereq",
        "weight": 1.0,
    }

    sb.table("concept_edges").upsert(
        row,
        on_conflict="class_id,from_concept_id,to_concept_id,type",
    ).execute()


def save_doc_mentions(class_id: str, doc_id: str, concept_ids: list[str]) -> None:
    sb = supabase()

    for cid in concept_ids:
        row = {
            "class_id": class_id,
            "document_id": doc_id,
            "concept_id": cid,
            "mention_count": 1,
        }

        sb.table("concept_doc_mentions").upsert(
            row,
            on_conflict="document_id,concept_id",
        ).execute()


def update_class_graph(user_id: str, class_id: str, doc_id: str, concepts: list[dict[str, Any]]) -> None:
    id_by_name: dict[str, str] = {}

    # 1. Upsert concept nodes
    for c in concepts:
        cid = upsert_concept(class_id, c)
        id_by_name[c["name"].strip()] = cid

    # 2. Save document mentions
    save_doc_mentions(class_id, doc_id, list(id_by_name.values()))

    # 3. Add prerequisite edges
    for c in concepts:
        to_id = id_by_name.get(c["name"].strip(), "")
        for p in c.get("prerequisites", []):
            from_id = id_by_name.get(str(p).strip(), "")
            add_edge(class_id, from_id, to_id)