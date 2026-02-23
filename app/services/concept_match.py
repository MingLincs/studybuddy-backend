from __future__ import annotations
from typing import Dict, Any
from ..supabase import supabase

def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def match_or_create_concepts(class_id: str, document_id: str, extracted: Dict[str, Any]) -> Dict[str, str]:
    concepts = extracted.get("concepts", [])
    name_to_id: Dict[str, str] = {}

    # preload aliases for fast match
    alias_rows = supabase.table("concept_aliases").select("concept_id,normalized_alias").eq("class_id", class_id).execute().data
    alias_map = {r["normalized_alias"]: r["concept_id"] for r in alias_rows}

    for c in concepts:
        name = c["name"].strip()
        desc = c.get("desc")
        n = norm(name)

        concept_id = alias_map.get(n)

        # if no alias match, try exact canonical name
        if not concept_id:
            existing = supabase.table("concepts") \
                .select("id") \
                .eq("class_id", class_id) \
                .eq("canonical_name", name) \
                .execute().data
            if existing:
                concept_id = existing[0]["id"]

        # create new concept if still not found
        if not concept_id:
            ins = supabase.table("concepts").insert({
                "class_id": class_id,
                "canonical_name": name,
                "canonical_description": desc
            }).execute().data[0]
            concept_id = ins["id"]

            # create alias
            supabase.table("concept_aliases").insert({
                "class_id": class_id,
                "concept_id": concept_id,
                "alias": name,
                "normalized_alias": n,
                "confidence": 0.95
            }).execute()
        else:
            # ensure alias exists (harmless if duplicates blocked by unique)
            try:
                supabase.table("concept_aliases").insert({
                    "class_id": class_id,
                    "concept_id": concept_id,
                    "alias": name,
                    "normalized_alias": n,
                    "confidence": 0.8
                }).execute()
            except Exception:
                pass

        name_to_id[name] = concept_id

        # upsert mention evidence
        existing_m = supabase.table("concept_doc_mentions") \
            .select("id,mention_count") \
            .eq("document_id", document_id) \
            .eq("concept_id", concept_id) \
            .execute().data
        if existing_m:
            m = existing_m[0]
            supabase.table("concept_doc_mentions").update({
                "mention_count": int(m["mention_count"]) + 1
            }).eq("id", m["id"]).execute()
        else:
            supabase.table("concept_doc_mentions").insert({
                "class_id": class_id,
                "document_id": document_id,
                "concept_id": concept_id,
                "mention_count": 1,
                "context_snippets": []
            }).execute()

    return name_to_id