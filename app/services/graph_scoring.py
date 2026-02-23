from __future__ import annotations
from collections import defaultdict
from ..supabase import supabase

def recompute_importance(class_id: str) -> None:
    # mentions
    mentions = supabase.table("concept_doc_mentions") \
        .select("concept_id,mention_count") \
        .eq("class_id", class_id).execute().data

    mention_sum = defaultdict(int)
    for m in mentions:
        mention_sum[m["concept_id"]] += int(m["mention_count"])

    # degree
    edges = supabase.table("concept_edges") \
        .select("from_concept_id,to_concept_id") \
        .eq("class_id", class_id).execute().data

    degree = defaultdict(int)
    for e in edges:
        degree[e["from_concept_id"]] += 1
        degree[e["to_concept_id"]] += 1

    # normalize
    max_mentions = max(mention_sum.values()) if mention_sum else 1
    max_degree = max(degree.values()) if degree else 1

    concepts = supabase.table("concepts").select("id,merged_into").eq("class_id", class_id).execute().data
    for c in concepts:
        if c["merged_into"] is not None:
            continue
        cid = c["id"]
        mscore = mention_sum.get(cid, 0) / max_mentions
        dscore = degree.get(cid, 0) / max_degree

        # weighted blend (tweak later)
        importance = float(0.7 * mscore + 0.3 * dscore)

        supabase.table("concepts").update({"importance_score": importance}).eq("id", cid).execute()