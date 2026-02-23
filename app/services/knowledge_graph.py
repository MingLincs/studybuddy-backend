"""app/services/knowledge_graph.py

End-to-end "high signal" concept + relationship extraction.

Design goals
------------
1) Work across ANY class (STEM + humanities + social science + writing-heavy).
2) Be resilient to LLM formatting issues (use safe_json_loads).
3) Produce important nodes only:
   - LLM proposes a larger set of candidates with importance scores.
   - Server enforces max size, minimum score, deduping.
4) Produce sane edges that actually connect:
   - Edges have coarse type (DB-safe) + fine label + evidence + confidence.
   - Server snaps edge endpoints to canonical kept node names.
   - Server validates evidence snippets appear in the text (cheap anti-hallucination).
   - Validation pass downgrades speculative prereqs to related (instead of deleting everything).

Output format is compatible with your existing UI + concept_engine:

{
  "concepts": [...],
  "edges": [
     {
       "from": "...",
       "to": "...",
       "type": "prereq|related|part_of|example_of|causes",
       "label": "defines|applies_to|contrasts_with|...",
       "strength": 1..5,
       "confidence": 0..1,
       "evidence": ["short phrase from text", ...]
     }
  ],
  "meta": {...}
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm import llm
from .json_utils import safe_json_loads


# -----------------------------
# Routing (choose extractor "mode")
# -----------------------------

ROUTER_PROMPT = r"""
You are classifying a document for study extraction.

Pick ONE extraction_mode:
- "stem" (math/cs/physics/engineering)
- "humanities" (history/literature/philosophy)
- "social_science" (psych/soc/econ/poli-sci)
- "writing" (composition/english writing/rhetoric)
- "mixed" (interdisciplinary, unclear)

Also determine doc_type:
- "syllabus" if it looks like course policies/schedule/grade breakdown
- otherwise "notes"

Return ONLY valid JSON:
{
  "extraction_mode": "stem|humanities|social_science|writing|mixed",
  "doc_type": "syllabus|notes",
  "confidence": 0.0,
  "reason": "short"
}
"""


async def route_extraction_mode(text: str) -> Dict:
    resp = await llm(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": text[:3000]},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    data = safe_json_loads(resp, default={})
    if not isinstance(data, dict):
        return {"extraction_mode": "mixed", "doc_type": "notes", "confidence": 0.0, "reason": "parse_failed"}

    mode = (data.get("extraction_mode") or "mixed").strip()
    doc_type = (data.get("doc_type") or "notes").strip()
    if mode not in {"stem", "humanities", "social_science", "writing", "mixed"}:
        mode = "mixed"
    if doc_type not in {"syllabus", "notes"}:
        doc_type = "notes"

    try:
        conf = float(data.get("confidence") or 0.0)
    except Exception:
        conf = 0.0

    return {"extraction_mode": mode, "doc_type": doc_type, "confidence": conf, "reason": data.get("reason") or ""}


# -----------------------------
# Candidate extraction prompts
# -----------------------------

def _candidate_prompt(mode: str) -> str:
    base = r"""
You will propose candidate "learning units" from a document.

Each unit must be something a student can be tested on.
Avoid trivial vocabulary, obvious section headings, or administrative fluff.

Return ONLY valid JSON:
{
  "candidates": [
    {
      "name": "...",              // short label
      "unit_type": "...",         // e.g., formula|method|theme|event|argument|skill|device|framework|policy|process
      "importance": 1,             // 1..5 (5 = essential to pass)
      "difficulty": "easy|medium|hard",
      "simple": "1-2 sentences",
      "detailed": "4-6 sentences",
      "technical": "optional: formalism/structure",
      "example": "specific example (numbers / quote / scenario)",
      "common_mistake": "realistic misunderstanding",
      "evidence": ["short phrases copied from text (<=12 words each)"],
      "prereqs": ["names of other units if clearly needed"]
    }
  ]
}

Rules:
- Propose 18-24 candidates to maximize coverage
- Importance MUST be meaningful (most should be 2-4; only a few 5)
- Evidence must be copied from the text (no invented quotes)
- Don't invent facts not present in the text
"""

    if mode == "stem":
        return base + "\nFocus on: definitions, formulas, algorithms, problem methods, key assumptions."
    if mode == "humanities":
        return base + "\nFocus on: themes, events, people, movements, arguments, primary-source claims."
    if mode == "social_science":
        return base + "\nFocus on: theories, variables, studies, methods, models, interpretations."
    if mode == "writing":
        return base + "\nFocus on: thesis building, evidence use, structure, rhetoric, style, revision strategies."
    return base + "\nFocus on: whatever would be tested in this course."


# NOTE: coarse edge type is DB-safe; label is the real meaning.
REFINE_PROMPT = r"""
You will refine candidate learning units into a final set of core study concepts AND propose meaningful edges.

Return ONLY valid JSON:
{
  "keep": [
    {
      "name": "...",              // must match a candidate name (best effort)
      "why_keep": "short",
      "final_importance": 1        // 1..5
    }
  ],
  "edges": [
    {
      "from": "Unit A",
      "to": "Unit B",

      // coarse type must be one of:
      "type": "prereq|related|part_of|example_of|causes",

      // label is the real meaning (more specific than type)
      // examples: defines, applies_to, derived_from, contrasts_with, motivates, leads_to, supports_claim
      "label": "short_verb_phrase",

      "strength": 1,               // 1..5
      "confidence": 0.0,           // 0..1
      "evidence": ["short phrases copied from the document (<=12 words)"],
      "why": "short"
    }
  ]
}

Rules:
- Keep 8-12 units total
- Create 8-16 edges if possible (avoid isolated nodes)
- Edges must be only between kept units
- Prefer specific relationships: part_of, causes, example_of
- Use prereq only if the document implies learning order (before/after/requires/must know)
- Use related only if label is specific (contrasts_with/influences/supports/etc.)
- Evidence MUST be copied from the text (no invention)
- Max 18 edges
"""


EDGE_VALIDATE_PROMPT = r"""
Validate edges in a course knowledge graph.

Given:
- kept concept names
- proposed edges with evidence snippets (copied from document)

Your job:
- Remove edges only if evidence clearly does NOT support it.
- If ordering is not justified, DOWNGRADE type from prereq -> related (do not drop).
- If type is wrong but relationship exists, fix type/label.

Return ONLY valid JSON:
{
  "edges": [
    {
      "from": "...",
      "to": "...",
      "type": "prereq|related|part_of|example_of|causes",
      "label": "...",
      "strength": 1,
      "confidence": 0.0,
      "evidence": ["..."]
    }
  ]
}

Rules:
- Prefer fewer, higher-quality edges, but try to keep at least 6 edges if possible.
- If unsure, keep as related with low confidence (0.45-0.6) instead of dropping.
- Evidence must remain copied from the document.
"""


# -----------------------------
# Server-side enforcement
# -----------------------------

def _normalize_name(s: str) -> str:
    """
    Strong normalization so concept matching works across:
    - curly quotes/apostrophes (’ “ ”) vs straight ( ' " )
    - punctuation differences
    - whitespace differences
    """
    if not s:
        return ""
    s = s.strip().lower()

    s = (
        s.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    s = re.sub(r"\s+", " ", s)

    # remove punctuation except quotes/hyphen
    s = re.sub(r"[^\w\s'\-]", "", s)

    return s.strip()


def _normalize_text_for_evidence(t: str) -> str:
    if not t:
        return ""
    return (
        t.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )


def _evidence_supported(evidence: List[str], text_norm: str) -> bool:
    """
    Cheap guardrail: at least one evidence snippet should appear in the text (case-insensitive).
    Prevents edges built on invented quotes.
    """
    for s in evidence or []:
        ss = (s or "").strip()
        if not ss:
            continue
        if _normalize_text_for_evidence(ss) in text_norm:
            return True
    return False


@dataclass
class Edge:
    src: str
    dst: str
    typ: str
    label: str
    strength: int
    confidence: float
    evidence: List[str]


DIRECTED_TYPES = {"prereq", "causes"}

# DB-safe coarse types
ALLOWED_EDGE_TYPES = {"prereq", "related", "part_of", "example_of", "causes"}


def _dedupe_candidates(cands: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for c in cands or []:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        key = _normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _pick_top(cands: List[Dict], keep_names: List[str], *, max_nodes: int = 12, min_importance: int = 3) -> List[Dict]:
    """
    Refine might return names with tiny formatting differences.
    We match by normalized form (strong normalization).
    """
    wanted_norm = {_normalize_name(n) for n in (keep_names or []) if n}
    chosen = []
    for c in cands:
        nm = (c.get("name") or "").strip()
        if nm and _normalize_name(nm) in wanted_norm:
            chosen.append(c)

    # If refine returned too many/few, fall back to importance sorting
    if len(chosen) < 6:
        def score(x):
            try:
                return int(x.get("importance") or 0)
            except Exception:
                return 0
        chosen = sorted(cands, key=score, reverse=True)

    # Enforce min_importance and max_nodes
    pruned = []
    for c in chosen:
        try:
            imp = int(c.get("importance") or 0)
        except Exception:
            imp = 0
        if imp < min_importance:
            continue
        pruned.append(c)
        if len(pruned) >= max_nodes:
            break

    return pruned


def _build_edge_list(
    edges_raw: List[Dict],
    kept_norm_to_name: Dict[str, str],
    text_norm: str,
    *,
    max_edges: int = 18,
) -> List[Edge]:
    """
    Important:
    - Snap endpoints to canonical kept names (fixes UI not connecting nodes).
    - Require evidence to appear in the doc text (cheap anti-hallucination).
    """
    dedup = set()
    out: List[Edge] = []

    for e in edges_raw or []:
        if not isinstance(e, dict):
            continue

        raw_src = (e.get("from") or "").strip()
        raw_dst = (e.get("to") or "").strip()
        if not raw_src or not raw_dst:
            continue

        src_n = _normalize_name(raw_src)
        dst_n = _normalize_name(raw_dst)
        if not src_n or not dst_n or src_n == dst_n:
            continue

        # Snap to canonical kept names
        if src_n not in kept_norm_to_name or dst_n not in kept_norm_to_name:
            continue
        src = kept_norm_to_name[src_n]
        dst = kept_norm_to_name[dst_n]

        typ = (e.get("type") or "").strip()
        if typ not in ALLOWED_EDGE_TYPES:
            continue

        label = (e.get("label") or "").strip()
        if not label:
            label = "related_to"
        label = label[:80]

        try:
            strength = int(e.get("strength") or 3)
        except Exception:
            strength = 3
        strength = max(1, min(5, strength))

        try:
            confidence = float(e.get("confidence") or 0.6)
        except Exception:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        evidence = e.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        evidence = [(" ".join(str(x).split())[:200]) for x in evidence if str(x).strip()][:6]

        # Evidence guardrail (do not accept hallucinated edges)
        if evidence and not _evidence_supported(evidence, text_norm):
            continue

        key = (src_n, dst_n, typ, _normalize_name(label))
        if key in dedup:
            continue
        dedup.add(key)

        out.append(Edge(src=src, dst=dst, typ=typ, label=label, strength=strength, confidence=confidence, evidence=evidence))
        if len(out) >= max_edges:
            break

    return out


def _break_cycles(edges: List[Edge]) -> List[Edge]:
    """Break cycles only for directed prerequisite-like edges.

We repeatedly detect a cycle among directed edges and remove the weakest
edge from that cycle.
"""

    def build_adj(es: List[Edge]):
        adj = {}
        for ed in es:
            if ed.typ not in DIRECTED_TYPES:
                continue
            adj.setdefault(_normalize_name(ed.src), []).append(_normalize_name(ed.dst))
        return adj

    def find_cycle(adj):
        visited = set()
        stack = set()
        parent = {}

        def dfs(u):
            visited.add(u)
            stack.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    parent[v] = u
                    cyc = dfs(v)
                    if cyc:
                        return cyc
                elif v in stack:
                    path = [v]
                    cur = u
                    while cur != v and cur in parent:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(v)
                    path.reverse()
                    return path
            stack.remove(u)
            return None

        for node in list(adj.keys()):
            if node not in visited:
                cyc = dfs(node)
                if cyc:
                    return cyc
        return None

    out = list(edges)
    while True:
        adj = build_adj(out)
        cyc = find_cycle(adj)
        if not cyc:
            break

        cycle_edges = []
        for i in range(len(cyc) - 1):
            a = cyc[i]
            b = cyc[i + 1]
            for ed in out:
                if ed.typ in DIRECTED_TYPES and _normalize_name(ed.src) == a and _normalize_name(ed.dst) == b:
                    cycle_edges.append(ed)

        if not cycle_edges:
            break

        weakest = sorted(cycle_edges, key=lambda x: x.strength)[0]
        out = [e for e in out if e is not weakest]

    return out


def _importance_bucket(score_1_to_5: int) -> str:
    if score_1_to_5 >= 5:
        return "core"
    if score_1_to_5 >= 4:
        return "important"
    return "advanced"


async def extract_knowledge_graph(text: str, *, max_nodes: int = 12) -> Dict:
    """Main entry point.

Returns a dict with concepts + edges.
"""
    route = await route_extraction_mode(text)
    mode = route.get("extraction_mode", "mixed")

    text_window = text[:6500]
    text_norm = _normalize_text_for_evidence(text_window)

    # 1) Candidate pass
    cand_prompt = _candidate_prompt(mode)
    cand_resp = await llm(
        [
            {"role": "system", "content": cand_prompt},
            {"role": "user", "content": text_window},
        ],
        max_tokens=3500,
        temperature=0.2,
    )
    cand_data = safe_json_loads(cand_resp, default={"candidates": []})
    cands = cand_data.get("candidates", []) if isinstance(cand_data, dict) else []
    if not isinstance(cands, list):
        cands = []
    cands = _dedupe_candidates(cands)

    # 2) Refine pass (pick top + edges)
    refine_input = {
        "candidates": [
            {
                "name": c.get("name"),
                "importance": c.get("importance"),
                "unit_type": c.get("unit_type"),
                "simple": c.get("simple"),
            }
            for c in cands[:24]
        ]
    }

    refine_resp = await llm(
        [
            {"role": "system", "content": REFINE_PROMPT},
            {"role": "user", "content": str(refine_input)},
        ],
        max_tokens=2000,
        temperature=0.2,
    )
    refine = safe_json_loads(refine_resp, default={"keep": [], "edges": []})

    keep = refine.get("keep", []) if isinstance(refine, dict) else []
    keep_names = [(k.get("name") or "").strip() for k in keep if isinstance(k, dict)]

    selected = _pick_top(cands, keep_names, max_nodes=max_nodes, min_importance=3)
    kept_names = [(c.get("name") or "").strip() for c in selected if (c.get("name") or "").strip()]

    # Build canonical map for endpoint snapping
    kept_norm_to_name = {_normalize_name(n): n for n in kept_names}

    # 3) Build + sanitize edges from refine
    edges_raw = refine.get("edges", []) if isinstance(refine, dict) else []
    edges1 = _build_edge_list(edges_raw, kept_norm_to_name, text_norm, max_edges=18)
    edges1 = _break_cycles(edges1)

    # 4) Validation pass (downgrade instead of deleting)
    edges_final = edges1
    if edges1:
        validate_payload = {
            "kept": kept_names,
            "edges": [
                {
                    "from": e.src,
                    "to": e.dst,
                    "type": e.typ,
                    "label": e.label,
                    "strength": e.strength,
                    "confidence": e.confidence,
                    "evidence": e.evidence,
                }
                for e in edges1[:18]
            ],
        }
        validate_resp = await llm(
            [
                {"role": "system", "content": EDGE_VALIDATE_PROMPT},
                {"role": "user", "content": str(validate_payload)},
            ],
            max_tokens=1600,
            temperature=0.1,
        )
        validated = safe_json_loads(validate_resp, default={"edges": []})
        vraw = validated.get("edges", []) if isinstance(validated, dict) else []
        edges2 = _build_edge_list(vraw, kept_norm_to_name, text_norm, max_edges=18)
        edges2 = _break_cycles(edges2)

        # Prefer validated edges if we still have a non-trivial graph
        if len(edges2) >= 4:
            edges_final = edges2

    # 5) Final shape compatible with UI + concept_engine
    concepts = []
    for c in selected:
        try:
            score = int(c.get("importance") or 3)
        except Exception:
            score = 3
        score = max(1, min(5, score))

        concepts.append(
            {
                "name": (c.get("name") or ""),
                "importance": _importance_bucket(score),
                "difficulty": (c.get("difficulty") or "medium"),
                "simple": c.get("simple") or "",
                "detailed": c.get("detailed") or "",
                "technical": c.get("technical") or "",
                "example": c.get("example") or "",
                "common_mistake": c.get("common_mistake") or "",
                "unit_type": c.get("unit_type") or "",
                "importance_score": score,
                "evidence": c.get("evidence") or [],
            }
        )

    edges = [
        {
            "from": e.src,
            "to": e.dst,
            "type": e.typ,
            "label": e.label,
            "strength": e.strength,
            "confidence": e.confidence,
            "evidence": e.evidence,
        }
        for e in edges_final
    ]

    return {
        "concepts": concepts,
        "edges": edges,
        "meta": {
            "extraction_mode": mode,
            "doc_type": route.get("doc_type"),
            "router_confidence": route.get("confidence"),
            "candidates_proposed": len(cands),
            "concepts_kept": len(concepts),
            "edges_kept": len(edges),
        },
    }