# app/services/universal_extractors.py
"""Universal subject-adaptive extractors.

These extractors do NOT try to capture *everything*.
They focus on the smallest set of high-value learning units that:
- show up on assessments,
- are foundational for later material,
- cause common student mistakes.

All extractors return a shared JSON shape:

{
  "units": [
    {
      "name": "...",
      "unit_type": "formula|theorem|procedure|theme|argument|event|framework|model|term|skill",
      "importance": "core|important|advanced",
      "difficulty": "easy|medium|hard",
      "simple": "...",
      "detailed": "...",
      "technical": "...",
      "example": "...",
      "common_mistake": "...",
      "signals": {"why_matters": "...", "likely_assessed": true/false}
    }
  ],
  "rejects": ["low-value topics ignored"],
  "coverage_notes": "short note"
}
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .llm import llm


BASE_RULES = """
Hard rules:
- Return ONLY valid JSON. No markdown.
- Extract 8–14 units maximum.
- Each unit must be genuinely test- or grade-relevant. Do NOT include course policies, instructor bio, or fluff.
- Prefer synthesis: merge duplicates, and phrase units as what a student must *know/do*.
- If a unit is too broad, split it.
- If a detail is too minor, skip it.

Quality requirements for each unit:
- simple: 1–2 sentences.
- detailed: 4–7 sentences (step-by-step if procedural).
- technical: optional, but include when the course uses formalism.
- example: concrete and specific (numbers for math; short quote/reference for humanities).
- common_mistake: something realistic.
"""


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"units": [], "rejects": [], "coverage_notes": "parse_error"}


def _normalize_units(data: Dict[str, Any]) -> Dict[str, Any]:
    units = data.get("units")
    if not isinstance(units, list):
        units = []
    out_units = []
    for u in units:
        if not isinstance(u, dict):
            continue
        name = (u.get("name") or "").strip()
        if not name:
            continue
        out_units.append(
            {
                "name": name[:200],
                "unit_type": (u.get("unit_type") or "term")[:40],
                "importance": (u.get("importance") or "important")[:20],
                "difficulty": (u.get("difficulty") or "medium")[:10],
                "simple": (u.get("simple") or "")[:600],
                "detailed": (u.get("detailed") or "")[:2500],
                "technical": (u.get("technical") or "")[:1200],
                "example": (u.get("example") or "")[:1200],
                "common_mistake": (u.get("common_mistake") or "")[:600],
                "signals": u.get("signals") if isinstance(u.get("signals"), dict) else {},
            }
        )
    rejects = data.get("rejects")
    if not isinstance(rejects, list):
        rejects = []
    notes = data.get("coverage_notes")
    if not isinstance(notes, str):
        notes = ""
    return {"units": out_units, "rejects": rejects[:20], "coverage_notes": notes[:600]}


async def extract_quantitative(text: str) -> Dict[str, Any]:
    prompt = f"""
You are extracting the highest-value learning units from QUANTITATIVE course material (math, physics, engineering, CS theory, stats).

Focus on:
- procedures students must execute (step sequences)
- formulas/theorems and when to use them
- canonical problem patterns (recognition cues)
- constraints/assumptions that change the method

Unit types to use: formula, theorem, procedure, algorithm, problem_type, definition.

{BASE_RULES}

Return ONLY JSON in this exact shape:
{{
  "units": [ ... ],
  "rejects": ["..."],
  "coverage_notes": "..."
}}

Document excerpt:
{(text or '')[:7000]}
"""
    resp = await llm(
        [{"role": "system", "content": "You extract study-critical units."}, {"role": "user", "content": prompt}],
        max_tokens=2200,
        temperature=0.2,
    )
    return _normalize_units(_safe_json_loads(resp))


async def extract_conceptual_science(text: str) -> Dict[str, Any]:
    prompt = f"""
You are extracting the highest-value learning units from CONCEPTUAL SCIENCE / SOCIAL SCIENCE material (bio, psych, econ theory, sociology).

Focus on:
- models/theories and their components
- cause-effect mechanisms
- key terms with operational meaning
- classic study/research patterns if present (method → finding → implication)

Unit types to use: model, theory, mechanism, term, study, process.

{BASE_RULES}

Return ONLY JSON in this exact shape:
{{
  "units": [ ... ],
  "rejects": ["..."],
  "coverage_notes": "..."
}}

Document excerpt:
{(text or '')[:7000]}
"""
    resp = await llm(
        [{"role": "system", "content": "You extract study-critical units."}, {"role": "user", "content": prompt}],
        max_tokens=2200,
        temperature=0.2,
    )
    return _normalize_units(_safe_json_loads(resp))


async def extract_humanities_writing(text: str) -> Dict[str, Any]:
    prompt = f"""
You are extracting the highest-value learning units from HUMANITIES / WRITING material (English, literature, philosophy, rhetoric).

Focus on:
- argument structures students must produce (thesis → claims → evidence → analysis)
- themes and how to support them with evidence
- writing skills/rubrics that affect grades
- literary/rhetorical devices only if they are actively used/assessed

Unit types to use: skill, argument, theme, device, term, framework.

{BASE_RULES}

Return ONLY JSON in this exact shape:
{{
  "units": [ ... ],
  "rejects": ["..."],
  "coverage_notes": "..."
}}

Document excerpt:
{(text or '')[:7000]}
"""
    resp = await llm(
        [{"role": "system", "content": "You extract study-critical units."}, {"role": "user", "content": prompt}],
        max_tokens=2200,
        temperature=0.2,
    )
    return _normalize_units(_safe_json_loads(resp))


async def extract_historical_timeline(text: str) -> Dict[str, Any]:
    prompt = f"""
You are extracting the highest-value learning units from HISTORICAL / TIMELINE-based material (history, gov, law-history).

Focus on:
- events/processes and their causes/effects
- timelines/turning points only when they are tied to graded understanding
- key terms/actors that students must connect in essays or exams
- competing interpretations if explicitly discussed

Unit types to use: event, turning_point, timeline, argument, term.

{BASE_RULES}

Return ONLY JSON in this exact shape:
{{
  "units": [ ... ],
  "rejects": ["..."],
  "coverage_notes": "..."
}}

Document excerpt:
{(text or '')[:7000]}
"""
    resp = await llm(
        [{"role": "system", "content": "You extract study-critical units."}, {"role": "user", "content": prompt}],
        max_tokens=2200,
        temperature=0.2,
    )
    return _normalize_units(_safe_json_loads(resp))


async def extract_applied_case(text: str) -> Dict[str, Any]:
    prompt = f"""
You are extracting the highest-value learning units from APPLIED / CASE-BASED material (business, nursing case studies, applied decision-making).

Focus on:
- frameworks used to evaluate situations
- decision criteria and tradeoffs
- "if X then Y" heuristics
- common failure modes (what students forget to consider)

Unit types to use: framework, procedure, decision_rule, rubric, term.

{BASE_RULES}

Return ONLY JSON in this exact shape:
{{
  "units": [ ... ],
  "rejects": ["..."],
  "coverage_notes": "..."
}}

Document excerpt:
{(text or '')[:7000]}
"""
    resp = await llm(
        [{"role": "system", "content": "You extract study-critical units."}, {"role": "user", "content": prompt}],
        max_tokens=2200,
        temperature=0.2,
    )
    return _normalize_units(_safe_json_loads(resp))


async def extract_by_learning_model(learning_model: str, text: str) -> Dict[str, Any]:
    learning_model = (learning_model or "").strip().lower()
    if learning_model == "quantitative":
        return await extract_quantitative(text)
    if learning_model == "conceptual_science":
        return await extract_conceptual_science(text)
    if learning_model == "humanities_writing":
        return await extract_humanities_writing(text)
    if learning_model == "historical_timeline":
        return await extract_historical_timeline(text)
    if learning_model == "applied_case":
        return await extract_applied_case(text)
    return await extract_conceptual_science(text)
