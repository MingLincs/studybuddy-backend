# app/services/extractor_router.py
"""LLM-based routing to the best extractor (learning model).

Why this exists:
- "subject_area" alone (stem/humanities/etc.) is too coarse.
- We want *stable* behavior per class, but flexible enough for mixed courses.
- The LLM is used only as a *classifier* with strict options.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from .llm import llm


LEARNING_MODELS = [
    "quantitative",         # math / physics / engineering / cs theory
    "conceptual_science",   # bio / psych / econ theory / social science models
    "humanities_writing",   # english / literature / philosophy / rhetoric
    "historical_timeline",  # history / government / law-history
    "applied_case",         # business / nursing cases / applied decision-making
]


ROUTER_PROMPT = """
You are routing an academic document to the best extraction strategy.

Choose ONE learning_model from this exact list:
- quantitative
- conceptual_science
- humanities_writing
- historical_timeline
- applied_case

Return ONLY valid JSON:
{
  "learning_model": "one_of_the_options",
  "confidence": 0.0,
  "reason": "short reason",
  "mapped_subject_area": "stem|humanities|social_science|business|other"
}

Rules:
- Base decision on how students succeed in THIS course: procedures, models, arguments, timelines, or cases.
- Ignore administrative text like attendance policy unless it defines graded work.
- If it's a syllabus, still route based on how the course content is taught.
- confidence should be lower for mixed/ambiguous docs.
"""


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


async def choose_learning_model(
    *,
    text_content: str,
    classification: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Ask the LLM to choose the extractor (learning model).

    Returns dict with:
    - learning_model
    - confidence
    - reason
    - mapped_subject_area
    """

    excerpt = (text_content or "")[:3500]
    cls = classification or {}

    prompt = {
        "classification": cls,
        "excerpt": excerpt,
    }

    resp = await llm(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=350,
        temperature=0.1,
    )

    data = _safe_json_loads(resp)
    lm = (data.get("learning_model") or "").strip()
    if lm not in LEARNING_MODELS:
        # fallback: map from classifier subject_area if possible
        subj = (cls.get("subject_area") or "").lower()
        if subj == "stem":
            lm = "quantitative"
        elif subj == "humanities":
            lm = "humanities_writing"
        elif subj == "social_science":
            lm = "conceptual_science"
        elif subj == "business":
            lm = "applied_case"
        else:
            lm = "conceptual_science"

    conf = data.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.55

    mapped_subject_area = (data.get("mapped_subject_area") or "other").strip().lower()
    if mapped_subject_area not in {"stem", "humanities", "social_science", "business", "other"}:
        mapped_subject_area = "other"

    return {
        "learning_model": lm,
        "confidence": max(0.0, min(1.0, conf)),
        "reason": (data.get("reason") or "").strip()[:400],
        "mapped_subject_area": mapped_subject_area,
    }

