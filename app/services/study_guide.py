import json
from typing import Any, Dict, List
from .llm import llm

GUIDE_SCHEMA_HINT = {
  "chapter_title": "string",
  "estimated_study_minutes": 45,
  "concepts": [
    {
      "id": "short_id_string",
      "name": "Concept name",
      "importance": "core|important|advanced",
      "difficulty": "easy|medium|hard",
      "prerequisites": ["concept_id", "concept_id"],
      "simple": "ELI5 explanation",
      "detailed": "Studying for test explanation",
      "technical": "Exam/mastery explanation",
      "example": "Relatable real-world example",
      "common_mistake": "What students usually get wrong"
    }
  ]
}

async def generate_study_guide(chapter_text: str, chapter_title: str, *, max_concepts: int = 10) -> Dict[str, Any]:
    """Generate an interactive study guide JSON object from chapter text."""
    # Keep prompt bounded
    src = chapter_text.strip()
    if len(src) > 24000:
        src = src[:24000]

    system = (
        "You are StudyBuddy, an expert tutor. "
        "Return ONLY valid JSON, no markdown, no prose. "
        "Follow the schema exactly. "
        "Make it practical, student-friendly, and accurate. "
        "Do not invent page numbers. If unsure, omit sources. "
        f"Generate 6 to {max_concepts} concepts max."
    )

    user = (
        f"Create a structured study guide for this chapter titled: {chapter_title!r}.\n"
        "You must:\n"
        "1) Identify the key concepts taught.\n"
        "2) Order them in a learning path (foundations first).\n"
        "3) Label each as importance: core/important/advanced.\n"
        "4) Label difficulty: easy/medium/hard. Mark tricky concepts as hard.\n"
        "5) Add prerequisites using other concept ids.\n"
        "6) For each concept write: simple, detailed, technical explanations + example + common_mistake.\n\n"
        "Rules:\n"
        "- Keep explanations concise (2-6 sentences each).\n"
        "- Make the simple explanation use an analogy when possible.\n"
        "- The technical explanation can mention OS terms, APIs, edge cases.\n"
        "- Make ids short, lowercase, snake_case.\n\n"
        "Return JSON in this shape (keys must match):\n"
        + json.dumps(GUIDE_SCHEMA_HINT, ensure_ascii=False)
        + "\n\n"
        "CHAPTER TEXT:\n"
        + src
    )

    raw = await llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=2600,
        temperature=0.2,
    )

    # Parse, repair if needed
    try:
        obj = json.loads(raw)
        return obj
    except Exception:
        fixed = await llm(
            [
                {"role": "system", "content": "Fix this into valid JSON only. No prose. Keep the same schema."},
                {"role": "user", "content": raw},
            ],
            max_tokens=2600,
            temperature=0,
        )
        return json.loads(fixed)
