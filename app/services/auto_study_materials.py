# app/services/auto_study_materials.py
"""
Automatic Study Material Generator
Creates flashcards, quizzes, and study guides adapted to subject type
"""

import json
from typing import Dict, List
from .llm import llm
from .json_utils import safe_json_loads


# ============== Flashcard Generation ==============

async def generate_flashcards(concepts: List[Dict], subject_area: str) -> List[Dict]:
    """
    Generate subject-appropriate flashcards
    
    Args:
        concepts: List of concepts in unified format
        subject_area: "stem", "humanities", etc.
        
    Returns:
        List of flashcard objects
    """
    
    if not concepts:
        return []
    
    if subject_area == "stem":
        return await _generate_stem_flashcards(concepts)
    elif subject_area == "humanities":
        return await _generate_humanities_flashcards(concepts)
    else:
        return await _generate_general_flashcards(concepts)


async def _generate_stem_flashcards(concepts: List[Dict]) -> List[Dict]:
    """STEM flashcards: Focus on formulas, definitions, problem-solving"""
    
    concept_summary = "\n\n".join([
        f"**{c['name']}**\n{c.get('definition', '')}\nFormula: {c.get('subject_specific_data', {}).get('formula', 'N/A')}"
        for c in concepts[:10]  # Limit to 10 concepts
    ])
    
    prompt = f"""
Create flashcards for STEM concepts. For each concept, create 2-3 cards:
1. Definition card (What is X?)
2. Formula/Application card (How do you use X?)
3. Problem card (Given scenario, which concept applies?)

Concepts:
{concept_summary}

Return ONLY valid JSON:
{{
  "flashcards": [
    {{
      "front": "Question or prompt",
      "back": "Answer with explanation",
      "type": "definition" | "formula" | "problem" | "application",
      "difficulty": "easy" | "medium" | "hard",
      "concept_name": "Related concept"
    }}
  ]
}}

Make cards concise but complete. Include examples where helpful.
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective study flashcards."},
            {"role": "user", "content": prompt}
        ], max_tokens=1500, temperature=0.3)
        
        result = safe_json_loads(response, default={"flashcards": []})
        if isinstance(result, dict):
            cards = result.get("flashcards", [])
            return cards if isinstance(cards, list) else []

        return []
    except Exception:
        # One retry: models sometimes add extra text
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective study flashcards. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=1500, temperature=0.3)
            result2 = safe_json_loads(response2, default={"flashcards": []})
            cards2 = result2.get("flashcards", []) if isinstance(result2, dict) else []
            return cards2 if isinstance(cards2, list) else []
        except Exception:
            return []


async def _generate_humanities_flashcards(concepts: List[Dict]) -> List[Dict]:
    """Humanities flashcards: Focus on themes, context, significance"""
    
    concept_summary = "\n\n".join([
        f"**{c['name']}**\n{c.get('definition', '')}\nSignificance: {c.get('subject_specific_data', {}).get('significance', '')}"
        for c in concepts[:10]
    ])
    
    prompt = f"""
Create flashcards for humanities concepts. For each concept, create 2-3 cards:
1. Meaning card (What is this theme/event?)
2. Context card (When/where/why did this occur?)
3. Significance card (Why does this matter?)
4. Analysis card (How does this connect to other ideas?)

Concepts:
{concept_summary}

Return ONLY valid JSON:
{{
  "flashcards": [
    {{
      "front": "Question or prompt",
      "back": "Answer with context and examples",
      "type": "definition" | "context" | "significance" | "analysis",
      "difficulty": "easy" | "medium" | "hard",
      "concept_name": "Related concept"
    }}
  ]
}}

Include specific examples and encourage deeper thinking.
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective study flashcards."},
            {"role": "user", "content": prompt}
        ], max_tokens=1500, temperature=0.3)
        
        result = safe_json_loads(response, default={"flashcards": []})
        cards = result.get("flashcards", []) if isinstance(result, dict) else []
        return cards if isinstance(cards, list) else []
    except Exception:
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective study flashcards. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=1500, temperature=0.3)
            result2 = safe_json_loads(response2, default={"flashcards": []})
            cards2 = result2.get("flashcards", []) if isinstance(result2, dict) else []
            return cards2 if isinstance(cards2, list) else []
        except Exception:
            return []


async def _generate_general_flashcards(concepts: List[Dict]) -> List[Dict]:
    """General flashcards for any subject"""
    
    concept_summary = "\n\n".join([
        f"**{c['name']}**\n{c.get('definition', '')}\nExample: {c.get('example', '')}"
        for c in concepts[:10]
    ])
    
    prompt = f"""
Create flashcards for these concepts:

{concept_summary}

Return ONLY valid JSON:
{{
  "flashcards": [
    {{
      "front": "Question",
      "back": "Answer",
      "type": "definition" | "example" | "application",
      "difficulty": "easy" | "medium" | "hard",
      "concept_name": "Related concept"
    }}
  ]
}}
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective study flashcards."},
            {"role": "user", "content": prompt}
        ], max_tokens=1000, temperature=0.3)
        
        result = safe_json_loads(response, default={"flashcards": []})
        cards = result.get("flashcards", []) if isinstance(result, dict) else []
        return cards if isinstance(cards, list) else []
    except Exception:
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective study flashcards. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=1000, temperature=0.3)
            result2 = safe_json_loads(response2, default={"flashcards": []})
            cards2 = result2.get("flashcards", []) if isinstance(result2, dict) else []
            return cards2 if isinstance(cards2, list) else []
        except Exception:
            return []


# ============== Quiz Generation ==============

async def generate_quiz(concepts: List[Dict], subject_area: str, difficulty: str = "medium") -> List[Dict]:
    """
    Generate subject-appropriate quiz questions
    
    Args:
        concepts: List of concepts
        subject_area: Subject type
        difficulty: "easy", "medium", "hard"
        
    Returns:
        List of quiz questions
    """
    
    if not concepts:
        return []
    
    if subject_area == "stem":
        return await _generate_stem_quiz(concepts, difficulty)
    elif subject_area == "humanities":
        return await _generate_humanities_quiz(concepts, difficulty)
    else:
        return await _generate_general_quiz(concepts, difficulty)


async def _generate_stem_quiz(concepts: List[Dict], difficulty: str) -> List[Dict]:
    """STEM quizzes: Problem-solving, calculations, conceptual"""
    
    concept_summary = "\n".join([f"- {c['name']}: {c.get('definition', '')[:200]}" for c in concepts[:8]])
    
    prompt = f"""
Create a STEM quiz with these question types:
1. Problem-solving (given values, calculate result)
2. Conceptual (why does this work?)
3. Application (when would you use this?)
4. Multiple choice with calculations

Difficulty: {difficulty}
Concepts:
{concept_summary}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "The question text",
      "type": "multiple_choice" | "short_answer" | "problem",
      "choices": ["A", "B", "C", "D"] or null,
      "correct_answer": "A" or "The answer",
      "explanation": "Why this is correct and others are wrong",
      "concept_tested": "Concept name",
      "difficulty": "{difficulty}",
      "hints": ["Hint 1", "Hint 2"]
    }}
  ]
}}

Create 5-8 questions. Mix question types.
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective quiz questions."},
            {"role": "user", "content": prompt}
        ], max_tokens=2000, temperature=0.3)
        
        result = safe_json_loads(response, default={"questions": []})
        qs = result.get("questions", []) if isinstance(result, dict) else []
        return qs if isinstance(qs, list) else []
    except Exception:
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective quiz questions. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=2000, temperature=0.3)
            result2 = safe_json_loads(response2, default={"questions": []})
            qs2 = result2.get("questions", []) if isinstance(result2, dict) else []
            return qs2 if isinstance(qs2, list) else []
        except Exception:
            return []


async def _generate_humanities_quiz(concepts: List[Dict], difficulty: str) -> List[Dict]:
    """Humanities quizzes: Analysis, interpretation, argumentation"""
    
    concept_summary = "\n".join([f"- {c['name']}: {c.get('definition', '')[:200]}" for c in concepts[:8]])
    
    prompt = f"""
Create a humanities quiz with these question types:
1. Analysis (analyze the significance of...)
2. Comparison (compare and contrast...)
3. Interpretation (what does this mean?)
4. Argument (argue for or against...)
5. Context (explain the historical context)

Difficulty: {difficulty}
Concepts:
{concept_summary}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "The question text",
      "type": "essay" | "short_answer" | "multiple_choice",
      "choices": ["A", "B", "C", "D"] or null,
      "sample_answer": "Example of strong answer",
      "key_points": ["Point 1", "Point 2"],
      "concept_tested": "Concept name",
      "difficulty": "{difficulty}"
    }}
  ]
}}

Create 5-8 questions. Encourage critical thinking.
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective quiz questions."},
            {"role": "user", "content": prompt}
        ], max_tokens=2000, temperature=0.3)
        
        result = safe_json_loads(response, default={"questions": []})
        qs = result.get("questions", []) if isinstance(result, dict) else []
        return qs if isinstance(qs, list) else []
    except Exception:
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective quiz questions. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=2000, temperature=0.3)
            result2 = safe_json_loads(response2, default={"questions": []})
            qs2 = result2.get("questions", []) if isinstance(result2, dict) else []
            return qs2 if isinstance(qs2, list) else []
        except Exception:
            return []


async def _generate_general_quiz(concepts: List[Dict], difficulty: str) -> List[Dict]:
    """General quiz questions"""
    
    concept_summary = "\n".join([f"- {c['name']}: {c.get('definition', '')[:150]}" for c in concepts[:8]])
    
    prompt = f"""
Create quiz questions for these concepts:

{concept_summary}

Difficulty: {difficulty}

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "Question text",
      "type": "multiple_choice",
      "choices": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "explanation": "Why correct",
      "concept_tested": "Concept name",
      "difficulty": "{difficulty}"
    }}
  ]
}}

Create 5-7 questions.
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective quiz questions."},
            {"role": "user", "content": prompt}
        ], max_tokens=1500, temperature=0.3)
        
        result = safe_json_loads(response, default={"questions": []})
        qs = result.get("questions", []) if isinstance(result, dict) else []
        return qs if isinstance(qs, list) else []
    except Exception:
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective quiz questions. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=1500, temperature=0.3)
            result2 = safe_json_loads(response2, default={"questions": []})
            qs2 = result2.get("questions", []) if isinstance(result2, dict) else []
            return qs2 if isinstance(qs2, list) else []
        except Exception:
            return []


# ============== Study Guide Generation ==============

async def generate_study_guide(concepts: List[Dict], subject_area: str) -> Dict:
    """
    Generate comprehensive study guide
    
    Returns:
        Study guide with organized sections
    """
    
    concept_text = "\n\n".join([
        f"**{c['name']}**\n{c.get('definition', '')}\nExample: {c.get('example', '')}"
        for c in concepts
    ])
    
    prompt = f"""
Create a comprehensive study guide for these concepts:

{concept_text[:3000]}

Subject area: {subject_area}

Organize as:
1. Key Concepts Overview
2. Must-Know Items (most important)
3. Common Confusions (what students mix up)
4. Study Tips (how to master this)
5. Practice Prompts (what to test yourself on)

Return ONLY valid JSON:
{{
  "title": "Study Guide Title",
  "overview": "Brief overview paragraph",
  "key_concepts": [
    {{
      "concept": "Name",
      "why_important": "Importance",
      "quick_summary": "1-2 sentences"
    }}
  ],
  "must_know": ["Critical item 1", "Critical item 2"],
  "common_confusions": ["Confusion 1 and how to avoid it"],
  "study_tips": ["Tip 1", "Tip 2"],
  "practice_prompts": ["Test yourself: ..."]
}}
"""
    
    try:
        response = await llm([
            {"role": "system", "content": "You create effective study guides."},
            {"role": "user", "content": prompt}
        ], max_tokens=1500, temperature=0.3)

        data = safe_json_loads(response, default={"title": "Study Guide", "overview": "", "key_concepts": []})
        return data if isinstance(data, dict) else {"title": "Study Guide", "overview": "", "key_concepts": []}
    except Exception:
        try:
            response2 = await llm([
                {"role": "system", "content": "You create effective study guides. Return raw JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Output raw JSON only."},
            ], max_tokens=1500, temperature=0.3)
            data2 = safe_json_loads(response2, default={"title": "Study Guide", "overview": "", "key_concepts": []})
            return data2 if isinstance(data2, dict) else {"title": "Study Guide", "overview": "", "key_concepts": []}
        except Exception:
            return {"title": "Study Guide", "overview": "", "key_concepts": []}


# ============== Master Generator ==============

async def generate_all_materials(concepts: List[Dict], subject_area: str) -> Dict:
    """
    Generate ALL study materials at once
    
    Returns:
        {
            "flashcards": [...],
            "quiz": [...],
            "study_guide": {...},
            "generated_at": timestamp
        }
    """
    
    import asyncio
    from datetime import datetime
    
    # Generate all materials concurrently
    flashcards_task = generate_flashcards(concepts, subject_area)
    quiz_task = generate_quiz(concepts, subject_area, "medium")
    guide_task = generate_study_guide(concepts, subject_area)
    
    flashcards, quiz, guide = await asyncio.gather(
        flashcards_task,
        quiz_task,
        guide_task,
        return_exceptions=True
    )
    
    # Handle any errors
    if isinstance(flashcards, Exception):
        flashcards = []
    if isinstance(quiz, Exception):
        quiz = []
    if isinstance(guide, Exception):
        guide = {}
    
    return {
        "flashcards": flashcards,
        "quiz": quiz,
        "study_guide": guide,
        "generated_at": datetime.utcnow().isoformat(),
        "subject_area": subject_area,
        "concept_count": len(concepts)
    }
