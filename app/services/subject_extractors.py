# app/services/subject_extractors.py
"""
Subject-Aware Content Extractors
Different extraction strategies for different subject types
"""

import json
from typing import Dict, List
from .llm import llm
from .json_utils import safe_json_loads


# ============== STEM Extractor ==============

STEM_EXTRACTION_PROMPT = """
You are extracting concepts from STEM course material (Math, CS, Physics, etc.).

Focus on:
- Theorems, formulas, algorithms
- Technical definitions
- Problem-solving methods
- Mathematical relationships
- Code/pseudocode patterns

Return ONLY valid JSON:

{
  "concepts": [
    {
      "name": "Concept Name",
      "type": "theorem" | "formula" | "algorithm" | "definition" | "technique",
      "definition": "Clear, precise definition",
      "formula": "LaTeX notation if applicable (e.g., E = mc^2)",
      "algorithm_steps": ["Step 1", "Step 2"] or null,
      "example_problem": "Concrete numerical/code example",
      "solution_approach": "How to solve problems with this",
      "common_mistakes": "What students typically get wrong",
      "prerequisites": ["Prior concepts needed"],
      "applications": ["Where this is used in real world"],
      "difficulty": "easy" | "medium" | "hard"
    }
  ],
  "practice_problems": [
    {
      "problem_statement": "The problem",
      "difficulty": "easy" | "medium" | "hard",
      "concepts_used": ["Concept names"],
      "hints": ["Hint 1", "Hint 2"],
      "solution_outline": "High-level approach (NOT full answer)"
    }
  ]
}
"""


async def extract_stem_content(text: str) -> Dict:
    """Extract STEM-specific content (formulas, problems, algorithms)"""
    
    try:
        response = await llm(
            [
                {"role": "system", "content": STEM_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract STEM concepts:\n\n{text[:4000]}"}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        parsed = safe_json_loads(response, default=None)
        if isinstance(parsed, dict) and (parsed.get("concepts") or parsed.get("practice_problems") is not None):
            return parsed

        # One retry with stricter instruction (models sometimes add extra text)
        response2 = await llm(
            [
                {"role": "system", "content": STEM_EXTRACTION_PROMPT + "\n\nIMPORTANT: Output raw JSON only. No markdown, no commentary."},
                {"role": "user", "content": f"Return ONLY JSON.\n\n{text[:4000]}"},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        parsed2 = safe_json_loads(response2, default={"concepts": [], "practice_problems": []})
        return parsed2 if isinstance(parsed2, dict) else {"concepts": [], "practice_problems": []}
    except Exception as e:
        print(f"STEM extraction error: {e}")
        return {"concepts": [], "practice_problems": []}


# ============== Humanities Extractor ==============

HUMANITIES_EXTRACTION_PROMPT = """
You are extracting concepts from humanities material (History, Literature, Philosophy, etc.).

Focus on:
- Themes and motifs
- Historical events and periods
- Arguments and perspectives
- Literary devices and techniques
- Cultural context

Return ONLY valid JSON:

{
  "concepts": [
    {
      "name": "Concept/Theme Name",
      "type": "theme" | "historical_event" | "literary_device" | "philosophical_idea" | "movement",
      "definition": "Clear explanation of meaning",
      "historical_context": "When/where this occurred or originated",
      "significance": "Why this matters; its impact",
      "key_figures": ["Important people associated with this"],
      "related_works": ["Books, documents, artifacts related to this"],
      "different_perspectives": ["How different scholars/groups view this"],
      "examples": ["Specific instances or passages"],
      "modern_relevance": "How this connects to today",
      "essay_angles": ["Potential essay topics about this"]
    }
  ],
  "key_arguments": [
    {
      "main_claim": "The central argument",
      "evidence": ["Supporting points"],
      "counterarguments": ["Opposing views"],
      "author_or_source": "Who made this argument"
    }
  ],
  "timeline_events": [
    {
      "date_or_period": "When this happened",
      "event": "What happened",
      "cause": "Why it happened",
      "effect": "What resulted",
      "significance": "Why it matters"
    }
  ]
}
"""


async def extract_humanities_content(text: str) -> Dict:
    """Extract humanities-specific content (themes, arguments, context)"""
    
    try:
        response = await llm(
            [
                {"role": "system", "content": HUMANITIES_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract humanities concepts:\n\n{text[:4000]}"}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        parsed = safe_json_loads(response, default=None)
        if isinstance(parsed, dict) and (parsed.get("concepts") or parsed.get("timeline_events") is not None):
            return parsed

        response2 = await llm(
            [
                {"role": "system", "content": HUMANITIES_EXTRACTION_PROMPT + "\n\nIMPORTANT: Output raw JSON only. No markdown, no commentary."},
                {"role": "user", "content": f"Return ONLY JSON.\n\n{text[:4000]}"},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        parsed2 = safe_json_loads(response2, default={"concepts": [], "key_arguments": [], "timeline_events": []})
        return parsed2 if isinstance(parsed2, dict) else {"concepts": [], "key_arguments": [], "timeline_events": []}
    except Exception as e:
        print(f"Humanities extraction error: {e}")
        return {"concepts": [], "key_arguments": [], "timeline_events": []}


# ============== Social Science Extractor ==============

SOCIAL_SCIENCE_EXTRACTION_PROMPT = """
You are extracting concepts from social science material (Psychology, Sociology, Economics, etc.).

Focus on:
- Theories and models
- Research studies
- Statistical concepts
- Social phenomena
- Experimental methods

Return ONLY valid JSON:

{
  "concepts": [
    {
      "name": "Concept Name",
      "type": "theory" | "research_method" | "phenomenon" | "model" | "term",
      "definition": "Clear definition",
      "key_researchers": ["Who developed/studied this"],
      "research_evidence": "Key studies supporting this",
      "real_world_examples": ["Concrete examples in society"],
      "applications": ["How this is used in practice"],
      "debates": "Controversies or different views",
      "related_concepts": ["Connected ideas"],
      "measurement": "How this is measured/studied if applicable"
    }
  ],
  "studies": [
    {
      "researcher": "Who conducted it",
      "year": "When",
      "methodology": "How it was done",
      "findings": "What was discovered",
      "significance": "Why it matters",
      "limitations": "Weaknesses of the study"
    }
  ]
}
"""


async def extract_social_science_content(text: str) -> Dict:
    """Extract social science content (theories, studies, phenomena)"""
    
    try:
        response = await llm(
            [
                {"role": "system", "content": SOCIAL_SCIENCE_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract social science concepts:\n\n{text[:4000]}"}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        parsed = safe_json_loads(response, default=None)
        if isinstance(parsed, dict) and (parsed.get("concepts") or parsed.get("studies") is not None):
            return parsed

        response2 = await llm(
            [
                {"role": "system", "content": SOCIAL_SCIENCE_EXTRACTION_PROMPT + "\n\nIMPORTANT: Output raw JSON only. No markdown, no commentary."},
                {"role": "user", "content": f"Return ONLY JSON.\n\n{text[:4000]}"},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        parsed2 = safe_json_loads(response2, default={"concepts": [], "studies": []})
        return parsed2 if isinstance(parsed2, dict) else {"concepts": [], "studies": []}
    except Exception as e:
        print(f"Social science extraction error: {e}")
        return {"concepts": [], "studies": []}


# ============== Master Extractor (Routes to appropriate extractor) ==============

async def extract_content_intelligent(text: str, subject_area: str, classification: Dict) -> Dict:
    """
    Routes to appropriate extractor based on subject
    
    Args:
        text: Document text
        subject_area: "stem", "humanities", "social_science", etc.
        classification: Full classification result
        
    Returns:
        Extracted content formatted for the subject type
    """
    
    if subject_area == "stem":
        result = await extract_stem_content(text)
    elif subject_area == "humanities":
        result = await extract_humanities_content(text)
    elif subject_area == "social_science":
        result = await extract_social_science_content(text)
    else:
        # Default to general extraction
        result = await extract_stem_content(text)  # Use STEM as fallback
    
    # Add metadata
    result['subject_area'] = subject_area
    result['classification'] = classification
    result['extraction_timestamp'] = _now()
    
    return result


def _now():
    from datetime import datetime
    return datetime.utcnow().isoformat()


# ============== Format Conversion ==============

def convert_to_unified_format(extracted_content: Dict) -> List[Dict]:
    """
    Converts subject-specific format to unified concept format
    for storage in database
    
    Returns list of concepts in standard format:
    {
      "name": str,
      "definition": str,
      "example": str,
      "application": str,
      "subject_specific_data": dict (varies by subject)
    }
    """
    
    concepts = extracted_content.get('concepts', [])
    subject_area = extracted_content.get('subject_area', 'other')
    
    unified = []
    
    for concept in concepts:
        # Common fields
        unified_concept = {
            "name": concept.get('name', 'Unknown'),
            "definition": concept.get('definition', ''),
            "subject_area": subject_area
        }
        
        # Subject-specific formatting
        if subject_area == "stem":
            unified_concept['example'] = concept.get('example_problem', '')
            unified_concept['application'] = '\n'.join(concept.get('applications', []))
            unified_concept['subject_specific_data'] = {
                "formula": concept.get('formula'),
                "algorithm_steps": concept.get('algorithm_steps'),
                "solution_approach": concept.get('solution_approach'),
                "common_mistakes": concept.get('common_mistakes'),
                "prerequisites": concept.get('prerequisites', [])
            }
            
        elif subject_area == "humanities":
            unified_concept['example'] = '\n'.join(concept.get('examples', []))
            unified_concept['application'] = concept.get('modern_relevance', '')
            unified_concept['subject_specific_data'] = {
                "historical_context": concept.get('historical_context'),
                "significance": concept.get('significance'),
                "key_figures": concept.get('key_figures', []),
                "different_perspectives": concept.get('different_perspectives', []),
                "essay_angles": concept.get('essay_angles', [])
            }
            
        elif subject_area == "social_science":
            unified_concept['example'] = '\n'.join(concept.get('real_world_examples', []))
            unified_concept['application'] = '\n'.join(concept.get('applications', []))
            unified_concept['subject_specific_data'] = {
                "key_researchers": concept.get('key_researchers', []),
                "research_evidence": concept.get('research_evidence'),
                "debates": concept.get('debates'),
                "measurement": concept.get('measurement')
            }
        
        unified.append(unified_concept)
    
    return unified
