# app/services/intelligent_classifier.py
"""
Revolutionary AI Document Classifier
Determines subject type and processing strategy
"""

import json
from typing import Dict, Optional
from .llm import llm


CLASSIFIER_PROMPT = """
You are an expert academic classifier. Analyze this document excerpt and determine:

1. Document Type (what kind of document is this?)
2. Subject Area (what field of study?)
3. Course Level (difficulty/year level?)
4. Teaching Approach (how is content presented?)

Return ONLY valid JSON:

{
  "document_type": "syllabus" | "lecture_notes" | "textbook_chapter" | "reading_material" | "assignment" | "exam_study_guide",
  "subject_area": "stem" | "humanities" | "social_science" | "arts" | "business" | "other",
  "specific_subject": "Computer Science" | "History" | "Psychology" | "Economics" | etc.,
  "course_level": "introductory" | "intermediate" | "advanced" | "graduate",
  "teaching_focus": "theoretical" | "practical" | "applied" | "mixed",
  "content_characteristics": {
    "has_formulas": true/false,
    "has_code": true/false,
    "has_dates": true/false,
    "has_analysis": true/false,
    "has_arguments": true/false,
    "has_problems": true/false,
    "language_heavy": true/false
  },
  "recommended_study_methods": ["flashcards", "practice_problems", "concept_map", "timeline", "essay_practice"],
  "confidence": 0.0-1.0
}

Guidelines:
- STEM: Math, CS, Physics, Chemistry, Engineering, Biology
- Humanities: History, Literature, Philosophy, Languages, Art History
- Social Science: Psychology, Sociology, Political Science, Economics, Anthropology
- Arts: Music Theory, Studio Art, Theater, Film
- Business: Finance, Marketing, Management, Accounting

Be accurate. If uncertain, mark confidence lower.
"""


async def classify_document(text_content: str) -> Dict:
    """
    Classifies document to determine optimal processing strategy
    
    Args:
        text_content: First ~3000 chars of document
        
    Returns:
        Classification dict with processing recommendations
    """
    
    # Use first 3000 chars for classification
    excerpt = text_content[:3000] if text_content else ""
    
    if not excerpt.strip():
        return _default_classification()
    
    try:
        response = await llm(
            [
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Document excerpt:\n\n{excerpt}"}
            ],
            max_tokens=800,
            temperature=0.1  # Low temperature for consistency
        )
        
        classification = json.loads(response)
        
        # Validate required fields
        if not all(k in classification for k in ['document_type', 'subject_area', 'specific_subject']):
            return _default_classification()
        
        return classification
        
    except Exception as e:
        print(f"Classification error: {e}")
        return _default_classification()


def _default_classification() -> Dict:
    """Fallback classification"""
    return {
        "document_type": "unknown",
        "subject_area": "other",
        "specific_subject": "General",
        "course_level": "intermediate",
        "teaching_focus": "mixed",
        "content_characteristics": {
            "has_formulas": False,
            "has_code": False,
            "has_dates": False,
            "has_analysis": False,
            "has_arguments": False,
            "has_problems": False,
            "language_heavy": False
        },
        "recommended_study_methods": ["flashcards", "concept_map"],
        "confidence": 0.3
    }


async def classify_and_recommend(text_content: str) -> Dict:
    """
    Classify document and provide specific recommendations
    
    Returns classification plus actionable recommendations
    """
    
    classification = await classify_document(text_content)
    
    # Add specific processing recommendations
    subject = classification['subject_area']
    doc_type = classification['document_type']
    
    recommendations = {
        "classification": classification,
        "processing_strategy": _get_processing_strategy(subject, doc_type),
        "study_material_types": _get_study_materials(subject, doc_type),
        "visualization_type": _get_visualization_type(subject),
        "quiz_style": _get_quiz_style(subject)
    }
    
    return recommendations


def _get_processing_strategy(subject: str, doc_type: str) -> str:
    """Determine how to process this document"""
    
    if doc_type == "syllabus":
        return "extract_schedule_and_timeline"
    
    if subject == "stem":
        return "extract_formulas_and_problems"
    elif subject == "humanities":
        return "extract_themes_and_arguments"
    elif subject == "social_science":
        return "extract_theories_and_studies"
    else:
        return "extract_general_concepts"


def _get_study_materials(subject: str, doc_type: str) -> list:
    """What study materials to generate"""
    
    materials = ["flashcards", "concept_map", "summary"]
    
    if subject == "stem":
        materials.extend(["practice_problems", "formula_sheet"])
    elif subject == "humanities":
        materials.extend(["timeline", "essay_prompts", "analysis_questions"])
    elif subject == "social_science":
        materials.extend(["case_studies", "research_summaries"])
    
    if doc_type == "syllabus":
        materials.append("study_calendar")
    
    return materials


def _get_visualization_type(subject: str) -> str:
    """What type of concept map visualization to use"""
    
    if subject == "stem":
        return "hierarchical"  # Shows prerequisites clearly
    elif subject == "humanities":
        return "thematic"  # Shows theme connections
    else:
        return "network"  # General network graph


def _get_quiz_style(subject: str) -> str:
    """What type of quizzes to generate"""
    
    quiz_styles = {
        "stem": "problem_solving",
        "humanities": "analysis_and_essay",
        "social_science": "application_and_case",
        "arts": "interpretation_and_critique",
        "business": "scenario_based"
    }
    
    return quiz_styles.get(subject, "multiple_choice")
