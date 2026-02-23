# app/services/syllabus_processor.py
"""
Intelligent Syllabus Processor
Extracts everything from syllabus and creates study timeline
"""

import json
from typing import Dict, List
from datetime import datetime, timedelta
from .llm import llm


SYLLABUS_EXTRACTION_PROMPT = """
You are analyzing a course syllabus. Extract ALL information comprehensively.

Return ONLY valid JSON:

{
  "course_info": {
    "name": "Course name",
    "code": "Course code (e.g., CS 101)",
    "professor": "Professor name",
    "semester": "Fall 2024, Spring 2025, etc.",
    "credits": 3,
    "meeting_times": "Days and times"
  },
  "schedule": [
    {
      "week": 1,
      "date_range": "Jan 15-19" or "Week of Jan 15",
      "topics": ["Topic 1", "Topic 2"],
      "readings": ["Reading 1 (pages)", "Reading 2"],
      "assignments_due": ["Assignment name"] or []
    }
  ],
  "assessments": [
    {
      "type": "exam" | "quiz" | "assignment" | "project" | "presentation" | "participation",
      "name": "Midterm Exam 1",
      "date": "March 15, 2024" or "Week 8",
      "weight_percent": 20,
      "topics_covered": ["Topics that will be tested"],
      "format": "multiple choice, essay, etc.",
      "details": "Any special instructions"
    }
  ],
  "grading_breakdown": {
    "exams": 40,
    "quizzes": 20,
    "assignments": 30,
    "participation": 10
  },
  "grading_scale": "A: 90-100, B: 80-89, etc.",
  "learning_objectives": [
    "Students will be able to...",
    "Students will understand..."
  ],
  "required_materials": [
    "Textbook name",
    "Software",
    "Other materials"
  ],
  "policies": {
    "attendance": "Attendance policy text",
    "late_work": "Late work policy",
    "academic_integrity": "Integrity policy"
  },
  "office_hours": "When and where",
  "important_dates": [
    {
      "date": "Feb 20",
      "event": "No class - holiday"
    }
  ]
}

Extract everything. If information is not present, use null or [].
Be thorough - this is critical for student success.
"""


async def process_syllabus(syllabus_text: str) -> Dict:
    """
    Extract complete syllabus information
    
    Args:
        syllabus_text: Full text of syllabus
        
    Returns:
        Structured syllabus data
    """
    
    try:
        response = await llm(
            [
                {"role": "system", "content": SYLLABUS_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Syllabus:\n\n{syllabus_text[:6000]}"}
            ],
            max_tokens=2500,
            temperature=0.1  # Very low - need accuracy
        )
        
        syllabus_data = json.loads(response)
        
        # Generate study timeline
        syllabus_data['study_timeline'] = await create_study_timeline(syllabus_data)
        
        return syllabus_data
        
    except Exception as e:
        print(f"Syllabus processing error: {e}")
        return _default_syllabus_structure()


def _default_syllabus_structure() -> Dict:
    """Fallback structure"""
    return {
        "course_info": {},
        "schedule": [],
        "assessments": [],
        "grading_breakdown": {},
        "learning_objectives": [],
        "study_timeline": []
    }


async def create_study_timeline(syllabus_data: Dict) -> List[Dict]:
    """
    Creates personalized week-by-week study plan
    
    Args:
        syllabus_data: Extracted syllabus information
        
    Returns:
        List of weekly study plans
    """
    
    schedule = syllabus_data.get('schedule', [])
    assessments = syllabus_data.get('assessments', [])
    
    if not schedule:
        return []
    
    # Build context for AI
    schedule_summary = json.dumps(schedule[:15], indent=2)  # First 15 weeks
    assessments_summary = json.dumps(assessments, indent=2)
    
    prompt = f"""
Create a strategic study plan based on this course schedule.

Schedule:
{schedule_summary}

Assessments:
{assessments_summary}

For each week, provide actionable study recommendations:

Return ONLY valid JSON:
{{
  "weekly_plans": [
    {{
      "week": 1,
      "week_title": "Getting Started",
      "topics_this_week": ["Topic 1", "Topic 2"],
      "what_to_study": [
        "Read chapter 1 (focus on sections 1.1-1.3)",
        "Complete practice problems 1-10",
        "Review lecture slides"
      ],
      "why_important": "This builds foundation for midterm",
      "estimated_study_hours": 5,
      "priority": "high" | "medium" | "low",
      "upcoming_deadlines": ["Assignment 1 due next week"],
      "preparation_for": "Sets up concepts for Week 3 exam",
      "key_concepts_to_master": ["Concept A", "Concept B"],
      "study_methods": ["flashcards", "practice_problems", "group_study"],
      "milestone": "By end of week, you should be able to..."
    }}
  ]
}}

Guidelines:
- Look ahead to exams and prepare early
- Identify critical weeks (before exams)
- Suggest more study time when needed
- Connect current topics to future assessments
- Be specific and actionable
"""
    
    try:
        response = await llm(
            [
                {"role": "system", "content": "You create effective study timelines."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.2
        )
        
        result = json.loads(response)
        return result.get('weekly_plans', [])
        
    except Exception as e:
        print(f"Timeline creation error: {e}")
        return _generate_basic_timeline(schedule)


def _generate_basic_timeline(schedule: List[Dict]) -> List[Dict]:
    """Fallback: Create basic timeline from schedule"""
    
    timeline = []
    for week_data in schedule:
        week = week_data.get('week', 0)
        topics = week_data.get('topics', [])
        
        timeline.append({
            "week": week,
            "topics_this_week": topics,
            "what_to_study": [
                f"Study {topic}" for topic in topics[:3]
            ],
            "estimated_study_hours": 4,
            "priority": "medium"
        })
    
    return timeline


async def get_this_weeks_tasks(syllabus_data: Dict, current_week: int) -> Dict:
    """
    Get specific study tasks for this week
    
    Args:
        syllabus_data: Full syllabus data with timeline
        current_week: Current week number
        
    Returns:
        This week's detailed tasks
    """
    
    timeline = syllabus_data.get('study_timeline', [])
    
    # Find this week
    this_week = None
    for week_plan in timeline:
        if week_plan.get('week') == current_week:
            this_week = week_plan
            break
    
    if not this_week:
        return {"week": current_week, "tasks": []}
    
    # Check for upcoming assessments
    upcoming_assessments = _get_upcoming_assessments(
        syllabus_data.get('assessments', []),
        current_week
    )
    
    return {
        "week": current_week,
        "title": this_week.get('week_title', f'Week {current_week}'),
        "topics": this_week.get('topics_this_week', []),
        "tasks": this_week.get('what_to_study', []),
        "estimated_hours": this_week.get('estimated_study_hours', 5),
        "priority": this_week.get('priority', 'medium'),
        "why_important": this_week.get('why_important', ''),
        "milestone": this_week.get('milestone', ''),
        "upcoming_assessments": upcoming_assessments,
        "study_methods": this_week.get('study_methods', [])
    }


def _get_upcoming_assessments(assessments: List[Dict], current_week: int) -> List[Dict]:
    """Find assessments coming up in next 2 weeks"""
    
    upcoming = []
    for assessment in assessments:
        # Try to parse week from date field
        date_str = assessment.get('date', '')
        
        # Simple check if "Week X" is in date
        if 'week' in date_str.lower():
            try:
                week_num = int(''.join(filter(str.isdigit, date_str)))
                if current_week <= week_num <= current_week + 2:
                    upcoming.append({
                        "name": assessment.get('name', 'Assessment'),
                        "type": assessment.get('type', 'exam'),
                        "week": week_num,
                        "weight": assessment.get('weight_percent', 0),
                        "topics": assessment.get('topics_covered', [])
                    })
            except:
                pass
    
    return upcoming


async def generate_exam_prep_plan(
    syllabus_data: Dict,
    exam_name: str,
    weeks_until_exam: int
) -> Dict:
    """
    Generate focused exam preparation plan
    
    Args:
        syllabus_data: Full syllabus data
        exam_name: Name of exam
        weeks_until_exam: How many weeks away
        
    Returns:
        Week-by-week exam prep plan
    """
    
    # Find the exam
    assessments = syllabus_data.get('assessments', [])
    exam = None
    for assessment in assessments:
        if exam_name.lower() in assessment.get('name', '').lower():
            exam = assessment
            break
    
    if not exam:
        return {"error": "Exam not found"}
    
    topics = exam.get('topics_covered', [])
    exam_format = exam.get('format', '')
    
    prompt = f"""
Create a {weeks_until_exam}-week exam preparation plan.

Exam: {exam_name}
Format: {exam_format}
Topics covered: {', '.join(topics)}

Create week-by-week plan focusing on:
- Week 1-2: Review and identify weak areas
- Week 3: Practice and drill
- Week 4: Full review and mock exams

Return ONLY valid JSON:
{{
  "prep_plan": [
    {{
      "week": 1,
      "focus": "Review fundamentals",
      "specific_tasks": ["Task 1", "Task 2"],
      "study_hours_needed": 8,
      "practice_types": ["flashcards", "problems"],
      "checkpoint": "What you should know by end of week"
    }}
  ],
  "study_strategies": ["Strategy 1", "Strategy 2"],
  "common_pitfalls": ["What students usually struggle with"],
  "day_before_tips": ["What to do day before exam"]
}}
"""
    
    try:
        response = await llm(
            [
                {"role": "system", "content": "You create effective exam prep plans."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        
        return json.loads(response)
    except:
        return {"prep_plan": []}
