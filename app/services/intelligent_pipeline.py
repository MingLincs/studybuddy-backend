# app/services/intelligent_pipeline.py
"""Single source of truth for document upload + AI processing.

This consolidates the old /upload pipeline and the /intelligent pipeline so:
- storage + db fields are always written correctly (pdf_path, cards_json, guide_json)
- LLM routing selects the right extractor
- syllabi get special handling

Routers should call `process_uploaded_pdf(...)` and return its response.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from .cache import sha256_bytes
from .db import new_uuid, upload_pdf_to_storage, upsert_document
from .extractor_router import choose_learning_model
from .intelligent_classifier import classify_and_recommend
from .llm import llm
from .pdf import extract_text_from_pdf
from .syllabus_processor import process_syllabus
from .universal_extractors import extract_by_learning_model
from .concept_engine import update_class_graph


def _to_guide_json(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert extractor units into your existing guide_json format."""
    concepts = []
    for u in units:
        concepts.append(
            {
                "name": u.get("name", "")[:200],
                "importance": u.get("importance", "important"),
                "difficulty": u.get("difficulty", "medium"),
                "simple": u.get("simple", ""),
                "detailed": u.get("detailed", ""),
                "technical": u.get("technical", ""),
                "example": u.get("example", ""),
                "common_mistake": u.get("common_mistake", ""),
            }
        )
    return {"concepts": concepts}


async def _make_markdown_summary(text: str, *, word_target: int) -> str:
    src = (text or "")[:18000]
    if not src.strip():
        return ""
    return await llm(
        [
            {
                "role": "system",
                "content": (
                    f"Write detailed structured study notes in markdown (~{word_target} words). "
                    "Use headings/subheadings, bullets, and clear spacing. "
                    "Prioritize what will be tested and what helps a student perform well. "
                    "Avoid fluff."
                ),
            },
            {"role": "user", "content": src},
        ],
        max_tokens=2500,
        temperature=0.2,
    )


async def _make_flashcards(units: list[dict[str, Any]], *, max_cards: int = 30) -> dict[str, Any]:
    """Return old-format cards_json: {"cards":[{"type","front","back"}]}"""
    if not units:
        return {"cards": []}

    # Build compact context
    blob = "\n\n".join(
        [
            f"{i+1}. {u.get('name','')}: {u.get('simple','') or u.get('detailed','')[:180]}\nExample: {u.get('example','')}\nMistake: {u.get('common_mistake','')}"
            for i, u in enumerate(units[:12])
        ]
    )

    prompt = f"""
Create {min(max_cards, 30)} high-quality flashcards from these learning units.

Return ONLY valid JSON:
{{
  "cards": [
    {{"type":"definition|qa|concept|procedure|application","front":"...","back":"..."}}
  ]
}}

Rules:
- Focus on testable knowledge and common mistakes.
- Keep fronts short. Backs should teach.
- Avoid duplicates.

Learning units:
{blob}
"""

    try:
        resp = await llm(
            [
                {"role": "system", "content": "You create effective study flashcards."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1600,
            temperature=0.2,
        )
        data = json.loads(resp)
        if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
            return {"cards": []}

        # normalize
        cards = []
        for c in data.get("cards", [])[: max_cards + 5]:
            if not isinstance(c, dict):
                continue
            front = (c.get("front") or "").strip()
            back = (c.get("back") or "").strip()
            if not front or not back:
                continue
            cards.append(
                {
                    "type": (c.get("type") or "qa")[:30],
                    "front": front[:500],
                    "back": back[:2000],
                }
            )
        return {"cards": cards[:max_cards]}
    except Exception:
        return {"cards": []}


async def process_uploaded_pdf(
    *,
    user_id: str,
    class_id: str,
    filename: str,
    raw_pdf: bytes,
    title: Optional[str] = None,
    word_target: int = 1600,
    want_summary: bool = True,
    want_cards: bool = True,
    want_guide: bool = True,
) -> Dict[str, Any]:
    """End-to-end pipeline used by both /upload and /intelligent endpoints."""

    if not raw_pdf:
        raise ValueError("Empty file")

    # 1) Extract text
    text_content = extract_text_from_pdf(raw_pdf) or ""
    if len(text_content.strip()) < 100:
        raise ValueError("Could not extract text")

    # 2) High-level classification
    classification_pack = await classify_and_recommend(text_content)
    cls = (classification_pack or {}).get("classification", {}) if isinstance(classification_pack, dict) else {}
    doc_type = (cls.get("document_type") or "document").lower()

    # 3) Choose learning model (extractor)
    route = await choose_learning_model(text_content=text_content, classification=cls)
    learning_model = route["learning_model"]
    mapped_subject_area = route["mapped_subject_area"]

    # 4) Create document id + upload to storage
    doc_id = new_uuid()
    content_hash = sha256_bytes(raw_pdf)
    pdf_path = upload_pdf_to_storage(user_id=user_id, doc_id=doc_id, raw_pdf=raw_pdf, filename=filename)

    # 5) Syllabus path
    if doc_type == "syllabus":
        syllabus_data = await process_syllabus(text_content)

        summary_md = await _make_markdown_summary(text_content, word_target=min(word_target, 1400)) if want_summary else ""

        upsert_document(
            user_id=user_id,
            doc_id=doc_id,
            class_id=class_id,
            title=title or (filename or "Syllabus"),
            summary=summary_md,
            cards_json=json.dumps({"cards": []}),
            guide_json=json.dumps({"concepts": []}),
            pdf_path=pdf_path,
            content_hash=content_hash,
        )

        # Caller can optionally persist syllabus_data into a separate table.
        return {
            "id": doc_id,
            "document_type": "syllabus",
            "learning_model": learning_model,
            "subject_area": mapped_subject_area,
            "classification": cls,
            "routing": route,
            "syllabus_data": syllabus_data,
            "summary": summary_md,
            "cards_json": json.dumps({"cards": []}),
            "guide_json": json.dumps({"concepts": []}),
            "pdf_path": pdf_path,
        }

    # 6) Extraction + materials
    extractor_result = await extract_by_learning_model(learning_model, text_content)
    units = extractor_result.get("units", []) if isinstance(extractor_result, dict) else []

    guide_obj = _to_guide_json(units) if want_guide else {"concepts": []}

    summary_task = _make_markdown_summary(text_content, word_target=word_target) if want_summary else asyncio.sleep(0, result="")
    cards_task = _make_flashcards(units) if want_cards else asyncio.sleep(0, result={"cards": []})

    summary_md, cards_obj = await asyncio.gather(summary_task, cards_task)

    guide_json = json.dumps(guide_obj, ensure_ascii=False)
    cards_json = json.dumps(cards_obj, ensure_ascii=False)

    # 7) Store document correctly
    upsert_document(
        user_id=user_id,
        doc_id=doc_id,
        class_id=class_id,
        title=title or (filename or "Document"),
        summary=summary_md or "",
        cards_json=cards_json,
        guide_json=guide_json,
        pdf_path=pdf_path,
        content_hash=content_hash,
    )

    # 8) Update concept graph (best-effort)
    if want_guide and guide_obj.get("concepts"):
        try:
            await update_class_graph(class_id=class_id, doc_id=doc_id, guide_json=guide_json)
        except Exception as e:
            logger.warning(f"[graph] update_class_graph failed: {e}")

    return {
        "id": doc_id,
        "document_type": doc_type,
        "learning_model": learning_model,
        "subject_area": mapped_subject_area,
        "classification": cls,
        "routing": route,
        "extractor": {
            "rejects": extractor_result.get("rejects", []),
            "coverage_notes": extractor_result.get("coverage_notes", ""),
            "units_count": len(units),
        },
        "summary": summary_md or "",
        "cards_json": cards_json,
        "guide_json": guide_json,
        "pdf_path": pdf_path,
    }
