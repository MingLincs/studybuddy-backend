import re
import asyncio
import fitz  # PyMuPDF
from fastapi import HTTPException
from ..settings import settings
from .llm import llm
from .cache import read_bullets, save_bullets


def extract_pages_text(pdf_path: str) -> list[str]:
    """Extract text from PDF file by path"""
    doc = fitz.open(pdf_path)
    out = []
    for p in doc:
        t = p.get_text() or ""
        t = re.sub(r"[ \t]+", " ", t).strip()
        out.append(t)
    return out


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from PDF bytes.
    Used by intelligent processing to get full document text.
    
    Args:
        pdf_bytes: PDF file as bytes
        
    Returns:
        Full text from all pages
    """
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract text from all pages
        text_parts = []
        for page in doc:
            text = page.get_text()
            if text:
                # Clean up whitespace
                text = re.sub(r"[ \t]+", " ", text)
                text_parts.append(text.strip())
        
        doc.close()
        
        # Join all pages with double newline
        full_text = "\n\n".join(text_parts)
        return full_text
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


async def build_bullets_from_pdf(tmp_path: str, doc_id: str) -> tuple[str, list[str]]:
    """Build bullet points from PDF file"""
    cached = read_bullets(doc_id)
    if cached:
        return cached["joined"], cached["bullets"]

    pages = extract_pages_text(tmp_path)
    if not any(p.strip() for p in pages):
        raise HTTPException(422, "No extractable text found (image-only PDF).")

    sem = asyncio.Semaphore(settings.CONCURRENCY)

    async def one(idx: int, txt: str):
        if not txt:
            return None
        snippet = txt[:1500]
        async with sem:
            b = await llm(
                [
                    {"role": "system", "content": "Return 3–6 dense, exam-focused bullets. No preface, no conclusion."},
                    {"role": "user", "content": f"Slide {idx} text:\n{snippet}"}
                ],
                max_tokens=220,
                temperature=0.2
            )
            return f"Slide {idx}:\n{b}"

    tasks = [one(i, t) for i, t in enumerate(pages[:settings.MAX_PAGES], start=1)]
    results = [r for r in await asyncio.gather(*tasks) if r]
    joined = "\n\n".join(results) if results else "No text found."

    save_bullets(doc_id, joined, results)
    return joined, results