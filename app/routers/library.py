# app/routers/library.py
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response
from uuid import UUID
import httpx
import re
import asyncio

from playwright.sync_api import sync_playwright
from markdown_it import MarkdownIt

from ..settings import settings
from ..auth import user_id_from_auth_header
from ..services.db import create_signed_download_url, delete_storage_object, supabase

router = APIRouter(prefix="/library", tags=["library"])

SUPABASE_URL = settings.SUPABASE_URL
SERVICE_KEY = settings.SUPABASE_SERVICE_ROLE_KEY

SR_HEADERS = {
    "apikey": SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type": "application/json",
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

async def _get_user_id_from_token(authorization: str | None) -> str | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None

    token = authorization.split(" ", 1)[1]

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SERVICE_KEY,
            },
        )

    if r.status_code != 200:
        return None

    return (r.json() or {}).get("id")


async def _ensure_owner(table: str, row_id: str, user_id: str):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}&select=id,user_id",
            headers=SR_HEADERS,
        )

    if r.status_code != 200 or not r.json():
        raise HTTPException(status_code=404, detail="Not found")

    row = r.json()[0]
    if row.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your row")


# --------------------------------------------------
# Download original uploaded PDF
# --------------------------------------------------

@router.get("/document/{doc_id}/download")
async def download_document(
    doc_id: str,
    mode: str = Query(default="download", pattern="^(download|inline)$"),
    Authorization: str | None = Header(default=None),
):
    try:
        UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document id")

    user_id = (
        user_id_from_auth_header(Authorization)
        or await _get_user_id_from_token(Authorization)
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    sb = supabase()
    r = (
        sb.table("documents")
        .select("id,user_id,pdf_path,title")
        .eq("id", doc_id)
        .limit(1)
        .execute()
    )

    rows = getattr(r, "data", None) or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")

    row = rows[0]
    if row.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your row")

    pdf_path = row.get("pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=400, detail="No stored PDF")

    url = create_signed_download_url(object_path=pdf_path)
    return {"url": url, "mode": mode}
def _normalize_math_delimiters(md: str) -> str:
    """
    MarkdownIt(commonmark) consumes backslashes before punctuation, so \\( \\) and \\[ \\]
    often lose the backslash and KaTeX auto-render can't detect math.
    Convert them into $...$ and $$...$$ BEFORE markdown rendering.

    We do this only outside triple-backtick code fences.
    """
    # Split by fenced code blocks; keep the fences in the result
    parts = re.split(r"(```.*?```)", md, flags=re.DOTALL)

    for i in range(0, len(parts), 2):  # only outside code blocks
        s = parts[i]

        # Handle both single and double backslashes (depending on how it was stored/escaped)
        # Replace display first
        s = s.replace("\\\\[", "$$").replace("\\\\]", "$$")
        s = s.replace("\\[", "$$").replace("\\]", "$$")

        # Replace inline
        s = s.replace("\\\\(", "$").replace("\\\\)", "$")
        s = s.replace("\\(", "$").replace("\\)", "$")

        parts[i] = s

    return "".join(parts)

# --------------------------------------------------
# Summary → PDF export (Markdown -> HTML -> KaTeX -> Print)
# --------------------------------------------------

def _markdown_to_html(md: str) -> str:
    md_parser = MarkdownIt("commonmark")
    html_body = md_parser.render(md)

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      padding: 42px;
      line-height: 1.5;
      font-size: 14px;
      color: #111;
    }}
    h1 {{ font-size: 26px; margin: 0 0 14px; }}
    h2 {{ font-size: 20px; margin: 18px 0 10px; }}
    h3 {{ font-size: 16px; margin: 14px 0 8px; }}
    p  {{ margin: 8px 0; }}
    ul {{ margin: 8px 0 8px 22px; }}
    li {{ margin: 4px 0; }}
    code, pre {{ background: #f6f6f6; padding: 2px 4px; border-radius: 4px; }}
    pre {{ padding: 10px; overflow-x: auto; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 16px 0; }}
  </style>

  <!-- KaTeX -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
</head>
<body>
  {html_body}

  <script>
    // Playwright will wait on this flag before printing to PDF.
    window.__katex_done = false;

    function runKatex() {{
      if (typeof renderMathInElement !== "function") {{
        // If KaTeX fails to load, don't hang forever.
        window.__katex_done = true;
        return;
      }}

      try {{
        renderMathInElement(document.body, {{
          delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$",  right: "$",  display: false}},
            {{left: "\\\\(", right: "\\\\)", display: false}},
            {{left: "\\\\[", right: "\\\\]", display: true}}
          ],
          throwOnError: false
        }});
      }} finally {{
        window.__katex_done = true;
      }}
    }}

    window.addEventListener("load", () => {{
      // Let layout settle, then render math.
      setTimeout(runKatex, 0);
    }});
  </script>
</body>
</html>
"""


def _build_pdf_sync(html: str) -> bytes:
    # Runs in a background thread, avoids Windows asyncio subprocess issues.
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # More reliable than "networkidle" when loading CDN assets like KaTeX.
        page.set_content(html, wait_until="load")

        # Wait for KaTeX to finish rendering before printing.
        page.wait_for_function("window.__katex_done === true", timeout=15000)

        # Small extra settle time so layout is final.
        page.wait_for_timeout(200)

        pdf_bytes = page.pdf(
            format="Letter",
            margin={"top": "0.5in", "right": "0.5in", "bottom": "0.5in", "left": "0.5in"},
            print_background=True,
        )

        browser.close()
        return pdf_bytes


async def _build_pdf(title: str, md: str) -> bytes:
    full_md = f"# {title}\n\n{md}"
    full_md = _normalize_math_delimiters(full_md)
    html = _markdown_to_html(full_md)
    return await asyncio.to_thread(_build_pdf_sync, html)


@router.get("/document/{doc_id}/summary-pdf")
async def download_summary_pdf(
    doc_id: str,
    Authorization: str | None = Header(default=None),
):
    try:
        UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document id")

    user_id = (
        user_id_from_auth_header(Authorization)
        or await _get_user_id_from_token(Authorization)
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    sb = supabase()
    r = (
        sb.table("documents")
        .select("id,user_id,title,summary")
        .eq("id", doc_id)
        .limit(1)
        .execute()
    )

    rows = getattr(r, "data", None) or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")

    row = rows[0]
    if row.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your row")

    summary = row.get("summary") or ""
    if not summary.strip():
        raise HTTPException(status_code=400, detail="No summary available")

    pdf_bytes = await _build_pdf(row.get("title") or "Summary", summary)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="summary.pdf"'},
    )


# --------------------------------------------------
# Delete Document
# --------------------------------------------------

@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str, Authorization: str | None = Header(default=None)):
    try:
        UUID(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document id")

    user_id = (
        user_id_from_auth_header(Authorization)
        or await _get_user_id_from_token(Authorization)
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await _ensure_owner("documents", doc_id, user_id)

    sb = supabase()
    r = (
        sb.table("documents")
        .select("pdf_path")
        .eq("id", doc_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    row = getattr(r, "data", None) or {}
    pdf_path = row.get("pdf_path")

    sb.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()

    if pdf_path:
        delete_storage_object(object_path=pdf_path)

    return {"ok": True}


# --------------------------------------------------
# Delete Quiz
# --------------------------------------------------

@router.delete("/quiz/{quiz_id}")
async def delete_quiz(quiz_id: str, Authorization: str | None = Header(default=None)):
    try:
        UUID(quiz_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid quiz id")

    user_id = (
        user_id_from_auth_header(Authorization)
        or await _get_user_id_from_token(Authorization)
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await _ensure_owner("quizzes", quiz_id, user_id)

    supabase().table("quizzes").delete().eq("id", quiz_id).eq("user_id", user_id).execute()
    return {"ok": True}