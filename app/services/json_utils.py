"""app/services/json_utils.py

Robust JSON parsing helpers for LLM outputs.

LLMs sometimes wrap JSON in code fences or add leading/trailing text.
These helpers aggressively extract a JSON object/array from a response
so your pipeline doesn't silently fall back to empty outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*|```", re.MULTILINE)


def clean_llm_text(s: str) -> str:
    """Remove markdown code fences and trim."""
    return _FENCE_RE.sub("", s or "").strip()


def extract_json_substring(s: str) -> Optional[str]:
    """Extract the most likely JSON object/array substring.

    Strategy:
    - remove code fences
    - find first '{' or '['
    - find the matching last '}' or ']'
    - return that slice
    """
    t = clean_llm_text(s)
    if not t:
        return None

    # Find first JSON start
    start_obj = t.find("{")
    start_arr = t.find("[")
    if start_obj == -1 and start_arr == -1:
        return None
    if start_obj == -1:
        start = start_arr
        end_char = "]"
    elif start_arr == -1:
        start = start_obj
        end_char = "}"
    else:
        # choose earliest
        start = min(start_obj, start_arr)
        end_char = "}" if start == start_obj else "]"

    # Find last matching end char
    end = t.rfind(end_char)
    if end == -1 or end <= start:
        return None
    return t[start : end + 1]


def safe_json_loads(s: str, *, default: Any = None) -> Any:
    """Best-effort json.loads for LLM responses."""
    if default is None:
        default = {}
    sub = extract_json_substring(s)
    if not sub:
        return default
    try:
        return json.loads(sub)
    except Exception:
        return default
