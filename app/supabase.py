from __future__ import annotations

from typing import Any
from .services.db import supabase as _get_supabase


class _SupabaseProxy:
    """
    Lazy proxy so we don't create the Supabase client at import time.
    Keeps existing code working (supabase.table(...)).
    """
    def __getattr__(self, name: str) -> Any:
        return getattr(_get_supabase(), name)


supabase = _SupabaseProxy()