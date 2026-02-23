from __future__ import annotations

from fastapi import APIRouter, Request

from ..auth import user_id_from_auth_header

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/whoami")
def whoami(request: Request):
    """Return Supabase user_id if Authorization header contains a valid JWT."""
    user_id = user_id_from_auth_header(request.headers.get("Authorization"))
    return {"user_id": user_id}
