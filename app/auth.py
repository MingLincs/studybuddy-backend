from typing import Optional, Any
from jose import jwt, JWTError
from fastapi import Header
import logging
from .settings import settings

logger = logging.getLogger("auth")

def _get_supabase_secret() -> str:
    """Return the Supabase JWT secret"""
    secret: Any = getattr(settings, "SUPABASE_JWT_SECRET", "")
    if hasattr(secret, "get_secret_value"):
        secret = secret.get_secret_value()
    if not isinstance(secret, str):
        secret = str(secret or "")
    return secret.strip()

def user_id_from_auth_header(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Read Authorization header and return user_id"""
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    secret = _get_supabase_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
        return payload.get("sub") or payload.get("user_id")
    except JWTError:
        return None

get_current_user_id = user_id_from_auth_header