# app/routers/library.py
from fastapi import APIRouter, Header, HTTPException
import os, requests
from uuid import UUID

router = APIRouter(prefix="/library", tags=["library"])

SUPABASE_URL = os.environ["SUPABASE_URL"]            # e.g. https://xxxx.supabase.co
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # server-only service role
SR_HEADERS = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}

def get_user_id_from_token(authorization: str | None) -> str | None:
  if not authorization or not authorization.lower().startswith("bearer "):
    return None
  token = authorization.split(" ", 1)[1]
  r = requests.get(f"{SUPABASE_URL}/auth/v1/user",
                   headers={"Authorization": f"Bearer {token}", "apikey": SERVICE_KEY})
  if r.status_code != 200:
    return None
  return r.json().get("id")

def ensure_owner(table: str, row_id: str, user_id: str):
  r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}&select=id,user_id", headers=SR_HEADERS)
  if r.status_code != 200 or not r.json():
    raise HTTPException(status_code=404, detail="Not found")
  row = r.json()[0]
  if row.get("user_id") != user_id:
    raise HTTPException(status_code=403, detail="Not your row")

@router.delete("/document/{doc_id}")
def delete_document(doc_id: str, authorization: str | None = Header(default=None)):
  try:
    UUID(doc_id)
  except Exception:
    raise HTTPException(status_code=400, detail="Invalid document id")
  user_id = get_user_id_from_token(authorization)
  if not user_id:
    raise HTTPException(status_code=401, detail="Unauthorized")

  ensure_owner("documents", doc_id, user_id)

  # If you DO NOT have FK cascade, delete child quizzes first (harmless if none).
  requests.delete(f"{SUPABASE_URL}/rest/v1/quizzes?doc_id=eq.{doc_id}", headers=SR_HEADERS)

  r = requests.delete(f"{SUPABASE_URL}/rest/v1/documents?id=eq.{doc_id}", headers=SR_HEADERS)
  if r.status_code not in (200, 204):
    raise HTTPException(status_code=400, detail=r.text)
  return {"ok": True}

@router.delete("/quiz/{quiz_id}")
def delete_quiz(quiz_id: str, authorization: str | None = Header(default=None)):
  try:
    UUID(quiz_id)
  except Exception:
    raise HTTPException(status_code=400, detail="Invalid quiz id")
  user_id = get_user_id_from_token(authorization)
  if not user_id:
    raise HTTPException(status_code=401, detail="Unauthorized")

  ensure_owner("quizzes", quiz_id, user_id)

  r = requests.delete(f"{SUPABASE_URL}/rest/v1/quizzes?id=eq.{quiz_id}", headers=SR_HEADERS)
  if r.status_code not in (200, 204):
    raise HTTPException(status_code=400, detail=r.text)
  return {"ok": True}
