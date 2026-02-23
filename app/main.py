from __future__ import annotations

import sys
import asyncio

# IMPORTANT: Playwright needs subprocess support on Windows.
# SelectorEventLoop on Windows can raise NotImplementedError for subprocesses.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from .settings import settings
from .routers import upload, quiz, debug, library
from .routers.classes import router as classes_router
from .routers.documents import router as documents_router
from .routers.class_graph import router as class_graph_router
from .routers.concepts_detail import router as concepts_detail_router
from .routers.edges_detail import router as edges_detail_router
from .routers.concept_merge import router as concept_merge_router
from .routers.graph_jobs import router as graph_jobs_router
from .routers.intelligent_processing import router as intelligent_router
from .routers.syllabus import router as syllabus_router
from .routers.calendar import router as calendar_router
# ---------- logging ----------
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
)


# ---------- app / limiter ----------
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
app = FastAPI(title="StudyBuddy API", version="1.1.0")
app.state.limiter = limiter


# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["Authorization"],
)


# SlowAPI middleware + handler
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------- routes ----------
app.include_router(debug.router)
app.include_router(upload.router)
app.include_router(quiz.router)
app.include_router(library.router)
app.include_router(classes_router)
app.include_router(documents_router)
app.include_router(class_graph_router)
app.include_router(concepts_detail_router)
app.include_router(edges_detail_router)
app.include_router(concept_merge_router)
app.include_router(graph_jobs_router)
app.include_router(intelligent_router)
app.include_router(syllabus_router)
app.include_router(calendar_router)