"""
LearnLocal Question Generation — FastAPI Question Generation Service
=============================================
Production-ready API for PDF → smart chunk selection → LLM question generation.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.logger import logger
from routers import generate, health, search


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LearnLocal Question Generation API starting up…")
    # Pre-warm embedding model so first request is fast
    from services.embedder import embedder
    embedder.warmup()
    logger.info("Embedding model warmed up.")
    yield
    logger.info("LearnLocal Question Generation API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LearnLocal Question Generation Question Generation API",
    description="Upload a PDF, get structured exam questions powered by LLMs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({elapsed}ms)")
    return response


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(health.router, tags=["Health"])
app.include_router(generate.router, prefix="/api/v1", tags=["Question Generation"])
app.include_router(search.router,   prefix="/api/v1", tags=["Vector Search"])

