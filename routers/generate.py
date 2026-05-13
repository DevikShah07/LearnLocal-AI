"""
routers/generate.py
===================
POST /api/v1/generate   — multipart form: PDF file + JSON config
POST /api/v1/generate/json — JSON only (for text input without PDF)
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from core.config import settings
from core.logger import logger
from models.schemas import (
    Difficulty,
    GenerateRequest,
    GenerateResponse,
    QuestionTypeConfig,
    QuestionType,
)
from services.pdf_extractor import extract_pdf
from services.pipeline import run_pipeline

router = APIRouter()

_MAX_BYTES = settings.MAX_PDF_SIZE_MB * 1024 * 1024


# ── Helper ─────────────────────────────────────────────────────────────────────

def _parse_config(config_json: Optional[str]) -> GenerateRequest:
    if not config_json:
        return GenerateRequest()
    try:
        return GenerateRequest.model_validate_json(config_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config JSON: {e}")


# ── PDF upload endpoint ────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate questions from a PDF",
    description="""
Upload a PDF and receive structured exam questions.

**config** is a JSON string with shape:
```json
{
  "question_types": [
    {"type": "mcq",         "count": 5, "marks": 1},
    {"type": "true_false",  "count": 3, "marks": 1},
    {"type": "short_answer","count": 2, "marks": 3}
  ],
  "difficulty": "medium",
  "keyword": "neural networks",
  "page_range": "1-20",
  "top_k": 10,
  "language": "English"
}
```
All fields are optional — defaults apply.

**Generation Limits per request:**
- MCQ: Max 10
- True/False: Max 10
- Short Answer: Max 5
- Descriptive: Max 5
- Fill in the Blank: Max 10
""",
)
async def generate_from_pdf(
    file: UploadFile = File(..., description="PDF file to generate questions from"),
    config: Optional[str] = Form(
        default=None,
        description="JSON string of GenerateRequest (all fields optional)",
    ),
    debug: bool = Query(default=False, description="Include chunk scores in response"),
):
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read and size-check
    file_bytes = await file.read()
    if len(file_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF too large. Max size: {settings.MAX_PDF_SIZE_MB}MB.",
        )
    if len(file_bytes) < 100:
        raise HTTPException(status_code=400, detail="PDF appears to be empty.")

    logger.info(f"Received PDF: {file.filename} ({len(file_bytes)//1024}KB)")

    # Parse config
    req = _parse_config(config)

    # Extract PDF
    try:
        doc = extract_pdf(file_bytes, file.filename)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    if not doc.chunks:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from this PDF. It may be image-based.",
        )

    logger.info(f"PDF extracted: {doc.total_pages} pages, {len(doc.chunks)} text chunks.")

    # Run pipeline
    try:
        response = await run_pipeline(
            doc=doc,
            question_types=req.question_types,
            difficulty=req.difficulty,
            keyword=req.keyword,
            page_range=req.page_range,
            top_k=req.top_k,
            language=req.language,
            include_chunk_scores=debug,
            pdf_size_bytes=len(file_bytes),
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {e}")

    return response


# ── Plain text endpoint (no PDF) ───────────────────────────────────────────────

@router.post(
    "/generate/text",
    response_model=GenerateResponse,
    summary="Generate questions from raw text",
)
async def generate_from_text(
    text: str = Form(..., description="Raw document text"),
    config: Optional[str] = Form(default=None),
    debug: bool = Query(default=False),
):
    from services.pdf_extractor import PageChunk, PDFDocument

    req = _parse_config(config)

    # Wrap plain text as a single-chunk document
    chunks = []
    # Split into ~1000-char chunks to enable scoring
    step = 1000
    raw = text.strip()
    for i, start in enumerate(range(0, len(raw), step)):
        chunk_text = raw[start: start + step]
        if chunk_text.strip():
            chunks.append(PageChunk(chunk_index=i, chunk_text=chunk_text))

    doc = PDFDocument(
        name="text_input.txt",
        total_pages=len(chunks),
        chunks=chunks,
        headings=[],
    )

    try:
        response = await run_pipeline(
            doc=doc,
            question_types=req.question_types,
            difficulty=req.difficulty,
            keyword=req.keyword,
            page_range=req.page_range,
            top_k=req.top_k,
            language=req.language,
            include_chunk_scores=debug,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {e}")

    return response
