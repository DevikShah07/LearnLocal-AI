"""
services/pipeline.py
====================
Orchestrates the full PDF → context → parallel LLM → structured response pipeline.

Key optimisations:
  - Embeddings computed in one batch (not per-chunk)
  - All question types generated in PARALLEL with asyncio.gather
  - Context assembled once and reused across all LLM calls
  - Chunk scoring is synchronous but fast (<50ms for 100-page docs)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from core.config import settings
from core.logger import logger
from models.schemas import (
    ChunkScoreBreakdown,
    Difficulty,
    GenerateResponse,
    QuestionGroup,
    QuestionTypeConfig,
)
from services.chunk_scorer import ScoredChunk, select_top_chunks
from services.embedder import embedder
from services.llm_client import llm_client
from services.pdf_extractor import PDFDocument, parse_page_range


async def run_pipeline(
    doc: PDFDocument,
    question_types: List[QuestionTypeConfig],
    difficulty: Difficulty,
    keyword: Optional[str],
    page_range: Optional[str],
    top_k: int,
    language: str,
    include_chunk_scores: bool = False,
) -> GenerateResponse:
    t0 = time.perf_counter()

    # ── 1. Page range filter ──────────────────────────────────────────────────
    page_indices = parse_page_range(page_range, doc.total_pages)

    # ── 2. Embeddings (batch, fast) ───────────────────────────────────────────
    query_embedding = None
    chunk_embeddings = None

    if keyword:
        texts = [c.chunk_text for c in doc.chunks]
        all_vecs = embedder.encode([keyword] + texts)
        if all_vecs is not None:
            query_embedding  = all_vecs[0]
            chunk_embeddings = all_vecs[1:]
            logger.info(f"Semantic embeddings computed: {len(texts)} chunks.")

    # ── 3. Smart chunk selection ──────────────────────────────────────────────
    selected_chunks, all_scored = select_top_chunks(
        chunks=doc.chunks,
        headings=doc.headings,
        top_k=top_k,
        keyword=keyword,
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        page_indices=page_indices,
        max_context_chars=settings.MAX_CONTEXT_TOKENS * 4,  # ~4 chars/token
    )

    # ── 4. Build context string ───────────────────────────────────────────────
    context = "\n\n".join(
        f"[Page {c.chunk_index + 1}]\n{c.chunk_text}"
        for c in selected_chunks
    )
    context_chars = len(context)
    logger.info(
        f"Context: {len(selected_chunks)} chunks, {context_chars} chars selected."
    )

    # ── 5. Parallel LLM calls (one per question type) ─────────────────────────
    async def generate_one(cfg: QuestionTypeConfig) -> tuple[QuestionTypeConfig, List[Dict]]:
        logger.info(f"Generating {cfg.count}× {cfg.type.value}…")
        questions = await llm_client.generate_questions_for_type(
            cfg=cfg,
            context=context,
            difficulty=difficulty,
            keyword=keyword,
            language=language,
        )
        return cfg, questions

    results = await asyncio.gather(
        *[generate_one(cfg) for cfg in question_types],
        return_exceptions=True,
    )

    # ── 6. Assemble output ────────────────────────────────────────────────────
    all_questions: List[Any] = []
    grouped: List[QuestionGroup] = []

    for outcome in results:
        if isinstance(outcome, Exception):
            logger.error(f"LLM call failed: {outcome}")
            continue
        cfg, questions = outcome
        # Stamp the type onto each question for flat list clarity
        for q in questions:
            q["type"] = cfg.type.value
        all_questions.extend(questions)
        grouped.append(QuestionGroup(
            type=cfg.type,
            count=len(questions),
            total_marks=sum(q.get("marks", cfg.marks) for q in questions),
            questions=questions,
        ))

    total_marks = sum(g.total_marks for g in grouped)
    elapsed_ms  = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"Pipeline done in {elapsed_ms}ms. {len(all_questions)} questions generated.")

    # ── 7. Optional chunk score debug output ──────────────────────────────────
    chunk_scores_out: Optional[List[ChunkScoreBreakdown]] = None
    if include_chunk_scores:
        chunk_scores_out = [
            ChunkScoreBreakdown(
                chunk_index=sc.chunk.chunk_index,
                score=sc.score,
                structure=sc.breakdown.get("structure", 0),
                concept_anchor=sc.breakdown.get("concept_anchor", 0),
                heading_proximity=sc.breakdown.get("heading_proximity", 0),
                cross_reference=sc.breakdown.get("cross_reference", 0),
                preview=sc.preview,
            )
            for sc in all_scored[:20]   # top 20 for debug
        ]

    return GenerateResponse(
        status="success",
        document_name=doc.name,
        total_pages=doc.total_pages,
        chunks_selected=len(selected_chunks),
        context_chars=context_chars,
        difficulty=difficulty,
        keyword=keyword,
        language=language,
        questions=all_questions,
        grouped=grouped,
        total_questions=len(all_questions),
        total_marks=total_marks,
        llm_model=settings.LLM_MODEL,
        processing_time_ms=elapsed_ms,
        chunk_scores=chunk_scores_out,
    )
