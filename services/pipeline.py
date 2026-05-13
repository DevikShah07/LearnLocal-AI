"""
services/pipeline.py
====================
Orchestrates the full PDF → FAISS MMR → parallel LangChain LLM → response pipeline.

Key changes from raw-Python version:
  - Chunk selection: custom 4-signal scorer replaced by FAISS MMR search
  - Parallel LLM: asyncio.gather replaced by LangChain RunnableParallel
  - Output parsing: handled by JsonOutputParser inside each chain
  - FAISS indexing: uses LangChain FAISS save_local()
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from core.config import settings
from core.logger import logger
from models.schemas import (
    Difficulty,
    GenerateResponse,
    QuestionGroup,
    QuestionTypeConfig,
)
from services.embedder import embedder
from services.faiss_store import faiss_store
from services.lc_chain import build_parallel_chain
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
    pdf_size_bytes: int = 0,
) -> GenerateResponse:
    t0 = time.perf_counter()

    # ── 1. Page range filter ──────────────────────────────────────────────────
    page_indices = parse_page_range(page_range, doc.total_pages)
    working_chunks = doc.chunks
    if page_indices is not None:
        idx_set = set(page_indices)
        filtered = [c for c in doc.chunks if c.chunk_index in idx_set]
        working_chunks = filtered if filtered else doc.chunks

    # ── 2. Embed all chunks (batch) ───────────────────────────────────────────
    all_chunk_texts = [c.chunk_text for c in working_chunks]
    chunk_embeddings = embedder.encode(all_chunk_texts)  # (N, D) or None

    if chunk_embeddings is None:
        logger.warning("Embedding model unavailable — using first top_k chunks as context.")
        selected_texts = all_chunk_texts[:top_k]
    else:
        logger.info(f"[Pipeline] Embeddings computed: {len(all_chunk_texts)} chunks.")

        # ── 3a. Persist FAISS index ───────────────────────────────────────────
        # Build the index now so we can use it for MMR chunk selection below
        faiss_doc_id_tmp: Optional[str] = None
        vectorstore = None
        if faiss_store.available:
            try:
                from langchain_community.vectorstores import FAISS

                text_embedding_pairs = list(zip(all_chunk_texts, chunk_embeddings.tolist()))
                vectorstore = FAISS.from_embeddings(
                    text_embeddings=text_embedding_pairs,
                    embedding=embedder._model if hasattr(embedder, "_model") else None,
                    metadatas=[{"chunk_index": i} for i in range(len(all_chunk_texts))],
                )
                # We'll persist after pipeline succeeds
            except Exception as exc:
                logger.warning(f"[FAISS] In-memory build failed (non-fatal): {exc}")

        # ── 3b. MMR chunk selection ───────────────────────────────────────────
        if vectorstore is not None and keyword:
            try:
                fetch_k = min(top_k * 3, len(all_chunk_texts))
                mmr_docs = vectorstore.max_marginal_relevance_search(
                    query=keyword,
                    k=top_k,
                    fetch_k=fetch_k,
                )
                selected_texts = [d.page_content for d in mmr_docs]
                logger.info(
                    f"[Pipeline] MMR selected {len(selected_texts)} chunks for keyword '{keyword}'."
                )
            except Exception as exc:
                logger.warning(f"[Pipeline] MMR failed, falling back to top-k: {exc}")
                selected_texts = all_chunk_texts[:top_k]
        elif vectorstore is not None:
            # No keyword → similarity search against a generic academic query
            try:
                sim_docs = vectorstore.similarity_search(
                    query="important concepts and definitions",
                    k=top_k,
                )
                selected_texts = [d.page_content for d in sim_docs]
                logger.info(f"[Pipeline] Similarity selected {len(selected_texts)} chunks.")
            except Exception as exc:
                logger.warning(f"[Pipeline] Similarity search failed, using top-k: {exc}")
                selected_texts = all_chunk_texts[:top_k]
        else:
            selected_texts = all_chunk_texts[:top_k]

    # Context window guard: trim if too long
    max_chars = settings.MAX_CONTEXT_TOKENS * 4
    trimmed_texts: List[str] = []
    total_chars = 0
    for text in selected_texts:
        if total_chars + len(text) > max_chars:
            break
        trimmed_texts.append(text)
        total_chars += len(text)
    selected_texts = trimmed_texts

    # ── 4. Build context string ───────────────────────────────────────────────
    context = "\n\n".join(
        f"[Chunk {i + 1}]\n{text}"
        for i, text in enumerate(selected_texts)
    )
    context_chars = len(context)
    logger.info(
        f"[Pipeline] Context: {len(selected_texts)} chunks, {context_chars} chars."
    )

    # ── 5. Parallel LLM generation via Robust LLMClient ──────────────────────
    from services.llm_client import llm_client

    logger.info(
        f"[Pipeline] Running robust parallel generation for "
        f"{[cfg.type.value for cfg in question_types]}"
    )

    raw_results: Dict[str, Any] = {}
    try:
        tasks = [
            llm_client.generate_questions_for_type(
                cfg=cfg,
                context=context,
                difficulty=difficulty,
                keyword=keyword,
                language=language
            ) for cfg in question_types
        ]
        
        # Execute all question generation tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map the results back to their respective question types
        for cfg, result in zip(question_types, results):
            raw_results[cfg.type.value] = result
            
    except Exception as exc:
        logger.error(f"[Pipeline] Parallel LLM call failed: {exc}", exc_info=True)

    # ── 6. Assemble output ────────────────────────────────────────────────────
    # Map question type value → config for mark lookup
    cfg_map = {cfg.type.value: cfg for cfg in question_types}

    all_questions: List[Any] = []
    grouped: List[QuestionGroup] = []

    for type_value, questions in raw_results.items():
        if isinstance(questions, Exception):
            logger.error(f"[Pipeline] LLM error for {type_value}: {questions}", exc_info=questions)
            continue
        if not isinstance(questions, list):
            logger.warning(f"[Pipeline] Unexpected result type for {type_value}: {type(questions)}")
            continue

        cfg = cfg_map.get(type_value)
        if cfg is None:
            continue

        # Stamp the type onto each question
        for q in questions:
            if isinstance(q, dict):
                q["type"] = type_value

        all_questions.extend(questions)
        grouped.append(QuestionGroup(
            type=cfg.type,
            count=len(questions),
            total_marks=sum(q.get("marks", cfg.marks) for q in questions if isinstance(q, dict)),
            questions=questions,
        ))

    total_marks = sum(g.total_marks for g in grouped)
    elapsed_ms  = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        f"[Pipeline] Done in {elapsed_ms}ms. "
        f"{len(all_questions)} questions generated."
    )

    # ── 7. Persist FAISS index ────────────────────────────────────────────────
    faiss_doc_id: Optional[str] = None
    if chunk_embeddings is not None and faiss_store.available:
        try:
            loop = asyncio.get_event_loop()
            faiss_doc_id = await loop.run_in_executor(
                None,
                lambda: faiss_store.save(
                    doc_name=doc.name,
                    chunk_texts=all_chunk_texts,
                    embeddings=chunk_embeddings,
                    pdf_size_bytes=pdf_size_bytes,
                ),
            )
            logger.info(f"[FAISS] Indexed '{doc.name}' → {faiss_doc_id}")
        except Exception as exc:
            logger.warning(f"[FAISS] Index save failed (non-fatal): {exc}")

    return GenerateResponse(
        status="success",
        document_name=doc.name,
        total_pages=doc.total_pages,
        chunks_selected=len(selected_texts),
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
        chunk_scores=None,            # chunk_scorer removed; use /search for FAISS scores
        faiss_doc_id=faiss_doc_id,
    )
