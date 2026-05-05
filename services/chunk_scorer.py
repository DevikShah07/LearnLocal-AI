"""
services/chunk_scorer.py
========================
4-signal composite importance scorer.
Signal 1: Structure (lists, enumerations, "types of")
Signal 2: Concept anchor (acronyms, formulas, named terms)
Signal 3: Heading proximity
Signal 4: Cross-reference count

When a keyword/query is given, semantic similarity (via embedder) is added
as Signal 5 and weighted most heavily.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from services.pdf_extractor import PageChunk


# ── Pattern banks ──────────────────────────────────────────────────────────────

_STRUCTURE = [
    re.compile(r'\b(types?|kinds?|categories|forms?|classes|variants?)\s+of\b', re.I),
    re.compile(r'^\s*\d+[\.\)]\s+\w', re.M),
    re.compile(r'^\s*[-•*–]\s+\w', re.M),
    re.compile(r'\b(following|below|listed|namely)\s*:', re.I),
    re.compile(r'\b(first|second|third|fourth|finally)[,\s]', re.I),
]

_CONCEPT = [
    re.compile(r'\b[A-Z]{2,7}\b'),
    re.compile(r'[a-zA-Z_]+\s*\([^)]{1,30}\)'),
    re.compile(
        r'\b(algorithm|equation|theorem|formula|architecture|'
        r'layer|function|model|network|method|technique|'
        r'framework|protocol|principle|law|definition)\b', re.I
    ),
    re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
]


# ── Scored chunk result ────────────────────────────────────────────────────────

@dataclass
class ScoredChunk:
    chunk: PageChunk
    score: float
    breakdown: dict = field(default_factory=dict)

    @property
    def preview(self) -> str:
        return self.chunk.chunk_text[:120].replace("\n", " ")


# ── Individual signal functions ────────────────────────────────────────────────

def _s1_structure(text: str) -> float:
    hits = sum(1 for p in _STRUCTURE if p.search(text))
    return min(hits * 0.12, 0.35)


def _s2_concept(text: str) -> float:
    matches: Set[str] = set()
    for p in _CONCEPT:
        matches.update(m.strip() for m in p.findall(text))
    meaningful = {m for m in matches if len(m) >= 2}
    return min(len(meaningful) * 0.04, 0.30)


def _s3_heading(chunk_index: int, heading_indices: Set[int]) -> float:
    if chunk_index in heading_indices:
        return 0.20
    if (chunk_index - 1) in heading_indices:
        return 0.12
    if (chunk_index - 2) in heading_indices:
        return 0.06
    return 0.0


def _s4_cross_ref(chunk_index: int, all_texts: List[str]) -> float:
    page_num = chunk_index + 1
    pat = re.compile(
        rf'\b(page|section|chapter|figure|table|eq\.?)\s*\.?\s*{page_num}\b', re.I
    )
    count = sum(
        1 for i, t in enumerate(all_texts)
        if i != chunk_index and pat.search(t)
    )
    return min(count * 0.05, 0.15)


# ── Main scorer ────────────────────────────────────────────────────────────────

def score_chunks(
    chunks: List[PageChunk],
    headings: List[Tuple[int, str]],
    keyword: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    chunk_embeddings: Optional[np.ndarray] = None,
) -> List[ScoredChunk]:
    """
    Score every chunk and return ScoredChunk list sorted best → worst.

    When query_embedding + chunk_embeddings are provided (keyword given),
    semantic similarity is Signal 5 and weighted 0.40 (dominant signal).
    """
    all_texts = [c.chunk_text for c in chunks]
    heading_indices = {idx for idx, _ in headings}
    has_semantic = (
        query_embedding is not None
        and chunk_embeddings is not None
        and len(chunk_embeddings) == len(chunks)
    )

    # Pre-compute semantic similarities in one vectorised call
    if has_semantic:
        sims = chunk_embeddings @ query_embedding      # (N,) cosine sims

    results: List[ScoredChunk] = []
    for i, chunk in enumerate(chunks):
        text = chunk.chunk_text

        s1 = _s1_structure(text)
        s2 = _s2_concept(text)
        s3 = _s3_heading(chunk.chunk_index, heading_indices)
        s4 = _s4_cross_ref(chunk.chunk_index, all_texts)

        if has_semantic:
            s5 = float(sims[i])
            # Re-weight: semantic is king when keyword given
            total = min(s1 * 0.15 + s2 * 0.15 + s3 * 0.10 + s4 * 0.05 + s5 * 0.40, 1.0)
        else:
            s5 = 0.0
            total = min(s1 + s2 + s3 + s4, 1.0)

        results.append(ScoredChunk(
            chunk=chunk,
            score=round(total, 4),
            breakdown={
                "structure":         round(s1, 3),
                "concept_anchor":    round(s2, 3),
                "heading_proximity": round(s3, 3),
                "cross_reference":   round(s4, 3),
                "semantic_sim":      round(s5, 3),
            },
        ))

    results.sort(key=lambda sc: sc.score, reverse=True)
    return results


def select_top_chunks(
    chunks: List[PageChunk],
    headings: List[Tuple[int, str]],
    top_k: int = 10,
    keyword: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    chunk_embeddings: Optional[np.ndarray] = None,
    page_indices: Optional[List[int]] = None,
    max_context_chars: int = 24_000,
) -> Tuple[List[PageChunk], List[ScoredChunk]]:
    """
    Returns (selected_chunks, all_scored_chunks).

    Steps:
      1. Filter by page_indices if given.
      2. Score all chunks.
      3. Take top 2×k by score → candidate pool.
      4. Re-sort by document position (spread across doc).
      5. Evenly sub-sample to top_k.
      6. Trim so total chars ≤ max_context_chars (context window guard).
    """
    working = chunks
    if page_indices is not None:
        idx_set = set(page_indices)
        working = [c for c in chunks if c.chunk_index in idx_set]
        if not working:
            working = chunks        # fallback: use all if filter is too aggressive

    all_scored = score_chunks(
        working, headings, keyword, query_embedding, chunk_embeddings
    )

    # Pool: top 2×k
    pool = [sc.chunk for sc in all_scored[: top_k * 2]]

    # Spread by position
    pool.sort(key=lambda c: c.chunk_index)

    if len(pool) <= top_k:
        selected = pool
    else:
        step = len(pool) / top_k
        selected = [pool[int(i * step)] for i in range(top_k)]

    # Context window guard: trim if too long
    trimmed: List[PageChunk] = []
    total_chars = 0
    for chunk in selected:
        if total_chars + len(chunk.chunk_text) > max_context_chars:
            break
        trimmed.append(chunk)
        total_chars += len(chunk.chunk_text)

    return trimmed, all_scored
