"""
services/pdf_extractor.py
=========================
PDF text extraction using LangChain's PyPDFLoader +
RecursiveCharacterTextSplitter.

Public API is unchanged:
    extract_pdf(file_bytes, filename) → PDFDocument
    parse_page_range(page_range, total_pages) → Optional[List[int]]

Internal change:
    - PyPDFLoader replaces raw pypdf page iteration
    - RecursiveCharacterTextSplitter produces overlapping sub-chunks
      (CHUNK_SIZE=1000, CHUNK_OVERLAP=200) for finer RAG retrieval
    - LangChain Document objects are converted to PageChunk dataclasses
      so the rest of the pipeline is unaffected
"""

from __future__ import annotations

import io
import re
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from core.config import settings
from core.logger import logger
from services.lc_chain import lc_embeddings


# ── Data classes (unchanged public API) ───────────────────────────────────────

@dataclass
class PageChunk:
    chunk_index: int
    chunk_text: str
    word_count: int = 0
    is_heading: bool = False
    source_page: int = 0          # original PDF page number (0-indexed)

    def __post_init__(self):
        self.word_count = len(self.chunk_text.split())


@dataclass
class PDFDocument:
    name: str
    total_pages: int
    chunks: List[PageChunk]
    headings: List[Tuple[int, str]] = field(default_factory=list)


# ── Heading detection patterns (kept for metadata) ─────────────────────────────

_HEADING_RE = [
    re.compile(r'^\s*\d+(\.\d+)*\s+[A-Z]', re.MULTILINE),    # "3.2 Activation"
    re.compile(r'^[A-Z][A-Z\s]{5,50}$', re.MULTILINE),         # ALL CAPS TITLE
    re.compile(r'^\s*#{1,4}\s+\w', re.MULTILINE),              # Markdown headings
    re.compile(r'^(Chapter|Section|Unit|Part)\s+\d', re.MULTILINE | re.IGNORECASE),
]


def _is_heading_chunk(text: str) -> bool:
    return any(p.search(text) for p in _HEADING_RE)


# ── Page range parser (unchanged) ─────────────────────────────────────────────

def parse_page_range(page_range: Optional[str], total_pages: int) -> Optional[List[int]]:
    """
    Parse '1-5' → [0,1,2,3,4]  or  '3,7,12' → [2,6,11]  (0-indexed).
    Returns None if page_range is None/empty (= use all pages).
    """
    if not page_range:
        return None
    indices: List[int] = []
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo) - 1, int(hi)))
        else:
            indices.append(int(part) - 1)
    return [i for i in indices if 0 <= i < total_pages]


# ── Main extractor (LangChain) ────────────────────────────────────────────────

def extract_pdf(file_bytes: bytes, filename: str) -> PDFDocument:
    """
    Extract and chunk text from a PDF using LangChain.

    Steps:
      1. Write bytes to a temporary file (PyPDFLoader requires a path)
      2. Load with PyPDFLoader → one Document per page
      3. Split with RecursiveCharacterTextSplitter → overlapping sub-chunks
      4. Convert Document objects → PageChunk dataclasses
    """
    # ── 1. Write to temp file ──────────────────────────────────────────────────
    suffix = os.path.splitext(filename)[-1] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── 2. Load PDF ────────────────────────────────────────────────────────
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()          # List[Document], one per page
        total_pages = len(pages)
        logger.info(f"[PDF] Loaded '{filename}': {total_pages} pages")

        if not pages:
            return PDFDocument(name=filename, total_pages=0, chunks=[], headings=[])

        # ── 3. Split into semantic chunks ──────────────────────────────────────
        splitter = SemanticChunker(
            lc_embeddings,
            breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=settings.SEMANTIC_BREAKPOINT_THRESHOLD,
        )
        docs = splitter.split_documents(pages)
        logger.info(
            f"[PDF] Split into {len(docs)} semantic chunks "
            f"(type={settings.SEMANTIC_BREAKPOINT_TYPE}, threshold={settings.SEMANTIC_BREAKPOINT_THRESHOLD})"
        )

        # ── 4. Convert to PageChunk dataclasses ────────────────────────────────
        chunks: List[PageChunk] = []
        headings: List[Tuple[int, str]] = []

        for i, doc in enumerate(docs):
            text = doc.page_content.strip()
            if len(text) < 20:           # skip near-empty chunks
                continue

            source_page = doc.metadata.get("page", 0)   # 0-indexed page from PyPDFLoader
            is_heading = _is_heading_chunk(text)

            chunk = PageChunk(
                chunk_index=i,
                chunk_text=text,
                is_heading=is_heading,
                source_page=source_page,
            )
            chunks.append(chunk)

            if is_heading:
                first_line = text.splitlines()[0].strip()
                headings.append((i, first_line))

        logger.info(f"[PDF] Final chunks: {len(chunks)}, headings: {len(headings)}")
        return PDFDocument(
            name=filename,
            total_pages=total_pages,
            chunks=chunks,
            headings=headings,
        )

    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
