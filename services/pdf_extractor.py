"""services/pdf_extractor.py — fast PDF text extraction with heading detection."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pypdf


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PageChunk:
    chunk_index: int
    chunk_text: str
    word_count: int = 0
    is_heading: bool = False

    def __post_init__(self):
        self.word_count = len(self.chunk_text.split())


@dataclass
class PDFDocument:
    name: str
    total_pages: int
    chunks: List[PageChunk]
    headings: List[Tuple[int, str]] = field(default_factory=list)


# ── Heading detection patterns ─────────────────────────────────────────────────

_HEADING_RE = [
    re.compile(r'^\s*\d+(\.\d+)*\s+[A-Z]', re.MULTILINE),    # "3.2 Activation"
    re.compile(r'^[A-Z][A-Z\s]{5,50}$', re.MULTILINE),         # ALL CAPS TITLE
    re.compile(r'^\s*#{1,4}\s+\w', re.MULTILINE),              # Markdown headings
    re.compile(r'^(Chapter|Section|Unit|Part)\s+\d', re.MULTILINE | re.IGNORECASE),
]


def _is_heading_chunk(text: str) -> bool:
    return any(p.search(text) for p in _HEADING_RE)


# ── Page range parser ──────────────────────────────────────────────────────────

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
    # Clamp to valid range
    return [i for i in indices if 0 <= i < total_pages]


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_pdf(file_bytes: bytes, filename: str) -> PDFDocument:
    """
    Extract text page-by-page from a PDF.
    Each page becomes one PageChunk.
    Empty/very short pages (< 20 chars) are skipped.
    """
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    chunks: List[PageChunk] = []
    headings: List[Tuple[int, str]] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if len(text) < 20:          # skip blank/image-only pages
            continue

        is_heading = _is_heading_chunk(text)
        chunk = PageChunk(
            chunk_index=i,
            chunk_text=text,
            is_heading=is_heading,
        )
        chunks.append(chunk)

        if is_heading:
            first_line = text.splitlines()[0].strip()
            headings.append((i, first_line))

    return PDFDocument(
        name=filename,
        total_pages=total_pages,
        chunks=chunks,
        headings=headings,
    )
