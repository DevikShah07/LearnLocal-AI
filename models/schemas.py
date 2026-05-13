"""models/schemas.py — all Pydantic v2 request & response models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    MCQ         = "mcq"
    TRUE_FALSE  = "true_false"
    FILL_BLANK  = "fill_blank"
    SHORT       = "short_answer"
    DESCRIPTIVE = "descriptive"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ── Request ────────────────────────────────────────────────────────────────────

class QuestionTypeConfig(BaseModel):
    type: QuestionType
    count: int = Field(default=5, ge=1, le=30)
    marks: int = Field(default=1, ge=1, le=10)

    @model_validator(mode="after")
    def check_count_limits(self) -> "QuestionTypeConfig":
        # Specific limits requested by user
        limits = {
            QuestionType.MCQ: 10,
            QuestionType.TRUE_FALSE: 10,
            QuestionType.SHORT: 5,
            QuestionType.DESCRIPTIVE: 5,
            QuestionType.FILL_BLANK: 10,
        }
        max_allowed = limits.get(self.type, 10)
        if self.count > max_allowed:
            raise ValueError(
                f"Maximum {max_allowed} questions allowed for '{self.type.value}'. "
                f"Requested: {self.count}"
            )
        return self


class GenerateRequest(BaseModel):
    """Body sent alongside the uploaded PDF (as form fields or JSON)."""
    question_types: List[QuestionTypeConfig] = Field(
        default=[QuestionTypeConfig(type=QuestionType.MCQ, count=5, marks=1)],
        description="List of question types with count and marks each.",
    )
    difficulty: Difficulty = Difficulty.MEDIUM
    keyword: Optional[str] = Field(
        default=None,
        description="Optional topic/keyword to focus questions on.",
    )
    page_range: Optional[str] = Field(
        default=None,
        description="Optional page range, e.g. '1-10' or '3,7,12'.",
    )
    top_k: int = Field(
        default=10,
        ge=3, le=30,
        description="Number of chunks to feed LLM as context.",
    )
    language: str = Field(default="English")

    @field_validator("keyword")
    @classmethod
    def strip_keyword(cls, v):
        return v.strip() if v else v


# ── Individual question variants ───────────────────────────────────────────────

class MCQQuestion(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str        # "A" | "B" | "C" | "D"
    explanation: Optional[str] = None
    marks: int = 1


class TrueFalseQuestion(BaseModel):
    question: str
    correct_answer: bool
    explanation: Optional[str] = None
    marks: int = 1


class FillBlankQuestion(BaseModel):
    question: str              # "___ is the loss function used in…"
    answer: str
    marks: int = 1


class ShortAnswerQuestion(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    question: str
    model_answer: str
    marks: int = 2


class DescriptiveQuestion(BaseModel):
    question: str
    key_points: List[str]      # Rubric hints for evaluator
    marks: int = 5


# ── Grouped output ─────────────────────────────────────────────────────────────

class QuestionGroup(BaseModel):
    type: QuestionType
    count: int
    total_marks: int
    questions: List[Any]       # typed above, mixed at runtime


# ── Top-level response ─────────────────────────────────────────────────────────

class ChunkScoreBreakdown(BaseModel):
    chunk_index: int
    score: float
    structure: float
    concept_anchor: float
    heading_proximity: float
    cross_reference: float
    preview: str               # first 120 chars of chunk text


class GenerateResponse(BaseModel):
    status: str = "success"
    document_name: str
    total_pages: int
    chunks_selected: int
    context_chars: int
    difficulty: Difficulty
    keyword: Optional[str]
    language: str

    # Flat list of ALL questions (any type)
    questions: List[Any]

    # Grouped by type
    grouped: List[QuestionGroup]

    # Meta
    total_questions: int
    total_marks: int
    llm_model: str
    processing_time_ms: float

    # FAISS index ID for this document (None if FAISS unavailable)
    faiss_doc_id: Optional[str] = None

    # Debug (optional)
    chunk_scores: Optional[List[ChunkScoreBreakdown]] = None


# ── FAISS search schemas ───────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query to search chunks.")
    k: int     = Field(default=5, ge=1, le=30, description="Number of results to return.")


class ChunkSearchResult(BaseModel):
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    doc_id: str
    query: str
    results: List[ChunkSearchResult]


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    total_chunks: int
    dimension: int
    created_at: float
    size_bytes: int


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    llm_model: str
