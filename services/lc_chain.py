"""
services/lc_chain.py
====================
Central LangChain module — all LangChain objects are built here and
exported as singletons so they are instantiated exactly once.

Exports:
    lc_llm          — ChatOpenAI pointing at OpenRouter
    lc_embeddings   — HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    build_question_chain(cfg)    — LCEL chain for one question type
    build_parallel_chain(cfgs)   — RunnableParallel for all types
"""

from __future__ import annotations

from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from core.config import settings
from core.logger import logger
from models.schemas import QuestionType, QuestionTypeConfig


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert academic question paper setter. "
    "Generate high-quality exam questions strictly from the provided context. "
    "NEVER make up information not present in the context. "
    "Always return ONLY valid JSON — no markdown, no preamble, no explanation."
)


# ── Per-type user prompt templates ─────────────────────────────────────────────

_TYPE_TEMPLATES: dict[QuestionType, str] = {
    QuestionType.MCQ: """Generate {count} multiple-choice questions.
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "option_a": string,
  "option_b": string,
  "option_c": string,
  "option_d": string,
  "correct_answer": "A"|"B"|"C"|"D",
  "explanation": string (one sentence why the answer is correct),
  "marks": {marks}

Context:
{context}
""",
    QuestionType.TRUE_FALSE: """Generate {count} true/false questions.
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "correct_answer": true|false,
  "explanation": string,
  "marks": {marks}

Context:
{context}
""",
    QuestionType.FILL_BLANK: """Generate {count} fill-in-the-blank questions.
Use ___ to indicate the blank. Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string (with ___ for blank),
  "answer": string (the word/phrase that fills the blank),
  "marks": {marks}

Context:
{context}
""",
    QuestionType.SHORT: """Generate {count} short-answer questions (2-4 sentence answers).
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "model_answer": string,
  "marks": {marks}

Context:
{context}
""",
    QuestionType.DESCRIPTIVE: """Generate {count} descriptive/essay-type questions.
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "key_points": [list of 3-5 strings - rubric points an evaluator should look for],
  "marks": {marks}

Context:
{context}
""",
}


# ── LLM singleton ──────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=settings.LLM_MAX_RETRIES,
        timeout=settings.LLM_TIMEOUT,
        default_headers={
            "HTTP-Referer": "https://sawal-ai.app",
            "X-Title": "LearnLocal Question Generation",
        },
    )


lc_llm: ChatOpenAI = _build_llm()


# ── Embeddings singleton ───────────────────────────────────────────────────────

def _build_embeddings() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    emb = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded.")
    return emb


lc_embeddings: HuggingFaceEmbeddings = _build_embeddings()


# ── Chain builders ─────────────────────────────────────────────────────────────

def build_question_chain(cfg: QuestionTypeConfig):
    """
    Returns an LCEL chain:
        ChatPromptTemplate | ChatOpenAI | JsonOutputParser
    Invoking with {"context", "difficulty", "keyword", "language"} returns
    a List[dict] of questions for that type.
    """
    template = _TYPE_TEMPLATES[cfg.type]

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", template),
    ]).partial(
        count=cfg.count,
        marks=cfg.marks,
    )

    parser = JsonOutputParser()
    return prompt | lc_llm | parser


def build_parallel_chain(cfgs: List[QuestionTypeConfig]) -> RunnableParallel:
    """
    Returns a RunnableParallel that runs one chain per question type
    concurrently. Keys are the question type values (e.g. "mcq", "true_false").

    Invoke with:
        {"context": str, "difficulty": str, "keyword_hint": str, "language": str}
    Returns:
        {"mcq": [...], "true_false": [...], ...}
    """
    return RunnableParallel(
        **{cfg.type.value: build_question_chain(cfg) for cfg in cfgs}
    )
