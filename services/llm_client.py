"""
services/llm_client.py
======================
Async OpenRouter client with:
  - Exponential backoff retries
  - Streaming support (optional)
  - Per-question-type prompt templates
  - Structured JSON output parsing with fence stripping
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from core.config import settings
from core.logger import logger
from models.schemas import Difficulty, QuestionType, QuestionTypeConfig


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert academic question paper setter. "
    "Generate high-quality exam questions strictly from the provided context. "
    "NEVER make up information not present in the context. "
    "Always return ONLY valid JSON — no markdown, no preamble, no explanation."
)

_TYPE_TEMPLATES: Dict[QuestionType, str] = {
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
""",

    QuestionType.TRUE_FALSE: """Generate {count} true/false questions.
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "correct_answer": true|false,
  "explanation": string,
  "marks": {marks}
""",

    QuestionType.FILL_BLANK: """Generate {count} fill-in-the-blank questions.
Use ___ to indicate the blank. Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string (with ___ for blank),
  "answer": string (the word/phrase that fills the blank),
  "marks": {marks}
""",

    QuestionType.SHORT: """Generate {count} short-answer questions (2–4 sentence answers).
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "model_answer": string,
  "marks": {marks}
""",

    QuestionType.DESCRIPTIVE: """Generate {count} descriptive/essay-type questions.
Difficulty: {difficulty}. Language: {language}. {keyword_hint}
Return a JSON array where each object has:
  "question": string,
  "key_points": [list of 3–5 strings — rubric points an evaluator should look for],
  "marks": {marks}
""",
}


def _build_prompt(
    cfg: QuestionTypeConfig,
    context: str,
    difficulty: Difficulty,
    keyword: Optional[str],
    language: str,
) -> str:
    keyword_hint = f'Focus specifically on the topic: "{keyword}".' if keyword else ""
    instruction = _TYPE_TEMPLATES[cfg.type].format(
        count=cfg.count,
        difficulty=difficulty.value,
        marks=cfg.marks,
        language=language,
        keyword_hint=keyword_hint,
    )
    return f"{instruction}\n\nContext:\n{context}"


# ── JSON parser ────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> List[Dict[str, Any]]:
    """Strip markdown fences and parse JSON array robustly."""
    # Remove ```json ... ``` or ``` ... ```
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Sometimes model wraps in {"questions": [...]}
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed
        # Try common wrapper keys
        for key in ("questions", "data", "items", "result"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return [parsed]
    except json.JSONDecodeError:
        # Last resort: find first [...] block
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse LLM output as JSON: {clean[:200]}")


# ── Async LLM client ───────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.LLM_TIMEOUT),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def _call_once(self, messages: List[Dict], temperature: float) -> str:
        client = await self._get_client()
        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "response_format": {"type": "text"},
        }
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sawal-ai.app",
            "X-Title": "SawalAI",
        }
        resp = await client.post(
            settings.OPENROUTER_BASE_URL,
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Call LLM with exponential-backoff retries."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                return await self._call_once(messages, temp)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 503):
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited. Retrying in {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    last_exc = e
                else:
                    raise
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                wait = 2 ** attempt
                logger.warning(f"LLM timeout/connect error. Retry in {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
                last_exc = e

        raise last_exc

    async def generate_questions_for_type(
        self,
        cfg: QuestionTypeConfig,
        context: str,
        difficulty: Difficulty,
        keyword: Optional[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        """Generate + parse questions for one question type."""
        prompt = _build_prompt(cfg, context, difficulty, keyword, language)
        raw = await self.generate(prompt)
        return _parse_json(raw)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Global singleton (reuses HTTP connections across requests)
llm_client = LLMClient()
