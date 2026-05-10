"""
services/llm_client.py
======================
LangChain-based LLM client using:
  - ChatOpenAI          → OpenRouter as backend
  - ChatPromptTemplate  → per-question-type prompts
  - JsonOutputParser    → strips markdown fences, parses JSON automatically
  - LCEL pipes (|)      → clean chain composition

The legacy `LLMClient` class is preserved as a thin wrapper over
`build_question_chain()` from lc_chain.py so routers don't need to change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.logger import logger
from models.schemas import Difficulty, QuestionTypeConfig
from services.lc_chain import build_question_chain


class LLMClient:
    """
    Thin wrapper that keeps the old `generate_questions_for_type()` API
    while delegating to LangChain LCEL chains internally.
    """

    async def generate_questions_for_type(
        self,
        cfg: QuestionTypeConfig,
        context: str,
        difficulty: Difficulty,
        keyword: Optional[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate and parse questions for one question type via LangChain chain.

        Returns a List[dict] of question objects.
        Raises on LLM or parse failure (caller handles exceptions).
        """
        chain = build_question_chain(cfg)

        keyword_hint = (
            f'Focus specifically on the topic: "{keyword}".' if keyword else ""
        )

        logger.info(
            f"[LLM] Invoking chain for {cfg.count}x {cfg.type.value} "
            f"| difficulty={difficulty.value} | keyword={keyword!r}"
        )

        result = await chain.ainvoke({
            "context":      context,
            "difficulty":   difficulty.value,
            "keyword_hint": keyword_hint,
            "language":     language,
        })

        # JsonOutputParser returns a list or dict; normalise to list
        if isinstance(result, list):
            questions = result
        elif isinstance(result, dict):
            # Unwrap common wrapper keys
            for key in ("questions", "data", "items", "result"):
                if key in result and isinstance(result[key], list):
                    questions = result[key]
                    break
            else:
                questions = [result]
        else:
            raise ValueError(
                f"Unexpected LLM output type {type(result)}: {str(result)[:200]}"
            )

        logger.info(
            f"[LLM] Parsed {len(questions)} questions for {cfg.type.value}"
        )
        return questions

    async def close(self):
        """No-op — LangChain manages HTTP connections internally."""
        pass


# Global singleton (reuses HTTP connections across requests via LangChain)
llm_client = LLMClient()
