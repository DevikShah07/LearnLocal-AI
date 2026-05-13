"""
services/llm_client.py
======================
LangChain-based LLM client with robust JSON extraction for small local models.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from core.logger import logger
from models.schemas import Difficulty, QuestionTypeConfig
from services.lc_chain import build_question_chain


class LLMClient:
    """
    Handles LLM interaction via LangChain and provides robust JSON parsing
    to handle "noisy" outputs from small local models (like Gemma 2b).
    """

    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Tries to find and parse a JSON array or object within a string.
        Handles markdown fences (```json ... ```) and extra conversational text.
        """
        if not text or not text.strip():
            logger.warning("[LLM] Received empty text from model.")
            return []

        # Clean up text - sometimes models add weird non-printable chars
        text = text.strip()

        # 1. Try to find content inside markdown JSON blocks
        # We look for all blocks and join them if necessary, or just take the biggest one
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if code_blocks:
            # Try parsing each code block until one works
            for block in code_blocks:
                res = self._parse_single_json_string(block)
                if res:
                    return res

        # 2. If no code blocks or they failed, try finding boundaries in the whole text
        return self._parse_single_json_string(text)

    def _parse_single_json_string(self, text: str) -> List[Dict[str, Any]]:
        """Helper to find boundaries and parse a single chunk of text as JSON."""
        candidate = text.strip()
        
        # Helper function to evaluate parsed data
        def _evaluate_data(data: Any) -> List[Dict[str, Any]]:
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("questions", "data", "items", "result"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            return []

        # Attempt 1: As-is
        try:
            return _evaluate_data(json.loads(candidate))
        except json.JSONDecodeError:
            pass

        # Attempt 2: Fix truncated JSON object (missing '}')
        try:
            return _evaluate_data(json.loads(candidate + "}"))
        except json.JSONDecodeError:
            pass

        # Attempt 3: Fix truncated JSON array with object (missing '}]')
        try:
            return _evaluate_data(json.loads(candidate + "}]"))
        except json.JSONDecodeError:
            pass

        # Attempt 4: Fix truncated JSON array (missing ']')
        try:
            return _evaluate_data(json.loads(candidate + "]"))
        except json.JSONDecodeError:
            pass

        # Attempt 5: Find bounded object if there is extra text at the end
        obj_start = candidate.find("{")
        obj_end = candidate.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_start < obj_end:
            try:
                return _evaluate_data(json.loads(candidate[obj_start : obj_end + 1]))
            except json.JSONDecodeError:
                pass

        logger.debug(f"[LLM] All JSON parse attempts failed.")
        return []

    async def generate_questions_for_type(
        self,
        cfg: QuestionTypeConfig,
        context: str,
        difficulty: Difficulty,
        keyword: Optional[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for one type. Now returns raw string from chain
        and parses it using robust _extract_json_from_text.
        """
        chain = build_question_chain(cfg)

        keyword_hint = (
            f'Focus specifically on the topic: "{keyword}".' if keyword else ""
        )

        logger.info(
            f"[LLM] Invoking local chain for {cfg.type.value} "
            f"| difficulty={difficulty.value}"
        )

        try:
            # Chain now returns raw string because we use StrOutputParser
            raw_text = await chain.ainvoke({
                "context":      context,
                "difficulty":   difficulty.value,
                "keyword_hint": keyword_hint,
                "language":     language,
            })
            
            questions = self._extract_json_from_text(raw_text)
            
            if not questions:
                logger.warning(f"[LLM] No questions parsed from response. FULL TEXT:\n{raw_text}")
            else:
                logger.info(f"[LLM] Successfully parsed {len(questions)} questions.")
                
            return questions

        except Exception as exc:
            logger.error(f"[LLM] Chain invocation failed: {exc}")
            raise

    async def close(self):
        pass


# Global singleton
llm_client = LLMClient()
