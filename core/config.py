"""core/config.py — centralised settings via environment variables."""

import os
import warnings
from typing import List

from dotenv import load_dotenv

# Load .env from project root (works from any working directory)
load_dotenv()


class Settings:
    # LLM - OpenRouter
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # LLM - Local (ngrok/localai/ollama)
    LOCAL_LLM_BASE_URL: str = os.getenv("LOCAL_LLM_BASE_URL", "")
    LOCAL_LLM_API_KEY: str = os.getenv("LOCAL_LLM_API_KEY", "no-key")  # Default for local models

    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "3000"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))          # seconds
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Chunk selection
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))  # ~24K chars

    # Text splitting (LangChain SemanticChunker)
    SEMANTIC_BREAKPOINT_TYPE: str = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")
    SEMANTIC_BREAKPOINT_THRESHOLD: float = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "95"))

    # Legacy splitting (kept for reference or fallback)
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # PDF
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "20"))

    # CORS
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        if not self.OPENROUTER_API_KEY and not self.LOCAL_LLM_BASE_URL:
            warnings.warn(
                "Neither OPENROUTER_API_KEY nor LOCAL_LLM_BASE_URL is set! LLM calls will fail.",
                stacklevel=2,
            )


settings = Settings()
