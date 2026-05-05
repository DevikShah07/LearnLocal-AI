"""
services/embedder.py — singleton embedding model with lazy load + warmup.

Uses sentence-transformers/all-MiniLM-L6-v2 locally (no API cost).
Falls back gracefully to TF-IDF scoring if the model isn't installed.
"""

from __future__ import annotations

import numpy as np
from core.config import settings
from core.logger import logger


class EmbedderService:
    """Wraps SentenceTransformer with lazy loading and a warmup call."""

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to TF-IDF only scoring."
            )
            self._model = None

    def warmup(self):
        """Pre-load model at startup to avoid cold-start latency on first request."""
        self._load()
        if self._model:
            self._model.encode(["warmup"], show_progress_bar=False)

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """
        Returns (N, D) float32 numpy array or None if model unavailable.
        Batch encode for speed.
        """
        self._load()
        if self._model is None:
            return None
        return self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine sim = dot product
        )

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        query_vec: (D,)  doc_vecs: (N, D)
        Returns (N,) similarity scores, already normalised so dot product == cosine.
        """
        return doc_vecs @ query_vec


# Global singleton
embedder = EmbedderService()
