"""
services/embedder.py
====================
Thin wrapper around the LangChain HuggingFaceEmbeddings singleton
from lc_chain.py.

Keeps the same public API as before (encode / warmup / cosine_similarity)
so pipeline.py and the rest of the codebase need no changes.
"""

from __future__ import annotations

import numpy as np

from core.logger import logger
from services.lc_chain import lc_embeddings


class EmbedderService:
    """Wraps LangChain HuggingFaceEmbeddings with the legacy encode() API."""

    def warmup(self):
        """Pre-warm the model with a dummy sentence to avoid cold-start latency."""
        try:
            lc_embeddings.embed_query("warmup")
            logger.info("Embedding model warmed up.")
        except Exception as exc:
            logger.warning(f"Embedding warmup failed (non-fatal): {exc}")

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """
        Batch-encode texts.
        Returns (N, D) float32 numpy array (unit-normalised),
        or None if the model is unavailable.
        """
        try:
            vecs = lc_embeddings.embed_documents(texts)   # List[List[float]]
            return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            logger.error(f"Embedding encode failed: {exc}")
            return None

    def embed_query(self, text: str) -> np.ndarray | None:
        """Encode a single query string → (D,) float32 array."""
        try:
            vec = lc_embeddings.embed_query(text)
            return np.array(vec, dtype=np.float32)
        except Exception as exc:
            logger.error(f"Query embedding failed: {exc}")
            return None

    def cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        query_vec: (D,)   doc_vecs: (N, D)
        Returns (N,) similarity scores (dot product == cosine because unit-normed).
        """
        return doc_vecs @ query_vec


# Global singleton
embedder = EmbedderService()
