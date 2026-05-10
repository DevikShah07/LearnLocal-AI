"""
services/faiss_store.py
=======================
Per-document FAISS vector store using langchain_community.vectorstores.FAISS.

Each uploaded document gets its own FAISS index persisted under:
    embeddings/<doc_id>/          ← LangChain save_local() format
    embeddings/<doc_id>/meta.json ← our custom metadata

Public API (unchanged):
    faiss_store.save(doc_name, chunk_texts, embeddings, pdf_size_bytes) → doc_id
    faiss_store.search(doc_id, query_embedding, k) → List[ChunkResult]
    faiss_store.mmr_search(doc_id, query, k, fetch_k) → List[ChunkResult]
    faiss_store.get_meta(doc_id) → Optional[DocumentMeta]
    faiss_store.list_documents() → List[DocumentMeta]
    faiss_store.delete(doc_id) → bool
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.logger import logger
from services.lc_chain import lc_embeddings

# Root folder where all FAISS indexes are persisted
_BASE_DIR = Path(__file__).resolve().parent.parent / "embeddings"


@dataclass
class ChunkResult:
    chunk_index: int
    text: str
    score: float           # relevance score (0-1)


@dataclass
class DocumentMeta:
    doc_id: str
    name: str
    total_chunks: int
    dimension: int
    created_at: float      # unix timestamp
    size_bytes: int        # size of the PDF that was indexed


class FAISSStore:
    """
    LangChain-backed per-document FAISS store.

    Uses langchain_community.vectorstores.FAISS which provides:
      - save_local() / load_local()       — simple persistence
      - similarity_search_with_relevance_scores() — 0-1 cosine scores
      - max_marginal_relevance_search()   — diverse chunk selection (MMR)
    """

    def __init__(self, base_dir: Path = _BASE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._check_available()

    def _check_available(self):
        try:
            from langchain_community.vectorstores import FAISS  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("langchain_community FAISS not available. Install langchain-community.")
            self._available = False

    def _doc_dir(self, doc_id: str) -> Path:
        return self.base_dir / doc_id

    def _meta_path(self, doc_id: str) -> Path:
        return self._doc_dir(doc_id) / "meta.json"

    @property
    def available(self) -> bool:
        return self._available

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        doc_name: str,
        chunk_texts: List[str],
        embeddings: np.ndarray,
        pdf_size_bytes: int = 0,
    ) -> str:
        """
        Build and persist a LangChain FAISS index for a document.
        Returns doc_id (UUID) for future lookups.
        """
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        doc_id   = str(uuid.uuid4())
        doc_path = self._doc_dir(doc_id)
        doc_path.mkdir(parents=True, exist_ok=True)

        # Convert (text, embedding) pairs into LangChain Documents
        text_embedding_pairs = list(zip(chunk_texts, embeddings.tolist()))

        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=lc_embeddings,
            metadatas=[{"chunk_index": i} for i in range(len(chunk_texts))],
        )

        # Persist index (creates index.faiss + index.pkl inside doc_path)
        vectorstore.save_local(str(doc_path))

        # Determine embedding dimension
        dimension = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])

        # Save our metadata separately
        meta = DocumentMeta(
            doc_id=doc_id,
            name=doc_name,
            total_chunks=len(chunk_texts),
            dimension=dimension,
            created_at=time.time(),
            size_bytes=pdf_size_bytes,
        )
        self._meta_path(doc_id).write_text(
            json.dumps(asdict(meta), indent=2), encoding="utf-8"
        )

        logger.info(
            f"[FAISS] Saved index for '{doc_name}' "
            f"({len(chunk_texts)} chunks, dim={dimension}) → doc_id={doc_id}"
        )
        return doc_id

    # ── Search (cosine similarity) ────────────────────────────────────────────

    def search(
        self,
        doc_id: str,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[ChunkResult]:
        """
        Similarity search using a pre-computed query embedding.
        Returns up to k results sorted by score descending.
        """
        from langchain_community.vectorstores import FAISS

        doc_path = self._doc_dir(doc_id)
        if not doc_path.exists():
            raise FileNotFoundError(f"No FAISS index found for doc_id={doc_id}")

        vectorstore = FAISS.load_local(
            str(doc_path),
            lc_embeddings,
            allow_dangerous_deserialization=True,
        )

        # Use embedding_function.similarity_search_by_vector for pre-computed vectors
        raw = vectorstore.similarity_search_by_vector_with_relevance_scores(
            query_embedding.tolist(), k=k
        )

        results = []
        for doc, score in raw:
            chunk_index = doc.metadata.get("chunk_index", -1)
            results.append(ChunkResult(
                chunk_index=int(chunk_index),
                text=doc.page_content,
                score=round(float(score), 4),
            ))
        return results

    # ── MMR Search (diverse chunk selection) ──────────────────────────────────

    def mmr_search(
        self,
        doc_id: str,
        query: str,
        k: int = 10,
        fetch_k: int = 30,
    ) -> List[ChunkResult]:
        """
        Maximal Marginal Relevance search — returns diverse top-k chunks
        relevant to query. Replaces the custom 4-signal chunk_scorer.

        Args:
            doc_id:  Document to search.
            query:   Natural language query / keyword.
            k:       Number of chunks to return.
            fetch_k: Candidate pool size before MMR re-ranking.
        """
        from langchain_community.vectorstores import FAISS

        doc_path = self._doc_dir(doc_id)
        if not doc_path.exists():
            raise FileNotFoundError(f"No FAISS index found for doc_id={doc_id}")

        vectorstore = FAISS.load_local(
            str(doc_path),
            lc_embeddings,
            allow_dangerous_deserialization=True,
        )

        docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
        )

        results = []
        for i, doc in enumerate(docs):
            chunk_index = doc.metadata.get("chunk_index", i)
            results.append(ChunkResult(
                chunk_index=int(chunk_index),
                text=doc.page_content,
                score=1.0,            # MMR doesn't return raw scores
            ))
        return results

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_meta(self, doc_id: str) -> Optional[DocumentMeta]:
        path = self._meta_path(doc_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return DocumentMeta(**data)

    def list_documents(self) -> List[DocumentMeta]:
        docs = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir():
                meta = self.get_meta(d.name)
                if meta:
                    docs.append(meta)
        docs.sort(key=lambda m: m.created_at, reverse=True)
        return docs

    def delete(self, doc_id: str) -> bool:
        doc_path = self._doc_dir(doc_id)
        if doc_path.exists():
            shutil.rmtree(doc_path)
            logger.info(f"[FAISS] Deleted index doc_id={doc_id}")
            return True
        return False


# Global singleton
faiss_store = FAISSStore()
