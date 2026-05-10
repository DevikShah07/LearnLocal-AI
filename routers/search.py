"""
routers/search.py
=================
GET  /api/v1/docs               — list all indexed documents
GET  /api/v1/docs/{doc_id}      — get metadata for one document
POST /api/v1/docs/{doc_id}/search — semantic search within a document
DELETE /api/v1/docs/{doc_id}    — remove an index from disk

Now uses LangChain FAISS .similarity_search_with_relevance_scores()
via faiss_store.search() — response format is unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.logger import logger
from models.schemas import (
    ChunkSearchResult,
    DocumentInfo,
    SearchRequest,
    SearchResponse,
)
from services.faiss_store import faiss_store
from services.lc_chain import lc_embeddings

router = APIRouter()


def _check_available():
    if not faiss_store.available:
        raise HTTPException(
            status_code=503,
            detail="FAISS is not available. Install langchain-community to enable vector search.",
        )


# ── List all indexed documents ─────────────────────────────────────────────────

@router.get(
    "/docs",
    response_model=list[DocumentInfo],
    summary="List all indexed documents",
)
def list_documents():
    _check_available()
    docs = faiss_store.list_documents()
    return [
        DocumentInfo(
            doc_id=d.doc_id,
            name=d.name,
            total_chunks=d.total_chunks,
            dimension=d.dimension,
            created_at=d.created_at,
            size_bytes=d.size_bytes,
        )
        for d in docs
    ]


# ── Get single document metadata ───────────────────────────────────────────────

@router.get(
    "/docs/{doc_id}",
    response_model=DocumentInfo,
    summary="Get metadata for an indexed document",
)
def get_document(doc_id: str):
    _check_available()
    meta = faiss_store.get_meta(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(
        doc_id=meta.doc_id,
        name=meta.name,
        total_chunks=meta.total_chunks,
        dimension=meta.dimension,
        created_at=meta.created_at,
        size_bytes=meta.size_bytes,
    )


# ── Semantic search within a document ─────────────────────────────────────────

@router.post(
    "/docs/{doc_id}/search",
    response_model=SearchResponse,
    summary="Semantic similarity search within an indexed document",
)
async def search_document(doc_id: str, body: SearchRequest):
    _check_available()

    meta = faiss_store.get_meta(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    # Embed the query using the LangChain embeddings singleton
    try:
        query_vec_list = lc_embeddings.embed_query(body.query)
        import numpy as np
        query_vec = np.array(query_vec_list, dtype=np.float32)
    except Exception as exc:
        logger.error(f"Query embedding failed: {exc}", exc_info=True)
        raise HTTPException(status_code=503, detail="Embedding model unavailable.")

    try:
        raw = faiss_store.search(doc_id, query_vec, k=body.k)
    except Exception as exc:
        logger.error(f"FAISS search error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        ChunkSearchResult(chunk_index=r.chunk_index, text=r.text, score=r.score)
        for r in raw
    ]
    return SearchResponse(doc_id=doc_id, query=body.query, results=results)


# ── Delete a document index ────────────────────────────────────────────────────

@router.delete(
    "/docs/{doc_id}",
    summary="Delete an indexed document",
)
def delete_document(doc_id: str):
    _check_available()
    deleted = faiss_store.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "deleted", "doc_id": doc_id}
