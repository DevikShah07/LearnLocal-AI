"""routers/health.py"""
from fastapi import APIRouter
from models.schemas import HealthResponse
from core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL,
    )

@router.get("/")
async def root():
    return {"message": "LearnLocal Question Generation API — visit /docs for the interactive UI"}
