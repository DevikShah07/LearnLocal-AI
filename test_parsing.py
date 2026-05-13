import asyncio
from core.config import settings
from models.schemas import QuestionTypeConfig, QuestionType, Difficulty
from services.llm_client import llm_client

async def test():
    print(f"Testing against LOCAL_LLM_BASE_URL: {settings.LOCAL_LLM_BASE_URL}")
    print(f"Using Model: {settings.LLM_MODEL}")
    
    cfg = QuestionTypeConfig(type=QuestionType.MCQ, count=1, marks=1)
    context = "The Earth revolves around the Sun. This was proposed by Copernicus."
    
    print("\n--- Generating Questions ---")
    try:
        questions = await llm_client.generate_questions_for_type(
            cfg=cfg,
            context=context,
            difficulty=Difficulty.EASY,
            keyword=None,
            language="English"
        )
        print(f"\nFinal Extracted Questions: {questions}")
    except Exception as e:
        print(f"Failed with exception: {e}")

if __name__ == "__main__":
    asyncio.run(test())
