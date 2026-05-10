"""Quick diagnostic: test OpenRouter LLM call directly."""
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY", "")
model   = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")

print(f"API Key set  : {bool(api_key)} (prefix: {api_key[:12]}...)")
print(f"Model        : {model}")
print("-" * 60)


async def test():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Always return valid JSON only."},
                    {"role": "user",   "content": 'Generate 1 MCQ as JSON array: [{"question":"...","option_a":"...","correct_answer":"A"}]'},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sawal-ai.app",
            },
        )
        print(f"HTTP Status  : {resp.status_code}")
        print(f"Response body:\n{resp.text[:1200]}")

asyncio.run(test())
