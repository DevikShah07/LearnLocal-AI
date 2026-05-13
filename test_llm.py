"""Quick diagnostic: test LLM call (OpenRouter or Local) directly."""
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

# Config
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
local_base_url = os.getenv("LOCAL_LLM_BASE_URL", "")
local_api_key  = os.getenv("LOCAL_LLM_API_KEY", "no-key")
model          = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")

if local_base_url:
    print(f"Mode         : LOCAL LLM")
    print(f"Base URL     : {local_base_url}")
    print(f"API Key      : {local_api_key[:6]}...")
    target_url   = f"{local_base_url}/generate"
    auth_header  = f"Bearer {local_api_key}"
    # Local API expects "prompt"
    payload = {
        "model": model,
        "prompt": "Generate 1 MCQ about Science as a JSON array: [{\"question\":\"...\",\"option_a\":\"...\",\"correct_answer\":\"A\"}]"
    }
else:
    print(f"Mode         : OPENROUTER")
    print(f"API Key set  : {bool(openrouter_key)} (prefix: {openrouter_key[:12]}...)")
    target_url   = "https://openrouter.ai/api/v1/chat/completions"
    auth_header  = f"Bearer {openrouter_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Always return valid JSON only."},
            {"role": "user",   "content": 'Generate 1 MCQ as JSON array: [{"question":"...","option_a":"...","correct_answer":"A"}]'},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }

print(f"Model        : {model}")
print("-" * 60)


async def test():
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                target_url,
                json=payload,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://sawal-ai.app",
                },
            )
            print(f"HTTP Status  : {resp.status_code}")
            print(f"Response body:\n{resp.text[:1200]}")
        except Exception as e:
            print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
