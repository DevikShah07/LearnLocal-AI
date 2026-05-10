# LearnLocal Question Generation — Question Generation API

Production-ready FastAPI service: PDF upload → smart chunk selection → parallel LLM calls → structured JSON questions.

---

## Project Structure

```
LeanLocal-AI/
├── main.py                  # FastAPI app, middleware, lifespan
├── requirements.txt
├── .env.example             # copy to .env and fill in your key
├── test_ui.html             # open in browser to test visually
│
├── core/
│   ├── config.py            # all settings via env vars
│   └── logger.py
│
├── models/
│   └── schemas.py           # all Pydantic request/response models
│
├── routers/
│   ├── health.py            # GET /health
│   └── generate.py          # POST /api/v1/generate
│
└── services/
    ├── pdf_extractor.py     # PyPDF2 → PageChunk list
    ├── embedder.py          # sentence-transformers singleton
    ├── chunk_scorer.py      # 4-signal importance scorer
    ├── llm_client.py        # async OpenRouter client + prompt templates
    └── pipeline.py          # orchestrates everything in parallel
```

---

## Setup

### 1. Clone / copy the project

```bash
cd LearnLocal-AI
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~80MB) on first run. After that it's cached locally — no API cost.

### 4. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

Get your free OpenRouter key at: https://openrouter.ai/keys

### 5. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server starts at: http://localhost:8000

---

## API Endpoints

### `GET /health`
Check server status, embedding model, and LLM model.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "openai/gpt-oss-120b:free"
}
```

---

### `POST /api/v1/generate`
Upload a PDF and generate questions.

**Form data:**
- `file` — PDF file (required)
- `config` — JSON string (optional, all fields have defaults)

**Config schema:**
```json
{
  "question_types": [
    {"type": "mcq",          "count": 5, "marks": 1},
    {"type": "true_false",   "count": 3, "marks": 1},
    {"type": "fill_blank",   "count": 3, "marks": 1},
    {"type": "short_answer", "count": 2, "marks": 3},
    {"type": "descriptive",  "count": 1, "marks": 5}
  ],
  "difficulty": "medium",
  "keyword": "neural networks",
  "page_range": "1-20",
  "top_k": 10,
  "language": "English"
}
```

**Supported question types:**
| type | description |
|------|-------------|
| `mcq` | 4-option MCQ with explanation |
| `true_false` | True/False with explanation |
| `fill_blank` | Fill in the blank |
| `short_answer` | 2–4 sentence answer |
| `descriptive` | Essay with rubric key points |

**Difficulty values:** `easy` · `medium` · `hard`

**Query params:**
- `?debug=true` — includes chunk scores in response (for inspection)

---

### `POST /api/v1/generate/text`
Same as above but accepts raw text instead of a PDF file.

**Form data:**
- `text` — raw document text (required)
- `config` — same JSON config (optional)

---

## Response Format

```json
{
  "status": "success",
  "document_name": "lecture_notes.pdf",
  "total_pages": 45,
  "chunks_selected": 10,
  "context_chars": 18420,
  "difficulty": "medium",
  "keyword": "neural networks",
  "language": "English",

  "questions": [
    {
      "type": "mcq",
      "question": "Which neural network type is best suited for image recognition?",
      "option_a": "RNN",
      "option_b": "CNN",
      "option_c": "LSTM",
      "option_d": "Transformer",
      "correct_answer": "B",
      "explanation": "CNNs use convolutional filters that detect spatial patterns in images.",
      "marks": 1
    },
    {
      "type": "true_false",
      "question": "LSTM networks are designed to handle the vanishing gradient problem.",
      "correct_answer": true,
      "explanation": "LSTM gates control information flow, preventing gradient vanishing.",
      "marks": 1
    },
    {
      "type": "fill_blank",
      "question": "___ is the optimization algorithm that updates weights using gradients.",
      "answer": "Gradient Descent",
      "marks": 1
    },
    {
      "type": "short_answer",
      "question": "What is backpropagation and why is it important?",
      "model_answer": "Backpropagation computes gradients of the loss with respect to each weight using the chain rule, enabling the network to learn by adjusting weights to minimize error.",
      "marks": 3
    },
    {
      "type": "descriptive",
      "question": "Explain the architecture and working of a Transformer model.",
      "key_points": [
        "Self-attention mechanism allows each token to attend to all others",
        "Multi-head attention runs attention in parallel across different subspaces",
        "Positional encoding adds sequence order information",
        "Feed-forward layers process each position independently",
        "Encoder-decoder structure used in seq2seq tasks"
      ],
      "marks": 5
    }
  ],

  "grouped": [
    { "type": "mcq", "count": 5, "total_marks": 5, "questions": [...] },
    { "type": "true_false", "count": 3, "total_marks": 3, "questions": [...] }
  ],

  "total_questions": 14,
  "total_marks": 25,
  "llm_model": "openai/gpt-oss-120b:free",
  "processing_time_ms": 4820.3,

  "chunk_scores": null
}
```

---

## cURL Examples

### Basic — 5 MCQ from full PDF
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -F "file=@your_document.pdf"
```

### Full config — all question types
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -F "file=@notes.pdf" \
  -F 'config={
    "question_types": [
      {"type":"mcq","count":5,"marks":1},
      {"type":"true_false","count":3,"marks":1},
      {"type":"fill_blank","count":3,"marks":1},
      {"type":"short_answer","count":2,"marks":3},
      {"type":"descriptive","count":1,"marks":5}
    ],
    "difficulty":"hard",
    "keyword":"backpropagation",
    "page_range":"5-25",
    "top_k":10,
    "language":"English"
  }'
```

### With debug chunk scores
```bash
curl -X POST "http://localhost:8000/api/v1/generate?debug=true" \
  -F "file=@notes.pdf"
```

### Python requests
```python
import requests, json

with open("notes.pdf", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/v1/generate",
        files={"file": f},
        data={"config": json.dumps({
            "question_types": [
                {"type": "mcq", "count": 5, "marks": 1},
                {"type": "short_answer", "count": 3, "marks": 3},
            ],
            "difficulty": "medium",
            "keyword": "neural networks",
        })}
    )

data = r.json()
for q in data["questions"]:
    print(q["question"])
```

---

## Test UI

Open `test_ui.html` directly in your browser — no server needed for the UI itself.

1. Set the API URL to `http://localhost:8000`
2. Click **Ping** to verify connection
3. Drop a PDF
4. Configure question types, difficulty, keyword
5. Click **⚡ Generate Questions**

The UI shows:
- All questions tab with rich rendering per type
- Per-type tabs with counts
- Raw JSON tab for copy/download
- Chunk Scores tab (enable debug mode)

---

## Performance Notes

- **Parallel LLM calls** — all question types are generated simultaneously with `asyncio.gather`, not sequentially
- **Embedding warmup** — model is pre-loaded at startup, first request has no cold start
- **HTTP connection pool** — `httpx.AsyncClient` is reused across requests (keep-alive)
- **Context window guard** — chunks are trimmed to `MAX_CONTEXT_TOKENS * 4` chars before sending to LLM
- **GZip middleware** — responses >1KB are compressed automatically
- **Retry with backoff** — rate limit (429) and timeout errors retry up to 3 times

---

## Integrating with your Django Project

The pipeline service is framework-agnostic. In your Django `views.py`:

```python
import asyncio
from services.pdf_extractor import extract_pdf
from services.pipeline import run_pipeline
from models.schemas import GenerateRequest, QuestionTypeConfig, QuestionType, Difficulty

def generate_questions_view(request):
    file_bytes = request.FILES['pdf'].read()
    doc = extract_pdf(file_bytes, request.FILES['pdf'].name)

    req = GenerateRequest(
        question_types=[QuestionTypeConfig(type=QuestionType.MCQ, count=5, marks=1)],
        difficulty=Difficulty.MEDIUM,
        keyword=request.POST.get('keyword'),
    )

    result = asyncio.run(run_pipeline(
        doc=doc,
        question_types=req.question_types,
        difficulty=req.difficulty,
        keyword=req.keyword,
        page_range=None,
        top_k=10,
        language="English",
    ))

    return JsonResponse(result.model_dump())
```
