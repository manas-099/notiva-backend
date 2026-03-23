# Notiva Backend — AI Meeting Notes API

A real-time AI-powered meeting note taker. Listens to speech, generates structured notes, action items, and summaries using LLMs — all streamed live to the frontend.

---

## Tech Stack

- **FastAPI** — REST API + SSE streaming
- **PostgreSQL** — Meeting storage (hosted on Railway)
- **Sarvam AI** — Speech-to-text (STT)
- **OpenRouter (arcee-ai/trinity)** — LLM for note generation
- **asyncpg** — Async PostgreSQL driver
- **structlog** — Structured logging
- **uvicorn** — ASGI server

---

## Features

- Real-time speech transcription via Sarvam AI WebSocket
- LLM-powered segment-by-segment note generation
- Live notes streamed to frontend via Server-Sent Events (SSE)
- Final meeting summary with action items, key points, and next steps
- PostgreSQL persistence for all meetings and segments
- Automatic retry logic for LLM failures
- Structured JSON logging for production debugging

---

## Project Structure

```
notiva-backend/
├── main.py              # FastAPI app entry point, CORS, lifespan
├── Note_taker.py        # Core note-taking engine (AsyncNoteTaker)
├── Meeting_llm.py       # LLM layer (OpenRouter, prompts, parsing)
├── Meetingdb.py         # PostgreSQL database layer
├── Dependences.py       # Shared state: SSE manager, note registry
├── logging_config.py    # Structlog configuration
├── routes/
│   └── meetings.py      # API route handlers
├── requirements.txt     # Python dependencies
├── Procfile             # Railway/Render start command
└── .env                 # Environment variables (not committed)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/meetings/start` | Start a new meeting session |
| POST | `/meetings/{id}/end` | End meeting and get final notes |
| GET | `/meetings/{id}/stream` | SSE stream for live notes |
| GET | `/` | Health check |
| GET | `/docs` | Swagger UI |

---

## Setup — Local Development

### 1. Clone the repo

```bash
git clone https://github.com/manas-099/notiva-backend
cd notiva-backend
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
uv add  -r requirements.txt
```

### 4. Create `.env` file

```env
DATABASE_URL=postgresql://user:password@host:port/dbname
SARVAM_API_KEY=your_sarvam_key
OPENROUTER_API_KEY=your_openrouter_key
LOG_FORMAT=console
LOG_LEVEL=INFO
```

### 5. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will be live at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

<!-- ## Deployment — Railway

### Environment Variables (set in Railway dashboard)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `SARVAM_API_KEY` | Sarvam AI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `LOG_FORMAT` | `json` for production |
| `LOG_LEVEL` | `INFO` | -->

### 6. Run this on terminal itself using STT_runner.py
```
cd Notiva

python -m STT_runner 
```
### Start Command


```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## How It Works

1. Frontend calls `POST /meetings/start` with user ID and API keys
2. Backend creates a meeting in PostgreSQL and returns a WebSocket URL + SSE URL
3. Frontend streams audio to Sarvam AI via WebSocket for real-time transcription
4. Transcribed text is fed into `AsyncNoteTaker` word-by-word
5. Every ~200 words or 4 seconds of silence, a segment is flushed to the LLM
6. LLM returns structured notes, action items, key points as JSON
7. Notes are pushed to the frontend via SSE in real time
8. When the meeting ends, `POST /meetings/{id}/end` triggers a final LLM call
9. Final summary, action items, and next steps are returned and saved to DB

---

## LLM Note Structure

Each segment produces:

```json
{
  "rolling_summary": "Full meeting summary so far",
  "notes": ["bullet point 1", "bullet point 2"],
  "action_items": [
    { "task": "...", "owner": "name", "deadline": "timeframe" }
  ],
  "key_points": [
    { "point": "important detail", "category": "number|date|risk|blocker" }
  ],
  "display_notes": {
    "headline": "One sentence about this segment",
    "summary_line": "1-2 sentence expansion",
    "notes": ["..."],
    "highlights": ["BLOCKER: ...", "NUMBER: ..."]
  }
}
```

---

## Environment Notes

- Python 3.12+ required
- All API keys are passed dynamically from the frontend — no hardcoded keys
- CORS is open (`*`) for Electron app compatibility
- DB migrations run automatically on startup

---

## Author

**Manas Patra** — [manaspatra481@gmail.com](mailto:manaspatra481@gmail.com)