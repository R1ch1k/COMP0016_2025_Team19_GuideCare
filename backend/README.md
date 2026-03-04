# GuideCare Backend

FastAPI backend with LangGraph orchestration, NICE guideline graph traversal engine, and real-time WebSocket chat.

## Stack

- **FastAPI** — async REST + WebSocket
- **PostgreSQL** — patient, conversation, and diagnosis storage (SQLAlchemy + asyncpg)
- **LangGraph** — state machine orchestration for the clinical pipeline
- **OpenAI API** — triage, variable extraction, clarification questions, guideline selection
- **Guideline Engine** — pure-Python BFS traversal of NICE decision trees (no LLM needed)

## Setup

### With Docker (recommended)

```bash
# From the repo root
cp backend/.env.example backend/.env
# Edit backend/.env — set OPENAI_API_KEY

docker-compose up --build
```

Backend runs on http://localhost:8000. API docs at http://localhost:8000/docs.

### Without Docker

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set DATABASE_URL and OPENAI_API_KEY

cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Requires a running PostgreSQL instance.

## Pipeline

The LangGraph state machine runs this flow for each user message:

```
load_patient → triage → clarify → select_guideline → extract_variables → walk_graph → format_output
```

1. **load_patient** — fetch patient record from DB
2. **triage** — LLM-based urgency assessment (emergency/urgent/moderate/routine) + guideline selection
3. **clarify** — if variables are missing, LLM generates targeted clarification questions
4. **select_guideline** — keyword mapping to one of 10 NICE guidelines, LLM fallback
5. **extract_variables** — LLM extracts clinical variables from conversation + regex post-processing
6. **walk_graph** — BFS traversal of the guideline decision tree using extracted variables
7. **format_output** — template-based recommendation from action nodes

Emergency cases (urgent_escalation=True) skip directly from triage to format_output with an immediate referral message.

Completed diagnoses are automatically persisted to the `diagnoses` table.

## API Reference

### REST

| Method | Path | Description |
|--------|------|-------------|
| GET | /patients | List patients |
| GET | /patients/{id} | Patient details |
| GET | /patients/{id}/context | Patient context for LLM |
| POST | /patients | Create patient |
| POST | /patients/import | Import patients from CSV/Excel |
| POST | /conversations | Start conversation |
| GET | /conversations/{id} | Conversation history |
| GET | /diagnoses | List all diagnoses |
| GET | /diagnoses/{id} | Single diagnosis detail |
| GET | /diagnoses/export?format=json\|csv | Export diagnoses |

### WebSocket

`/ws/chat/{patient_id}` — send/receive JSON messages:

```json
{"role": "user", "content": "patient has a sore throat, 38.5C fever", "meta": {}}
```

Response events stream back as the pipeline progresses (triage, clarification, recommendation).

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | — | `postgresql+asyncpg://user:pass@host:5432/db` |
| OPENAI_API_KEY | Yes | — | OpenAI API key |
| OPENAI_MODEL | No | gpt-4o | Model for LLM calls |
| CORS_ORIGINS | No | * | Allowed origins (comma-separated) |

## Testing

```bash
pip install -r requirements-dev.txt

# With Docker-based PostgreSQL (via Testcontainers)
cd src && PYTHONPATH=. pytest -q ../tests

# Or with an existing PostgreSQL instance
export TEST_DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/guidecare_test"
cd src && PYTHONPATH=. pytest -q ../tests
```
