# GuideCare — NHS NICE Guideline Clinical Decision Support

A full-stack clinical decision support system that traverses NICE (National Institute for Health and Care Excellence) guideline decision trees to provide evidence-based recommendations. Combines a real graph-traversal engine with LLM-powered triage, variable extraction, and clarification.

## Architecture

```
Frontend (Next.js)  <--WebSocket-->  Backend (FastAPI + LangGraph)
                                         |
                                    Guideline Engine
                                    (graph traversal)
                                         |
                                    LLM (OpenAI GPT-4o API)
```

**Frontend** — Next.js chat UI with WebSocket real-time messaging, patient selector, data import modal, and guideline-based recommendations display.

**Backend** — FastAPI with async PostgreSQL (SQLAlchemy + asyncpg), LangGraph state machine orchestration, and WebSocket endpoint for per-patient conversations.

**Guideline Engine** — Pure-Python BFS traversal of NICE guideline decision trees. Evaluates condition nodes (numeric comparisons, BP ranges, AND/OR logic) and returns reached action nodes with the full decision path.

**LLM Layer** — Used for triage (urgency assessment + guideline selection), variable extraction from conversation, and clarification question generation.

## NICE Guidelines Covered

| ID | Guideline | Topic |
|----|-----------|-------|
| NG84 | Sore throat (acute) | Antibiotic prescribing |
| NG91 | Otitis media (acute) | Ear infection management |
| NG112 | UTI (lower) | Urinary tract infection |
| NG133 | Hypertension in pregnancy | Pre-eclampsia screening |
| NG136 | Hypertension in adults | BP diagnosis and management |
| NG184 | Bite wounds | Animal/human bite management |
| NG222 | Depression in adults | Treatment pathways |
| NG232 | Head injury | Assessment and early management |
| NG81 (Glaucoma) | Chronic open-angle glaucoma | IOP-based treatment |
| NG81 (Hypertension) | Ocular hypertension | Risk-based treatment |

Each guideline is stored as two JSON files:
- `backend/data/guidelines/<id>.json` — decision tree (condition/action nodes + edges)
- `backend/data/evaluators/<id>_eval.json` — condition evaluation logic per node

## Pipeline Flow

Each user message goes through a LangGraph state machine:

1. **Load Patient** — fetch patient record from the database
2. **Triage** — LLM determines urgency (emergency/urgent/moderate/routine) and selects the best-matching NICE guideline. Emergency cases skip directly to step 7 with an immediate referral message
3. **Clarify** — if required clinical variables are missing, generate a clarification question and wait for the user's answer
4. **Select Guideline** — load the guideline JSON decision tree
5. **Extract Variables** — LLM extracts structured clinical variables from the conversation, with regex post-processing for edge cases
6. **Walk Graph** — BFS traversal of the guideline decision tree using extracted variables
7. **Format Recommendation** — produce a structured recommendation with the decision pathway and NICE citation

On completion, a `Diagnosis` record is automatically persisted to the database.

### Urgency Triage

The triage step performs a structured urgency assessment with clinical red flags:

| Level | Action | Examples |
|-------|--------|---------|
| **Emergency** | Immediate referral, skip guideline | Airway compromise, BP >= 180/120 with symptoms, loss of consciousness, sepsis signs, suicidal ideation with plan |
| **Urgent** | Same-day assessment, proceed with guideline | Fever > 38.5C with moderate symptoms, significant pain, acute worsening |
| **Moderate** | 1-3 day assessment | Mild-moderate stable symptoms, low-grade fever |
| **Routine** | Standard GP appointment | Very mild symptoms, monitoring, preventive care |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/R1ch1k/guide-care.git
cd guide-care
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your `OPENAI_API_KEY`. Optionally change `TEAM_PASSWORD` for the login screen.

### 2. Start everything (one command)

```bash
docker-compose up --build
```

This starts all three services:
- **PostgreSQL** on port 5432 (database auto-configured)
- **FastAPI backend** on port 8000 (creates tables and seeds sample patients on first run)
- **Next.js frontend** on port 3000

Open http://localhost:3000 in your browser. API docs at http://localhost:8000/docs.

To stop:

```bash
docker-compose down          # Stop containers (keeps database)
docker-compose down -v       # Stop and delete database volume
```

To rebuild after code changes:

```bash
docker-compose up --build
```

### Alternative: Run without Docker

If you prefer running services manually (requires Node.js 18+ and PostgreSQL):

**Backend:**

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set DATABASE_URL to your PostgreSQL instance and OPENAI_API_KEY

cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

### 3. Use the application

The application has two modes, toggled via the pill switcher in the top-left of the header:

**Consultation mode (🩺)** — patient-specific NICE guideline pipeline:
1. Select a patient from the sidebar (or import your own via the **Connect** button)
2. Describe symptoms in the chat (e.g. "patient has a sore throat, 38.5C fever, no cough")
3. The system will:
   - Triage the symptoms for urgency (emergency/urgent/moderate/routine)
   - Select the appropriate NICE guideline
   - Extract clinical variables from the conversation
   - Ask clarification questions if variables are missing
   - Traverse the guideline decision tree
   - Return the evidence-based recommendation with the decision path
4. After a recommendation, you can ask follow-up questions in the same chat without restarting the pipeline
5. Completed diagnoses are automatically saved and can be exported via the API

**General Chat mode (💬)** — a persistent ChatGPT-style assistant for general clinical queries:
- Ask any medical question: guidelines, drug interactions, clinical reasoning, score interpretation
- Chat history is preserved when switching to Consultation mode and back
- Use the **Clear chat** button to start a fresh conversation
- Suggested starter questions are shown on a fresh chat

## Importing Patient Data

Click the **Connect** button in the patient info panel to open the data import modal.

### CSV Upload

Upload a `.csv` file with the following columns:

| nhs_number | first_name | last_name | date_of_birth | gender | conditions | medications | allergies |
|------------|------------|-----------|---------------|--------|------------|-------------|-----------|
| 123-456-7890 | Jane | Smith | 1985-03-15 | Female | Asthma, Anxiety | Salbutamol | Penicillin |

- `conditions` and `allergies` can be comma-separated strings or JSON arrays
- `medications` can be a JSON array of `{"name": "...", "dose": "..."}` objects or comma-separated names
- A sample CSV template is available for download in the modal

### Excel Upload

Upload a `.xlsx` file with the same column headers in the first row.

## LLM Configuration

### GPT-4o via OpenAI API (default)

The backend uses the OpenAI API (`gpt-4o` by default) for all LLM operations: triage, variable extraction, clarification generation, and guideline selection fallback. Set your key in `.env` (root):

```bash
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o          # or gpt-4o-mini for lower cost
```

The `docker-compose.yml` passes these through as environment variables:

```yaml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
```

### Switching between GPT-4o API and a local model

The backend has a unified `generate()` function in `backend/src/app/llm.py` that routes all LLM calls based on a single environment variable: **`LLM_MODE`**.

| `LLM_MODE` | Behaviour | When to use |
|-------------|-----------|-------------|
| `api` (default) | All LLM calls go to the OpenAI API (GPT-4o) | Standard deployment, best accuracy |
| `local` | Most LLM calls go to a self-hosted model server; **triage always uses the OpenAI API** for reliable urgency classification | Privacy-sensitive / offline / research deployments |

#### How it works under the hood

- `generate()` — used for clarification, variable extraction, guideline selection, and recommendation formatting. Routes to OpenAI or local based on `LLM_MODE`.
- `generate_api_only()` — used **only** for triage (urgency assessment). Always calls the OpenAI API regardless of `LLM_MODE`, because emergency detection must remain highly reliable.

This means even in `local` mode, you still need a valid `OPENAI_API_KEY` for the triage step.

#### To switch to a local model

1. **Set up a local model server** running an OpenAI-compatible API (must expose `/v1/chat/completions`):
   - [vLLM](https://docs.vllm.ai/) — recommended for GPU inference
   - [text-generation-inference](https://huggingface.co/docs/text-generation-inference) — Hugging Face's inference server
   - [Ollama](https://ollama.ai/) — easiest setup for local development

2. **Change one variable** in your `.env` (root for Docker, `backend/.env` without Docker):

   ```bash
   LLM_MODE=local                                    # Switch from "api" to "local"
   LOCAL_MODEL_URL=http://localhost:8080/v1           # Your model server's URL
   LOCAL_MODEL_NAME=gpt-oss-20b                      # Model name your server expects
   OPENAI_API_KEY=sk-proj-your-key-here              # Still needed for triage
   ```

3. **Restart the backend** — no code changes needed:

   ```bash
   # With Docker
   docker-compose up --build

   # Without Docker
   cd backend/src && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Docker Compose configuration

The `docker-compose.yml` already passes these through as environment variables:

```yaml
environment:
  - LLM_MODE=${LLM_MODE:-api}
  - LOCAL_MODEL_URL=${LOCAL_MODEL_URL:-http://host.docker.internal:8080/v1}
  - LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-gpt-oss-20b}
```

When running inside Docker, `host.docker.internal` lets the container reach a model server running on your host machine.

#### Local model options

The system is designed and tested around **gpt-oss-20b**, a 20-billion parameter open-source model fine-tuned for instruction following. However, any OpenAI-compatible model can be used — just change `LOCAL_MODEL_NAME` in your `.env` to match.

| Model | Parameters | VRAM Required | Notes |
|-------|-----------|---------------|-------|
| **gpt-oss-20b** (default) | 20B | ~40 GB (A100) | The model the pipeline prompts are tuned for |
| Mistral-7B-Instruct | 7B | ~16 GB (T4/A10) | Lighter alternative, good general instruction following |
| Llama-3-8B-Instruct | 8B | ~16 GB | Meta's open model, strong reasoning |
| Mixtral-8x7B-Instruct | 47B (MoE) | ~90 GB (2×A100) | Mixture-of-experts, high quality but heavier |

#### Downloading and serving a local model

Models are downloaded from [Hugging Face](https://huggingface.co/models). vLLM handles the download automatically on first run:

```bash
# Install vLLM (requires CUDA-capable GPU)
pip install vllm

# Serve gpt-oss-20b (downloads weights automatically on first launch)
python -m vllm.entrypoints.openai.api_server \
  --model gpt-oss-20b \
  --host 0.0.0.0 --port 8080

# Or serve a different model (e.g. Mistral-7B-Instruct)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 --port 8080
```

The server will expose an OpenAI-compatible API at `http://localhost:8080/v1`.

**Alternative: using Ollama** (simplest setup, no CUDA config needed):

```bash
# Install Ollama — https://ollama.ai
ollama pull mistral           # or any supported model
ollama serve                  # starts on http://localhost:11434

# Then in your .env:
LOCAL_MODEL_URL=http://localhost:11434/v1
LOCAL_MODEL_NAME=mistral
```

**Note:** Pipeline prompts (variable extraction, clarification, formatting) were designed with gpt-oss-20b in mind. Other models will work but may need prompt adjustments for optimal output quality.

## Backend API

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/patients` | List all patients |
| GET | `/patients/{id}` | Get patient details |
| GET | `/patients/{id}/context` | Get patient context for LLM |
| POST | `/patients` | Create a patient |
| PATCH | `/patients/{id}` | Update patient details (conditions, medications, allergies, notes) |
| POST | `/patients/import` | Import patients from CSV/Excel file |
| POST | `/conversations` | Start a new conversation |
| GET | `/conversations/{id}` | Get conversation history |
| GET | `/diagnoses` | List all completed diagnoses |
| GET | `/diagnoses/{id}` | Get a single diagnosis |
| GET | `/diagnoses/export?format=json\|csv` | Export all diagnoses |

### WebSocket

Connect to `/ws/chat/{patient_id}` for real-time chat. Send JSON:

```json
{"role": "user", "content": "patient has a sore throat", "meta": {}}
```

The server streams events back including triage results, clarification questions, and final recommendations.

## Project Structure

```
guide-care/
├── app/                        # Next.js frontend pages
│   ├── page.tsx                # Main application page (Consultation + General Chat modes)
│   ├── api/chat/route.ts       # Legacy guideline-specific chat API route
│   ├── api/general-chat/route.ts # General clinical assistant API route (no guideline required)
│   └── api/parse-pdf/route.ts  # PDF guideline parser
├── components/
│   ├── ChatPanel.tsx           # Consultation chat interface with WebSocket + LangGraph pipeline
│   ├── GeneralChatPanel.tsx    # Persistent general-purpose clinical assistant chat
│   ├── PipelineViewer.tsx      # Real-time LangGraph node visualization
│   ├── PatientInfoPanel.tsx    # Patient details sidebar
│   ├── ConnectDataModal.tsx    # Data import modal (CSV/Excel upload)
│   ├── SampleInputModal.tsx    # Sample inputs with multi-guideline examples
│   ├── DecisionCard.tsx        # Decision pathway visualization
│   ├── AddPatientModal.tsx     # Manual patient creation form
│   └── ui/                     # Shared UI components (shadcn)
├── lib/                        # Frontend utilities and types
├── backend/
│   ├── data/
│   │   ├── guidelines/         # 10 NICE guideline JSON decision trees
│   │   └── evaluators/         # 10 evaluator JSON files
│   ├── src/
│   │   ├── main.py             # FastAPI app entry point
│   │   ├── graph_export.py     # Graph export for LangGraph Studio
│   │   └── app/
│   │       ├── core/config.py  # Settings (DB, OpenAI, CORS)
│   │       ├── db/
│   │       │   ├── models.py   # Patient, Conversation, Diagnosis models
│   │       │   └── session.py  # Async SQLAlchemy session
│   │       ├── api/
│   │       │   ├── patients.py     # Patient CRUD + CSV/Excel import
│   │       │   ├── conversations.py # Conversation endpoints
│   │       │   └── diagnoses.py    # Diagnosis list, detail, export
│   │       ├── guideline_engine.py  # Graph traversal + recommendation formatting
│   │       ├── llm.py              # Async OpenAI wrapper
│   │       ├── orchestration/
│   │       │   ├── graph.py    # LangGraph state machine definition
│   │       │   ├── state.py    # ConversationState TypedDict
│   │       │   ├── deps.py     # Triage, clarify, extract, traverse, format
│   │       │   └── runner.py   # process_user_turn() with astream visualization
│   │       ├── schemas.py      # Pydantic request/response models
│   │       ├── crud.py         # Database CRUD operations
│   │       ├── seed.py         # Sample patient data seeder
│   │       └── ws_manager.py   # WebSocket manager + diagnosis auto-persist
│   ├── tests/                  # Backend test suite (320 tests, 86% coverage)
│   │   ├── conftest.py             # Fixtures: in-memory SQLite, async client
│   │   ├── test_guideline_engine.py # 77 unit tests for engine functions
│   │   ├── test_pipeline_e2e.py    # 35 E2E tests across all 10 guidelines
│   │   ├── test_api.py            # 24 HTTP endpoint integration tests
│   │   ├── test_orchestration.py  # 9 LangGraph pipeline tests (mocked LLM)
│   │   ├── test_crud.py          # 15 database CRUD tests
│   │   ├── test_llm.py           # 19 unit tests for LLM routing (mocked OpenAI)
│   │   ├── test_ws_manager.py    # 39 unit tests for WebSocket manager (mocked)
│   │   ├── test_deps.py          # 89 unit tests for orchestration deps (mocked LLM)
│   │   ├── test_seed_utils.py    # 12 unit tests for seed helpers and retry/timeout
│   │   └── test_patients.py      # 1 legacy patient test
│   ├── langgraph.json          # LangGraph Studio config
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example
├── Dockerfile                 # Frontend Docker image
├── docker-compose.yml         # One-command startup (postgres + backend + frontend)
├── .env.example               # Environment template for Docker
└── package.json
```

## Testing

### Backend Tests (320 tests)

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt

cd src
PYTHONPATH=. pytest -v ../tests
```

To see coverage:

```bash
cd backend/src
PYTHONPATH=. pytest --cov=app --cov-report=term-missing ../tests
```

The backend test suite (`backend/tests/`) includes:

| File | Tests | What is covered |
|------|-------|-----------------|
| `test_guideline_engine.py` | 77 | All 19 pure functions: parse_bp, evaluate_condition, traverse_graph, format_recommendation, fix_variable_extraction, etc. |
| `test_pipeline_e2e.py` | 35 | All 10 NICE guidelines (11 scenarios): graph traversal, recommendation content, no double periods |
| `test_api.py` | 24 | HTTP endpoints: patients CRUD, PATCH, CSV import, conversations, diagnoses list/get/export with real data |
| `test_orchestration.py` | 9 | LangGraph pipeline with mocked LLM: triage routing, clarification, variable extraction, full pipeline |
| `test_crud.py` | 15 | Database CRUD: compute_age, patient/conversation/message operations, update_patient, get_patient_diagnoses |
| `test_llm.py` | 19 | LLM routing: `generate()`, `_generate_api()`, `_generate_local()`, `generate_api_only()` — all with mocked OpenAI client |
| `test_ws_manager.py` | 39 | WebSocket lifecycle, diagnosis auto-persist, patient vitals update, followup mode, orchestration failure paths |
| `test_deps.py` | 89 | `triage_agent`, `gpt_clarifier`, `select_guideline_fn`, `extract_variables_20b` — all with pre-baked mocked LLM responses |
| `test_seed_utils.py` | 12 | `calculate_age`, `seed_if_empty`, `with_retry_timeout`, `log_step` |
| `test_patients.py` | 1 | Legacy patient CRUD test |

All LLM-dependent tests use `unittest.mock.AsyncMock` and `patch` to inject pre-baked JSON responses — no live API key required to run the test suite.

Tests use in-memory SQLite by default (no Docker needed). Set `TEST_DATABASE_URL` to point at a PostgreSQL instance instead.

Overall coverage: **86%** — core business logic (`guideline_engine`, `crud`, `schemas`, `graph`) is at 84–100%; the LLM and WebSocket layers (`llm.py`, `ws_manager.py`, `deps.py`) are at 82–100% via mocked unit tests.

## LangGraph Studio (Visual Pipeline Debugging)

LangGraph Studio is the official browser-based IDE by LangChain for visualizing LangGraph state machines. It shows the 7-node pipeline graph with real-time execution flow.

```bash
cd backend
pip install langgraph-cli

langgraph dev
# Opens browser at https://smith.langchain.com/studio/?baseUrl=http://localhost:2024
```

This shows the full pipeline graph: `load_patient -> triage -> clarify -> select_guideline -> extract_variables -> walk_graph -> format_output`. You can send messages and watch execution flow through nodes in real-time.

The in-app pipeline viewer in the chat UI also shows which nodes were visited for each conversation turn.

## Environment Variables

When using Docker, all variables are set in the root `.env` file (copied from `.env.example`). Docker Compose passes them to each service automatically.

### Root (`.env`) — used by Docker Compose

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | — | OpenAI API key for LLM features |
| OPENAI_MODEL | No | gpt-4o | OpenAI model to use |
| TEAM_PASSWORD | No | changeme123 | Login password for the frontend |
| LLM_MODE | No | api | `api` for OpenAI, `local` for local model |
| LOCAL_MODEL_URL | No | — | OpenAI-compatible endpoint for local model |
| LOCAL_MODEL_NAME | No | gpt-oss-20b | Local model name |

### Backend (`backend/.env`) — used without Docker

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | — | PostgreSQL async connection string |
| OPENAI_API_KEY | Yes | — | OpenAI API key for LLM features |
| OPENAI_MODEL | No | gpt-4o | OpenAI model to use |
| CORS_ORIGINS | No | * | Comma-separated allowed origins |
| LLM_MODE | No | api | `api` for OpenAI, `local` for local model |

### Frontend (`.env.local`) — used without Docker

| Variable | Required | Description |
|----------|----------|-------------|
| NEXT_PUBLIC_BACKEND_URL | No | Backend URL (defaults to http://localhost:8000) |
| TEAM_PASSWORD | No | Login password (defaults to changeme123) |
| OPENAI_API_KEY | Yes (for General Chat) | OpenAI API key — used by the `/api/general-chat` Next.js route |
| OPENAI_MODEL | No | OpenAI model for General Chat (defaults to gpt-4o) |

## Patient Data Privacy

The system is designed to minimise patient data exposure to external LLM services:

**What is sent to the LLM (GPT-4o or local model):**
- Age, gender, medical conditions, current medications, allergies
- Symptoms described in the chat conversation
- Extracted clinical variables (e.g. blood pressure readings, fever temperature)

**What is NEVER sent to the LLM:**
- Patient name (first or last)
- NHS number
- Date of birth
- Full clinical notes or doctor's notes

PII fields are loaded from the database for display in the frontend UI only. The LLM prompts in `backend/src/app/llm.py` and `backend/src/app/orchestration/deps.py` are constructed using only clinically relevant, de-identified data.

For deployments where no patient data should leave the network, set `LLM_MODE=local` to route LLM calls to a self-hosted model (see [Switching between GPT-4o API and a local model](#switching-between-gpt-4o-api-and-a-local-model)). Note that triage still uses the OpenAI API in local mode — for fully offline operation, a TRIAGE_API_URL can be configured to point to a local triage service.

## Safety Notice

This tool is for healthcare professionals only and does not replace clinical judgment. Always consider individual patient context, contraindications, and local protocols. All recommendations cite the source NICE guideline.
