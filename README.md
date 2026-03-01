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
- Node.js 18+
- An OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/R1ch1k/guide-care.git
cd guide-care
```

Create the backend environment file:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set your `OPENAI_API_KEY`.

### 2. Start the backend (Docker)

```bash
docker-compose up --build
```

This starts:
- **PostgreSQL** on port 5432 (database: `guidecare`, user: `guidecare`, password: `guidecare`)
- **FastAPI backend** on port 8000

On first startup, the backend creates all database tables and seeds sample patients.

API documentation is available at http://localhost:8000/docs.

To stop:

```bash
docker-compose down          # Stop containers (keeps data)
docker-compose down -v       # Stop and delete database volume
```

To rebuild after code changes:

```bash
docker-compose up --build    # Rebuilds the backend image
```

### 2b. Start the backend (without Docker)

If you prefer running without Docker, you need a PostgreSQL instance running separately:

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set DATABASE_URL to your PostgreSQL instance and OPENAI_API_KEY

cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the frontend

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

### 4. Use the application

1. Select a patient from the sidebar (or import your own via the **Connect** button)
2. Describe symptoms in the chat (e.g. "patient has a sore throat, 38.5C fever, no cough")
3. The system will:
   - Triage the symptoms for urgency (emergency/urgent/moderate/routine)
   - Select the appropriate NICE guideline
   - Extract clinical variables from the conversation
   - Ask clarification questions if variables are missing
   - Traverse the guideline decision tree
   - Return the evidence-based recommendation with the decision path
4. Completed diagnoses are automatically saved and can be exported via the API

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

The backend uses the OpenAI API (`gpt-4o` by default) for all LLM operations: triage, variable extraction, clarification generation, and guideline selection fallback. Set your key in `backend/.env`:

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

### Using gpt-oss-20b (local model, for research / offline use)

The test notebooks in `testing/` support running with a local 21B-parameter MoE model (3.6B active parameters) called gpt-oss-20b. This model runs on a single A100 GPU.

**To use in the test notebooks (Google Colab):**

1. Open `testing/gpt_oss_20b_final_test_suite.ipynb` on Google Colab
2. Select an A100 GPU runtime
3. In the configuration cell, set `USE_API = False`
4. The notebook will download and load the model in BF16
5. All pipeline tests (variable extraction, clarification, recommendation formatting) will run against the local model

**To switch back to API mode:**

1. Set `USE_API = True`
2. Set your API key in the `API_KEY` variable (or use Colab secrets)

**To use in the backend (advanced):**

The `docker-compose.yml` already includes environment variables for a local model:

```yaml
environment:
  - LLM_MODE=${LLM_MODE:-api}                    # Set to "local" for local model
  - LOCAL_MODEL_URL=${LOCAL_MODEL_URL:-http://host.docker.internal:8080/v1}
  - LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-gpt-oss-20b}
```

To use a local model with the backend:
1. Host gpt-oss-20b (or any OpenAI-compatible model) on a GPU server with an OpenAI-compatible API endpoint (e.g. vLLM, text-generation-inference)
2. Set `LLM_MODE=local` in your `.env`
3. Set `LOCAL_MODEL_URL` to your model's API endpoint
4. The backend's `app/llm.py` will route requests to the local endpoint instead of OpenAI

**Note:** The local model achieves comparable pipeline scores to GPT-4o on the test suite (see Test Results below), making it viable for privacy-sensitive deployments where data must not leave the network.

## Backend API

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/patients` | List all patients |
| GET | `/patients/{id}` | Get patient details |
| GET | `/patients/{id}/context` | Get patient context for LLM |
| POST | `/patients` | Create a patient |
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
│   ├── page.tsx                # Main application page
│   ├── api/chat/route.ts       # Legacy chat API route
│   └── api/parse-pdf/route.ts  # PDF guideline parser
├── components/
│   ├── ChatPanel.tsx           # Main chat interface with WebSocket
│   ├── PatientInfoPanel.tsx    # Patient details sidebar
│   ├── ConnectDataModal.tsx    # Data import modal (CSV/Excel upload)
│   ├── SampleInputModal.tsx    # Sample inputs with multi-guideline examples
│   ├── DecisionCard.tsx        # Decision pathway visualization
│   ├── AddPatientModal.tsx     # Manual patient creation form
│   └── ui/                     # Shared UI components (shadcn)
├── lib/                        # Frontend utilities and types
├── testing/                    # Test notebooks (Colab)
│   ├── gpt_api_final_test_suite.ipynb
│   ├── gpt_oss_20b_final_test_suite.ipynb
│   └── medical_chatbot_triage_testing.ipynb
├── backend/
│   ├── data/
│   │   ├── guidelines/         # 10 NICE guideline JSON decision trees
│   │   └── evaluators/         # 10 evaluator JSON files
│   ├── src/
│   │   ├── main.py             # FastAPI app entry point
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
│   │       │   └── runner.py   # process_user_turn() entry point
│   │       ├── schemas.py      # Pydantic request/response models
│   │       ├── crud.py         # Database CRUD operations
│   │       ├── seed.py         # Sample patient data seeder
│   │       └── ws_manager.py   # WebSocket manager + diagnosis auto-persist
│   ├── tests/                  # Backend unit + E2E tests
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── docker-compose.yml
└── package.json
```

## Testing

### Backend Tests (146 tests)

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt

cd src
PYTHONPATH=. pytest -v ../tests
```

The backend test suite (`backend/tests/`) includes:

| File | Tests | Coverage |
|------|-------|----------|
| `test_guideline_engine.py` | 79 | All 19 pure functions: parse_bp, evaluate_condition, traverse_graph, format_recommendation, fix_variable_extraction, etc. |
| `test_pipeline_e2e.py` | 35 | All 10 NICE guidelines (11 scenarios): graph traversal, recommendation content, no double periods |
| `test_api.py` | 15 | HTTP endpoints: patients CRUD, CSV import, conversations, diagnoses export |
| `test_orchestration.py` | 8 | LangGraph pipeline with mocked LLM: triage routing, clarification, variable extraction, full pipeline |
| `test_crud.py` | 8 | Database CRUD: compute_age, patient/conversation/message operations |
| `test_patients.py` | 1 | Legacy patient CRUD test |

Tests use in-memory SQLite by default (no Docker needed). Set `TEST_DATABASE_URL` for PostgreSQL.

### Test Notebooks (Colab)

Three Colab notebooks in `testing/` validate the pipeline independently:

| Notebook | Purpose | Requirements |
|----------|---------|-------------|
| `gpt_api_final_test_suite.ipynb` | Full pipeline validation using GPT-4o API | OpenAI API key |
| `gpt_oss_20b_final_test_suite.ipynb` | Full pipeline validation using local model | A100 GPU on Colab |
| `medical_chatbot_triage_testing.ipynb` | Triage urgency assessment + red flag detection | OpenAI API key |

**Test coverage:** variable extraction (30 cases), multi-turn clarification (10 cases), recommendation formatting (10 cases), error handling (10 cases).

**Latest API results:** 10/10 pipeline success, 85% extraction accuracy, 96.2% overall score.

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

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DATABASE_URL | Yes | — | PostgreSQL async connection string |
| OPENAI_API_KEY | Yes | — | OpenAI API key for LLM features |
| OPENAI_MODEL | No | gpt-4o | OpenAI model to use |
| CORS_ORIGINS | No | * | Comma-separated allowed origins |
| LLM_MODE | No | api | `api` for OpenAI, `local` for local model |
| LOCAL_MODEL_URL | No | — | OpenAI-compatible endpoint for local model |
| LOCAL_MODEL_NAME | No | gpt-oss-20b | Local model name |

### Frontend (`.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| NEXT_PUBLIC_BACKEND_URL | No | Backend URL (defaults to http://localhost:8000) |

## Safety Notice

This tool is for healthcare professionals only and does not replace clinical judgment. Always consider individual patient context, contraindications, and local protocols. All recommendations cite the source NICE guideline.
