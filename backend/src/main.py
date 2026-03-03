import logging
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import conversations, diagnoses, patients
from app.db.session import init_db
from app.seed import seed_if_empty
from app.ws_manager import manager

from app.orchestration.deps import build_orchestration_deps
from app.orchestration.graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guidecare-backend")

app = FastAPI(title="GuideCare Backend")

# CORS (browser-safe)
if settings.CORS_ORIGINS:
    allow_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    allow_credentials = True
else:
    allow_origins = ["*"]
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/patients", tags=["patients"])
app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
app.include_router(diagnoses.router, prefix="/diagnoses", tags=["diagnoses"])


@app.on_event("startup")
async def on_startup():
    await init_db()
    await seed_if_empty()

    # Build orchestration ONCE
    deps = build_orchestration_deps()
    graph = build_graph(deps)

    app.state.orch_deps = deps
    app.state.orch_graph = graph

    # Wire into websocket manager
    manager.set_orchestrator(graph)


@app.websocket("/ws/chat/{patient_id}")
async def websocket_chat(websocket: WebSocket, patient_id: UUID):
    await manager.connect(str(patient_id), websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_incoming_message(str(patient_id), data)
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # Happens when the client closes mid-receive (e.g. New Conversation reconnect)
        pass
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        await manager.disconnect(str(patient_id), websocket)
