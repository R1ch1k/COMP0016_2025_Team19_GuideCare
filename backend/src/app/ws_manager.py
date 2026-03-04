import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Set
from uuid import UUID, uuid4

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Conversation, Diagnosis, Patient
from app.db.session import AsyncSessionLocal
from app.orchestration.runner import process_user_turn

# Guideline ID → condition name mapping for auto-updating patient conditions
_GUIDELINE_CONDITION_MAP = {
    "NG84": "Sore throat",
    "NG91": "Otitis media",
    "NG112": "Urinary tract infection",
    "NG133": "Hypertension in pregnancy",
    "NG136": "Hypertension",
    "NG184": "Animal/human bite",
    "NG222": "Depression",
    "NG232": "Head injury",
    "NG81_GLAUCOMA": "Chronic glaucoma",
    "NG81_HYPERTENSION": "Ocular hypertension",
}

logger = logging.getLogger("ws_manager")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: Dict[str, Set[WebSocket]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self._orch_graph = None

    def set_orchestrator(self, graph) -> None:
        self._orch_graph = graph
        logger.info("Orchestrator connected to ws_manager")

    async def connect(self, patient_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.setdefault(patient_id, set()).add(websocket)
        self.locks.setdefault(patient_id, asyncio.Lock())
        logger.info("WebSocket connected for patient %s", patient_id)

    async def disconnect(self, patient_id: str, websocket: WebSocket) -> None:
        conns = self.active.get(patient_id)
        if not conns:
            return
        conns.discard(websocket)
        if not conns:
            self.active.pop(patient_id, None)
            self.locks.pop(patient_id, None)
        logger.info("WebSocket disconnected for patient %s", patient_id)

    async def broadcast(self, patient_id: str, message: dict) -> None:
        conns = self.active.get(patient_id, set())
        dead: Set[WebSocket] = set()
        for ws in list(conns):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
                logger.exception("Failed to send websocket message (patient=%s)", patient_id)

        for ws in dead:
            conns.discard(ws)
        if not conns and patient_id in self.active:
            self.active.pop(patient_id, None)
            self.locks.pop(patient_id, None)

    async def _get_or_create_conversation(self, db, pid: UUID) -> Conversation:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.patient_id == pid)
            .where(Conversation.status == "in_progress")
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        conv = result.scalars().first()
        if conv:
            return conv

        conv = Conversation(patient_id=pid, messages=[], status="in_progress")
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    async def _close_current_conversation(self, pid: UUID) -> None:
        """Mark the current in-progress conversation as completed so a new one is created."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Conversation)
                .where(Conversation.patient_id == pid)
                .where(Conversation.status == "in_progress")
                .order_by(Conversation.updated_at.desc())
                .limit(1)
            )
            conv = result.scalars().first()
            if conv:
                conv.status = "completed"
                conv.updated_at = datetime.now(timezone.utc)
                await db.commit()
                logger.info("Closed conversation %s for patient %s", conv.id, pid)

    async def _update_patient_from_diagnosis(self, db, pid: UUID, event: dict) -> None:
        """Enrich the patient record with data from a completed diagnosis.

        Updates:
        - clinical_notes: appends a visit summary (keeps last 10)
        - conditions: adds the diagnosed condition if not already present
        - recent_vitals: updates BP readings from extracted variables
        """
        try:
            patient = await db.get(Patient, pid)
            if not patient:
                return

            guideline_id = event.get("selected_guideline") or ""
            recommendation = event.get("final_recommendation") or ""
            extracted = event.get("extracted_variables") or {}
            urgency = "urgent" if event.get("urgent_escalation") else event.get("meta", {}).get("urgency")

            # 1. Append clinical note (visit summary)
            notes = list(patient.clinical_notes or [])
            visit_note = {
                "date": datetime.now(timezone.utc).isoformat(),
                "guideline": guideline_id,
                "recommendation": recommendation[:500],  # Truncate for storage
                "urgency": urgency,
                "variables": {k: v for k, v in extracted.items() if v is not None},
            }
            notes.insert(0, visit_note)
            patient.clinical_notes = notes[:10]  # Keep last 10 visits

            # 2. Add condition if not already present
            conditions = list(patient.conditions or [])
            for gid_prefix, condition_name in _GUIDELINE_CONDITION_MAP.items():
                if guideline_id.lower().startswith(gid_prefix.lower()):
                    if condition_name not in conditions:
                        conditions.append(condition_name)
                    break
            patient.conditions = conditions

            # 3. Update recent vitals (BP, temp, etc.)
            vitals = dict(patient.recent_vitals or {})
            if extracted.get("clinic_bp"):
                vitals["bp"] = extracted["clinic_bp"]
            if extracted.get("abpm_daytime"):
                vitals["abpm_daytime"] = extracted["abpm_daytime"]
            if extracted.get("hbpm_average"):
                vitals["hbpm_average"] = extracted["hbpm_average"]
            if extracted.get("temperature") or extracted.get("fever"):
                temp = extracted.get("temperature")
                if isinstance(temp, (int, float)):
                    vitals["temperature"] = temp
            patient.recent_vitals = vitals

            patient.updated_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info("Patient record updated from diagnosis for patient %s", pid)
        except Exception:
            logger.exception("Failed to update patient record from diagnosis")

    async def handle_incoming_message(self, patient_id: str, message: dict) -> None:
        if self._orch_graph is None:
            await self.broadcast(patient_id, {"type": "error", "detail": "Orchestrator not initialized"})
            return

        try:
            pid = UUID(patient_id)
        except Exception:
            await self.broadcast(patient_id, {"type": "error", "detail": "Invalid patient_id"})
            return

        # Handle "new_conversation" signal: close current conversation so next
        # message creates a fresh one with clean LangGraph state.
        # No ack broadcast — the frontend may close the WS right after sending this.
        if message.get("type") == "new_conversation":
            lock = self.locks.setdefault(patient_id, asyncio.Lock())
            async with lock:
                await self._close_current_conversation(pid)
            logger.info("New conversation requested for patient %s", patient_id)
            return

        lock = self.locks.setdefault(patient_id, asyncio.Lock())

        user_record = {
            "id": str(uuid4()),
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
            "meta": message.get("meta") or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Persist user message
        async with lock:
            async with AsyncSessionLocal() as db:
                try:
                    conv = await self._get_or_create_conversation(db, pid)

                    msgs = conv.messages or []
                    msgs.append(user_record)
                    conv.messages = msgs
                    conv.updated_at = datetime.now(timezone.utc)

                    await db.commit()
                    await db.refresh(conv)
                except IntegrityError:
                    await db.rollback()
                    logger.exception("Integrity error saving user message")
                    await self.broadcast(patient_id, {"type": "error", "detail": "Patient not found"})
                    return
                except Exception:
                    await db.rollback()
                    logger.exception("DB error saving user message")
                    await self.broadcast(patient_id, {"type": "error", "detail": "Failed to save message"})
                    return

        # Broadcast user message
        await self.broadcast(
            patient_id,
            {"type": "message", "message": user_record, "conversation_id": str(conv.id)},
        )

        # Run orchestration for this turn (no DB lock)
        try:
            event = await process_user_turn(
                graph=self._orch_graph,
                patient_id=patient_id,
                conversation_id=str(conv.id),
                user_message=user_record,
            )
        except Exception:
            logger.exception("Orchestration failed")
            await self.broadcast(patient_id, {"type": "error", "detail": "Orchestration failed"})
            return

        if not event:
            return

        assistant_msg = {
            "id": str(uuid4()),
            "role": "assistant",
            "content": event.get("content", ""),
            "meta": event.get("meta") or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Persist assistant output + structured fields
        async with lock:
            async with AsyncSessionLocal() as db:
                try:
                    conv2 = await db.get(Conversation, UUID(str(conv.id)))
                    if conv2:
                        msgs = conv2.messages or []
                        msgs.append(assistant_msg)
                        conv2.messages = msgs

                        if event.get("final_recommendation"):
                            conv2.final_recommendation = event["final_recommendation"]
                        if event.get("selected_guideline"):
                            conv2.selected_guideline = event["selected_guideline"]
                        if event.get("extracted_variables") is not None:
                            conv2.extracted_variables = event["extracted_variables"]
                        if event.get("status"):
                            conv2.status = event["status"]

                        conv2.updated_at = datetime.now(timezone.utc)
                        await db.commit()

                        # Auto-create Diagnosis record when conversation completes
                        if event.get("status") == "completed" and event.get("final_recommendation"):
                            try:
                                diag = Diagnosis(
                                    patient_id=pid,
                                    conversation_id=UUID(str(conv.id)),
                                    selected_guideline=event.get("selected_guideline"),
                                    extracted_variables=event.get("extracted_variables") or {},
                                    pathway_walked=event.get("pathway_walked") or [],
                                    final_recommendation=event["final_recommendation"],
                                    urgency="urgent" if event.get("urgent_escalation") else event.get("meta", {}).get("urgency"),
                                )
                                db.add(diag)
                                await db.commit()
                                logger.info("Diagnosis record created for patient %s", patient_id)

                                # Auto-update patient record with diagnosis info
                                await self._update_patient_from_diagnosis(db, pid, event)
                            except Exception:
                                await db.rollback()
                                logger.exception("Failed to create diagnosis record")
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to persist assistant event")

        await self.broadcast(
            patient_id,
            {
                "type": event.get("type", "assistant_event"),
                "message": assistant_msg,
                "conversation_id": str(conv.id),
                "payload": event,
            },
        )


manager = ConnectionManager()
