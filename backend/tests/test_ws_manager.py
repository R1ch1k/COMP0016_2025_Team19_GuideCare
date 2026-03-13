"""
Tests for app.ws_manager.ConnectionManager — all WebSocket, DB, and LLM calls
are mocked so no live connections or API keys are needed.
"""

import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.ws_manager import ConnectionManager, _GUIDELINE_CONDITION_MAP


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ws():
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


def _make_conv(pid=None, status="in_progress", final_rec=None):
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.patient_id = pid or uuid.uuid4()
    conv.messages = []
    conv.status = status
    conv.final_recommendation = final_rec
    conv.selected_guideline = None
    conv.extracted_variables = None
    conv.updated_at = datetime.now(timezone.utc)
    return conv


def _make_patient():
    p = MagicMock()
    p.id = uuid.uuid4()
    p.clinical_notes = []
    p.conditions = []
    p.recent_vitals = {}
    p.medications = []
    p.updated_at = None
    return p


def _db_mock(conv=None, patient=None):
    """Return (mock_db, mock_AsyncSessionLocal) pair."""
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = conv
    mock_db.execute = AsyncMock(return_value=result)
    mock_db.get = AsyncMock(return_value=patient or conv)
    mock_db.commit = AsyncMock()
    mock_db.rollback = AsyncMock()
    mock_db.refresh = AsyncMock()
    mock_db.add = MagicMock()

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_db)
    cm.__aexit__ = AsyncMock(return_value=False)

    mock_sl = MagicMock(return_value=cm)
    return mock_db, mock_sl


# ── ConnectionManager basics ───────────────────────────────────────────────────

class TestConnectionManagerBasics:
    def test_initial_state(self):
        mgr = ConnectionManager()
        assert mgr.active == {}
        assert mgr.locks == {}
        assert mgr._orch_graph is None

    def test_set_orchestrator(self):
        mgr = ConnectionManager()
        fake_graph = MagicMock()
        mgr.set_orchestrator(fake_graph)
        assert mgr._orch_graph is fake_graph

    @pytest.mark.asyncio
    async def test_connect_accepts_and_registers(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect("patient-1", ws)
        ws.accept.assert_awaited_once()
        assert ws in mgr.active["patient-1"]
        assert "patient-1" in mgr.locks

    @pytest.mark.asyncio
    async def test_connect_multiple_sockets_same_patient(self):
        mgr = ConnectionManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await mgr.connect("p1", ws1)
        await mgr.connect("p1", ws2)
        assert len(mgr.active["p1"]) == 2

    @pytest.mark.asyncio
    async def test_disconnect_removes_socket(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect("p1", ws)
        await mgr.disconnect("p1", ws)
        assert "p1" not in mgr.active

    @pytest.mark.asyncio
    async def test_disconnect_keeps_others(self):
        mgr = ConnectionManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await mgr.connect("p1", ws1)
        await mgr.connect("p1", ws2)
        await mgr.disconnect("p1", ws1)
        assert ws2 in mgr.active["p1"]
        assert ws1 not in mgr.active["p1"]

    @pytest.mark.asyncio
    async def test_disconnect_noop_when_not_connected(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        # Should not raise
        await mgr.disconnect("nobody", ws)


# ── broadcast() ───────────────────────────────────────────────────────────────

class TestBroadcast:
    @pytest.mark.asyncio
    async def test_sends_to_all_connected(self):
        mgr = ConnectionManager()
        ws1, ws2 = _make_ws(), _make_ws()
        await mgr.connect("p1", ws1)
        await mgr.connect("p1", ws2)
        await mgr.broadcast("p1", {"type": "test"})
        ws1.send_json.assert_awaited_once_with({"type": "test"})
        ws2.send_json.assert_awaited_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_dead_socket_removed_on_broadcast(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        ws.send_json.side_effect = Exception("connection closed")
        await mgr.connect("p1", ws)
        # Should not raise; dead socket should be cleaned up
        await mgr.broadcast("p1", {"type": "ping"})
        assert "p1" not in mgr.active

    @pytest.mark.asyncio
    async def test_broadcast_to_unknown_patient_noop(self):
        mgr = ConnectionManager()
        # Should not raise
        await mgr.broadcast("nobody", {"type": "ping"})


# ── _answer_followup() ────────────────────────────────────────────────────────

class TestAnswerFollowup:
    @pytest.mark.asyncio
    async def test_returns_llm_answer(self):
        mgr = ConnectionManager()
        with patch("app.ws_manager.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "You should continue amlodipine."
            result = await mgr._answer_followup(
                "Should I continue the medication?",
                "Patient has Stage 2 hypertension. Start amlodipine 5mg.",
            )
            assert result == "You should continue amlodipine."
            mock_gen.assert_awaited_once()
            # Prompt should reference both recommendation and question
            prompt = mock_gen.call_args.args[0]
            assert "amlodipine" in prompt
            assert "medication" in prompt

    @pytest.mark.asyncio
    async def test_uses_low_temperature(self):
        mgr = ConnectionManager()
        with patch("app.ws_manager.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Answer"
            await mgr._answer_followup("Q", "Rec")
            kwargs = mock_gen.call_args.kwargs
            assert kwargs.get("temperature", mock_gen.call_args.args[2] if len(mock_gen.call_args.args) > 2 else 0) == 0.0


# ── _update_patient_from_diagnosis() ─────────────────────────────────────────

class TestUpdatePatientFromDiagnosis:
    @pytest.mark.asyncio
    async def test_appends_clinical_note(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Start amlodipine",
            "extracted_variables": {"clinic_bp": "155/95"},
            "urgent_escalation": False,
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)

        mock_db.commit.assert_awaited()
        assert len(patient.clinical_notes) == 1
        note = patient.clinical_notes[0]
        assert note["guideline"] == "NG136"

    @pytest.mark.asyncio
    async def test_adds_condition_from_guideline(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        patient.conditions = []
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Manage hypertension",
            "extracted_variables": {},
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert "Hypertension" in patient.conditions

    @pytest.mark.asyncio
    async def test_updates_bp_vitals(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Rec",
            "extracted_variables": {"clinic_bp": "160/100"},
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert patient.recent_vitals["bp"] == "160/100"

    @pytest.mark.asyncio
    async def test_patient_not_found_skips_gracefully(self):
        mgr = ConnectionManager()
        mock_db, _ = _db_mock()
        mock_db.get = AsyncMock(return_value=None)

        # Should not raise
        await mgr._update_patient_from_diagnosis(mock_db, uuid.uuid4(), {"selected_guideline": "NG136"})
        mock_db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_keeps_at_most_10_notes(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        patient.clinical_notes = [{"note": i} for i in range(10)]
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {"selected_guideline": "NG84", "final_recommendation": "New rec", "extracted_variables": {}}
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert len(patient.clinical_notes) == 10


# ── handle_incoming_message() ─────────────────────────────────────────────────

class TestHandleIncomingMessage:
    @pytest.mark.asyncio
    async def test_no_orchestrator_broadcasts_error(self):
        mgr = ConnectionManager()
        ws = _make_ws()
        await mgr.connect("patient-1", ws)

        await mgr.handle_incoming_message("patient-1", {"role": "user", "content": "hi"})
        ws.send_json.assert_awaited_once()
        payload = ws.send_json.call_args.args[0]
        assert payload["type"] == "error"

    @pytest.mark.asyncio
    async def test_invalid_uuid_broadcasts_error(self):
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        ws = _make_ws()
        await mgr.connect("not-a-uuid", ws)

        await mgr.handle_incoming_message("not-a-uuid", {"role": "user", "content": "hi"})
        ws.send_json.assert_awaited_once()
        payload = ws.send_json.call_args.args[0]
        assert payload["type"] == "error"
        assert "Invalid" in payload["detail"]

    @pytest.mark.asyncio
    async def test_new_conversation_signal_closes_and_returns(self):
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl):
            await mgr.handle_incoming_message(pid, {"type": "new_conversation"})

        # Should return early — no broadcast (frontend expects no ack)
        ws.send_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normal_message_broadcasts_user_and_assistant(self):
        mgr = ConnectionManager()
        graph = MagicMock()
        mgr.set_orchestrator(graph)
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        orch_event = {
            "type": "assistant_event",
            "content": "Ask the patient about fever duration.",
            "meta": {},
        }
        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch("app.ws_manager.process_user_turn", new_callable=AsyncMock) as mock_orch:
            mock_orch.return_value = orch_event
            await mgr.handle_incoming_message(
                pid,
                {"role": "user", "content": "Patient has a sore throat"},
            )

        # ws.send_json called twice: user message + assistant event
        assert ws.send_json.await_count == 2
        calls = [c.args[0] for c in ws.send_json.await_args_list]
        types = [c["type"] for c in calls]
        assert "message" in types
        assert "assistant_event" in types

    @pytest.mark.asyncio
    async def test_completed_event_creates_diagnosis(self):
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        orch_event = {
            "type": "final",
            "content": "Start amlodipine 5mg once daily.",
            "meta": {},
            "status": "completed",
            "final_recommendation": "Start amlodipine 5mg once daily.",
            "selected_guideline": "NG136",
            "extracted_variables": {"clinic_bp": "160/100"},
            "pathway_walked": ["start", "n1", "n2"],
        }
        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch("app.ws_manager.process_user_turn", new_callable=AsyncMock) as mock_orch, \
             patch.object(mgr, "_update_patient_from_diagnosis", new_callable=AsyncMock):
            mock_orch.return_value = orch_event
            await mgr.handle_incoming_message(
                pid,
                {"role": "user", "content": "BP is 160/100"},
            )

        # A Diagnosis object should have been added to the DB
        mock_db.add.assert_called()

    @pytest.mark.asyncio
    async def test_followup_message_uses_llm_answer(self):
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch.object(mgr, "_get_latest_recommendation", new_callable=AsyncMock) as mock_rec, \
             patch.object(mgr, "_answer_followup", new_callable=AsyncMock) as mock_ans:
            mock_rec.return_value = "Start amlodipine 5mg."
            mock_ans.return_value = "Continue the medication as prescribed."

            await mgr.handle_incoming_message(
                pid,
                {
                    "role": "user",
                    "content": "Should I continue this medication?",
                    "meta": {"followup": True},
                },
            )

        mock_ans.assert_awaited_once()
        # Should broadcast followup answer
        calls = [c.args[0] for c in ws.send_json.await_args_list]
        followup_calls = [c for c in calls if c.get("type") == "assistant_event"]
        assert followup_calls
        assert followup_calls[0]["payload"]["type"] == "followup"

    @pytest.mark.asyncio
    async def test_db_integrity_error_broadcasts_error(self):
        from sqlalchemy.exc import IntegrityError
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.commit.side_effect = IntegrityError("", {}, Exception())

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl):
            await mgr.handle_incoming_message(pid, {"role": "user", "content": "hi"})

        error_calls = [c.args[0] for c in ws.send_json.await_args_list if c.args[0].get("type") == "error"]
        assert error_calls

    @pytest.mark.asyncio
    async def test_orchestration_exception_broadcasts_error(self):
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch("app.ws_manager.process_user_turn", new_callable=AsyncMock) as mock_orch:
            mock_orch.side_effect = RuntimeError("Orchestration failed")
            await mgr.handle_incoming_message(pid, {"role": "user", "content": "hi"})

        error_calls = [c.args[0] for c in ws.send_json.await_args_list if c.args[0].get("type") == "error"]
        assert error_calls

    @pytest.mark.asyncio
    async def test_empty_orchestration_result_no_broadcast(self):
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch("app.ws_manager.process_user_turn", new_callable=AsyncMock) as mock_orch:
            mock_orch.return_value = None  # No event
            await mgr.handle_incoming_message(pid, {"role": "user", "content": "hi"})

        # Only user message broadcast; no assistant broadcast
        calls = [c.args[0] for c in ws.send_json.await_args_list]
        assert all(c["type"] == "message" for c in calls)


# ── _get_or_create_conversation() ────────────────────────────────────────────

class TestGetOrCreateConversation:
    @pytest.mark.asyncio
    async def test_returns_existing_conversation(self):
        mgr = ConnectionManager()
        existing = _make_conv()
        mock_db, _ = _db_mock(conv=existing)

        result = await mgr._get_or_create_conversation(mock_db, existing.patient_id)
        assert result is existing
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_new_when_none_exists(self):
        mgr = ConnectionManager()
        mock_db, _ = _db_mock(conv=None)
        pid = uuid.uuid4()

        # When no conv exists, a new one is created and added
        await mgr._get_or_create_conversation(mock_db, pid)
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited()


# ── _get_latest_recommendation() ────────────────────────────────────────────

class TestGetLatestRecommendation:
    @pytest.mark.asyncio
    async def test_returns_recommendation_when_exists(self):
        conv = _make_conv(status="completed", final_rec="Take amlodipine.")
        mock_db, mock_sl = _db_mock(conv=conv)

        mgr = ConnectionManager()
        with patch("app.ws_manager.AsyncSessionLocal", mock_sl):
            result = await mgr._get_latest_recommendation(uuid.uuid4())
        assert result == "Take amlodipine."

    @pytest.mark.asyncio
    async def test_returns_none_when_no_completed_conv(self):
        mock_db, mock_sl = _db_mock(conv=None)

        mgr = ConnectionManager()
        with patch("app.ws_manager.AsyncSessionLocal", mock_sl):
            result = await mgr._get_latest_recommendation(uuid.uuid4())
        assert result is None


# ── Additional edge cases ──────────────────────────────────────────────────────

class TestUpdatePatientFromDiagnosisEdgeCases:
    @pytest.mark.asyncio
    async def test_updates_abpm_daytime_vitals(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Rec",
            "extracted_variables": {"abpm_daytime": "145/90"},
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert patient.recent_vitals.get("abpm_daytime") == "145/90"

    @pytest.mark.asyncio
    async def test_updates_hbpm_vitals(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Rec",
            "extracted_variables": {"hbpm_average": "148/92"},
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert patient.recent_vitals.get("hbpm_average") == "148/92"

    @pytest.mark.asyncio
    async def test_updates_temperature_vitals(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG84",
            "final_recommendation": "Rec",
            "extracted_variables": {"temperature": 38.5},
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert patient.recent_vitals.get("temperature") == 38.5

    @pytest.mark.asyncio
    async def test_urgent_escalation_sets_urgency(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Emergency",
            "extracted_variables": {},
            "urgent_escalation": True,
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert patient.clinical_notes[0]["urgency"] == "urgent"

    @pytest.mark.asyncio
    async def test_does_not_duplicate_condition(self):
        mgr = ConnectionManager()
        patient = _make_patient()
        patient.conditions = ["Hypertension"]  # Already has the condition
        mock_db, _ = _db_mock(patient=patient)
        mock_db.get = AsyncMock(return_value=patient)

        event = {
            "selected_guideline": "NG136",
            "final_recommendation": "Continue treatment",
            "extracted_variables": {},
        }
        await mgr._update_patient_from_diagnosis(mock_db, patient.id, event)
        assert patient.conditions.count("Hypertension") == 1


class TestHandleIncomingMessageEdgeCases:
    @pytest.mark.asyncio
    async def test_db_general_error_broadcasts_error(self):
        """Non-IntegrityError DB failure should also broadcast an error."""
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.commit.side_effect = Exception("DB connection lost")

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl):
            await mgr.handle_incoming_message(pid, {"role": "user", "content": "hi"})

        error_calls = [c.args[0] for c in ws.send_json.await_args_list if c.args[0].get("type") == "error"]
        assert error_calls

    @pytest.mark.asyncio
    async def test_followup_llm_failure_returns_fallback(self):
        """If _answer_followup raises, a fallback message is sent."""
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch.object(mgr, "_get_latest_recommendation", new_callable=AsyncMock) as mock_rec, \
             patch.object(mgr, "_answer_followup", new_callable=AsyncMock) as mock_ans:
            mock_rec.return_value = "Prior recommendation."
            mock_ans.side_effect = Exception("LLM down")

            await mgr.handle_incoming_message(
                pid,
                {"role": "user", "content": "Follow up?", "meta": {"followup": True}},
            )

        # Should still broadcast a followup message (fallback text)
        calls = [c.args[0] for c in ws.send_json.await_args_list]
        followup_calls = [c for c in calls if c.get("type") == "assistant_event"]
        assert followup_calls

    @pytest.mark.asyncio
    async def test_followup_no_prior_recommendation_skips_to_pipeline(self):
        """If no prior recommendation exists, followup falls through to orchestration."""
        mgr = ConnectionManager()
        mgr.set_orchestrator(MagicMock())
        pid = str(uuid.uuid4())
        ws = _make_ws()
        await mgr.connect(pid, ws)

        conv = _make_conv()
        mock_db, mock_sl = _db_mock(conv=conv)
        mock_db.get = AsyncMock(return_value=conv)

        orch_event = {"type": "assistant_event", "content": "Pipeline answer.", "meta": {}}

        with patch("app.ws_manager.AsyncSessionLocal", mock_sl), \
             patch.object(mgr, "_get_latest_recommendation", new_callable=AsyncMock) as mock_rec, \
             patch("app.ws_manager.process_user_turn", new_callable=AsyncMock) as mock_orch:
            mock_rec.return_value = None  # No prior recommendation
            mock_orch.return_value = orch_event

            await mgr.handle_incoming_message(
                pid,
                {"role": "user", "content": "question", "meta": {"followup": True}},
            )

        # Should have called orchestration since no prior rec
        mock_orch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_current_conversation_no_existing(self):
        """_close_current_conversation should be a no-op when no conversation exists."""
        mock_db, mock_sl = _db_mock(conv=None)

        mgr = ConnectionManager()
        with patch("app.ws_manager.AsyncSessionLocal", mock_sl):
            # Should not raise
            await mgr._close_current_conversation(uuid.uuid4())
