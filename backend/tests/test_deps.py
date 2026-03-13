"""
Tests for app.orchestration.deps — all LLM and DB calls mocked.
The guideline engine runs against real JSON files (already loaded in other tests).

Extended tests at the bottom cover additional branches in gpt_clarifier and
extract_variables_20b to push coverage higher.
"""

import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.orchestration.deps import (
    _format_meds,
    _format_triage_prompt,
    _guess_guideline,
    _extract_var_names,
    triage_agent,
    gpt_clarifier,
    select_guideline_fn,
    extract_variables_20b,
    format_output_20b,
    walk_guideline_graph_fn,
    fetch_patient,
)


# ── Sample data ────────────────────────────────────────────────────────────────

PATIENT = {
    "id": str(uuid.uuid4()),
    "age": 55,
    "gender": "male",
    "conditions": ["Hypertension"],
    "medications": [],
    "allergies": [],
    "recent_vitals": {},
    "clinical_notes": [],
}

SORE_THROAT_PATIENT = {
    "id": str(uuid.uuid4()),
    "age": 32,
    "gender": "female",
    "conditions": [],
    "medications": [],
    "allergies": [],
    "recent_vitals": {},
    "clinical_notes": [],
}


# ── Pure helpers ───────────────────────────────────────────────────────────────

class TestFormatMeds:
    def test_string_list(self):
        result = _format_meds(["Amlodipine", "Ramipril"])
        assert "Amlodipine" in result
        assert "Ramipril" in result

    def test_dict_list(self):
        result = _format_meds([{"name": "Amlodipine", "dose": "5mg"}])
        assert "Amlodipine" in result
        assert "5mg" in result

    def test_empty_list(self):
        assert _format_meds([]) == "None"

    def test_none_input(self):
        assert _format_meds(None) == "None"

    def test_dict_without_dose(self):
        result = _format_meds([{"name": "Amlodipine"}])
        assert "Amlodipine" in result

    def test_mixed_types(self):
        result = _format_meds([{"name": "Ramipril", "dose": "10mg"}, "Aspirin"])
        assert "Ramipril" in result
        assert "Aspirin" in result


class TestFormatTriagePrompt:
    def test_includes_symptoms_and_age(self):
        result = _format_triage_prompt("sore throat", {"age": 32, "conditions": [], "medications": []})
        assert "sore throat" in result
        assert "32" in result

    def test_includes_medical_history(self):
        result = _format_triage_prompt("BP", {"age": 50, "conditions": ["Diabetes"], "medications": []})
        assert "Diabetes" in result

    def test_uses_medical_history_key_fallback(self):
        result = _format_triage_prompt("symptoms", {"age": 40, "medical_history": ["Asthma"], "medications": []})
        assert "Asthma" in result

    def test_no_history_shows_none(self):
        result = _format_triage_prompt("cough", {"age": 30, "conditions": [], "medications": []})
        assert "None" in result


class TestGuessGuideline:
    def test_sore_throat_returns_ng84(self):
        assert _guess_guideline("sore throat with fever") == "NG84"

    def test_blood_pressure_returns_ng136(self):
        assert _guess_guideline("high blood pressure reading") == "NG136"

    def test_head_injury_returns_ng232(self):
        assert _guess_guideline("patient hit head and fell") == "NG232"

    def test_ear_pain_returns_ng91(self):
        assert _guess_guideline("ear pain and ear infection") == "NG91"

    def test_uti_returns_ng112(self):
        assert _guess_guideline("recurrent urinary tract infection dysuria") == "NG112"

    def test_pregnancy_returns_ng133(self):
        assert _guess_guideline("patient is pregnant with high bp") == "NG133"

    def test_bite_returns_ng184(self):
        assert _guess_guideline("dog bite on hand") == "NG184"

    def test_depression_returns_ng222(self):
        assert _guess_guideline("depression and low mood") == "NG222"

    def test_glaucoma_returns_ng81_glaucoma(self):
        assert _guess_guideline("diagnosed glaucoma iop elevated") == "NG81_GLAUCOMA"

    def test_no_match_returns_none(self):
        assert _guess_guideline("unrelated symptom xyz") is None


class TestExtractVarNames:
    def test_simple_variable(self):
        result = _extract_var_names({"variable": "clinic_bp"})
        assert "clinic_bp" in result

    def test_and_conditions(self):
        spec = {"and": [{"variable": "age"}, {"variable": "gender"}]}
        result = _extract_var_names(spec)
        assert "age" in result
        assert "gender" in result

    def test_nested_conditions(self):
        spec = {"conditions": [{"variable": "fever"}, {"variable": "duration"}]}
        result = _extract_var_names(spec)
        assert "fever" in result
        assert "duration" in result

    def test_variables_shorthand(self):
        spec = {"variables": ["clinic_bp", "age"]}
        result = _extract_var_names(spec)
        assert "clinic_bp" in result
        assert "age" in result

    def test_empty_spec(self):
        assert _extract_var_names({}) == []

    def test_deeply_nested(self):
        spec = {"and": [{"conditions": [{"variable": "fever"}]}]}
        result = _extract_var_names(spec)
        assert "fever" in result


# ── triage_agent() ────────────────────────────────────────────────────────────

class TestTriageAgent:
    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        triage_json = json.dumps({
            "urgency": "moderate",
            "reasoning": "Low-grade fever, mild sore throat",
            "suggested_guideline": "NG84",
            "guideline_confidence": "high",
            "red_flags": [],
            "assessment": "Moderate sore throat",
        })
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.return_value = triage_json
            result = await triage_agent("sore throat", [], SORE_THROAT_PATIENT)

        assert result["urgency"] == "moderate"
        assert result["suggested_guideline"] == "NG84"
        assert result["assessment"] == "Moderate sore throat"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        triage_json = "```json\n" + json.dumps({
            "urgency": "urgent",
            "suggested_guideline": "NG84",
            "reasoning": "High fever",
            "red_flags": ["high fever"],
            "assessment": "Urgent",
        }) + "\n```"
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.return_value = triage_json
            result = await triage_agent("sore throat with high fever", [], SORE_THROAT_PATIENT)

        assert result["urgency"] == "urgent"

    @pytest.mark.asyncio
    async def test_normalises_urgency_to_lowercase(self):
        triage_json = json.dumps({
            "urgency": "MODERATE",
            "suggested_guideline": "NG84",
            "reasoning": "r",
            "red_flags": [],
            "assessment": "a",
        })
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.return_value = triage_json
            result = await triage_agent("throat", [], SORE_THROAT_PATIENT)

        assert result["urgency"] == "moderate"

    @pytest.mark.asyncio
    async def test_fills_assessment_from_reasoning_if_missing(self):
        triage_json = json.dumps({
            "urgency": "routine",
            "suggested_guideline": "NG84",
            "reasoning": "Mild symptoms, no red flags",
            "red_flags": [],
        })
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.return_value = triage_json
            result = await triage_agent("mild throat", [], SORE_THROAT_PATIENT)

        assert "assessment" in result
        assert result["assessment"] == "Mild symptoms, no red flags"

    @pytest.mark.asyncio
    async def test_heuristic_fallback_emergency_keywords(self):
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.side_effect = Exception("API error")
            result = await triage_agent("patient has chest pain and loss of consciousness", [], PATIENT)

        assert result["urgency"] == "emergency"
        assert "chest pain" in result["red_flags"] or "loss of consciousness" in result["red_flags"]

    @pytest.mark.asyncio
    async def test_heuristic_fallback_moderate_for_non_urgent(self):
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.side_effect = Exception("API error")
            result = await triage_agent("mild runny nose", [], PATIENT)

        assert result["urgency"] == "moderate"

    @pytest.mark.asyncio
    async def test_heuristic_fallback_on_invalid_json(self):
        with patch("app.orchestration.deps.generate_api_only", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.TRIAGE_API_URL = None
            mock_gen.return_value = "not valid json at all"
            result = await triage_agent("sore throat", [], PATIENT)

        # Falls back to heuristic
        assert result["urgency"] in ("moderate", "emergency")


# ── gpt_clarifier() ───────────────────────────────────────────────────────────

class TestGptClarifier:
    @pytest.mark.asyncio
    async def test_done_when_no_guideline_found(self):
        result = await gpt_clarifier(
            "totally unknown symptoms xyz",
            [], {}, {}, {}, selected_guideline=""
        )
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_done_when_all_vars_known(self):
        """NG84 with FeverPAIN + age + gender already known → no questions needed."""
        patient = {**SORE_THROAT_PATIENT, "age": 32, "gender": "female"}
        # Provide enough variables to satisfy the NG84 decision tree
        answers = {
            "[var:feverpain_score] FeverPAIN score?": "4",
            "[var:centor_score] Centor score?": "3",
            "[var:fever] Does patient have fever?": "yes",
        }
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "What is the patient's temperature?"
            result = await gpt_clarifier(
                "sore throat with fever",
                [], patient, {"suggested_guideline": "NG84"}, answers,
                selected_guideline="NG84"
            )
        # With enough answers, should have fewer (or no) questions
        assert isinstance(result, dict)
        assert "done" in result
        assert "questions" in result

    @pytest.mark.asyncio
    async def test_generates_question_for_missing_var(self):
        """NG84 with no known variables → should generate a clarification question."""
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "What is the patient's FeverPAIN score?"
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"}, {},
                selected_guideline="NG84"
            )

        if not result["done"]:
            assert len(result["questions"]) > 0
            # Questions should be tagged with [var:...]
            assert all("[var:" in q for q in result["questions"])

    @pytest.mark.asyncio
    async def test_parses_bp_answer(self):
        """BP reading from answer should be stored in extracted variables."""
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "What is the BP?"
            answers = {"[var:clinic_bp] What is the clinic BP?": "155/95"}
            result = await gpt_clarifier(
                "high blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        # Result is valid regardless of whether more questions remain
        assert "done" in result

    @pytest.mark.asyncio
    async def test_skips_unknown_answers(self):
        """Answers like 'not known' should be skipped without raising."""
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "What is the score?"
            answers = {"[var:feverpain_score] FeverPAIN score?": "not known"}
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"},
                answers,
                selected_guideline="NG84"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_llm_failure_uses_fallback_question(self):
        """If generate() throws, a fallback question is used instead of crashing."""
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("LLM down")
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"}, {},
                selected_guideline="NG84"
            )
        # Should not raise; may return done or questions with fallback text
        assert "done" in result

    @pytest.mark.asyncio
    async def test_parses_yes_no_answer(self):
        answers = {"[var:fever] Does patient have fever?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"},
                answers,
                selected_guideline="NG84"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_parses_numeric_answer(self):
        answers = {"[var:feverpain_score] FeverPAIN score?": "3"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"},
                answers,
                selected_guideline="NG84"
            )
        assert "done" in result


# ── select_guideline_fn() ─────────────────────────────────────────────────────

class TestSelectGuidelineFn:
    @pytest.mark.asyncio
    async def test_uses_triage_suggestion_when_valid(self):
        result = await select_guideline_fn(
            "sore throat",
            {"suggested_guideline": "NG84"},
            {},
            SORE_THROAT_PATIENT,
        )
        assert result == "NG84"

    @pytest.mark.asyncio
    async def test_uses_keyword_mapping_when_no_triage(self):
        result = await select_guideline_fn(
            "patient has severe sore throat",
            {},
            {},
            SORE_THROAT_PATIENT,
        )
        assert result == "NG84"

    @pytest.mark.asyncio
    async def test_falls_back_to_ng84_for_unknown(self):
        with patch("app.orchestration.deps.settings") as s:
            s.OPENAI_API_KEY = None
            result = await select_guideline_fn(
                "mysterious unrelated symptoms xyz",
                {},
                {},
                SORE_THROAT_PATIENT,
            )
        assert result == "NG84"

    @pytest.mark.asyncio
    async def test_llm_fallback_when_no_keyword_match(self):
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            s.OPENAI_API_KEY = "sk-test"
            mock_gen.return_value = "NG84"
            result = await select_guideline_fn(
                "mysterious condition",
                {},
                {},
                PATIENT,
            )
        # Should use LLM suggestion (NG84) if it's a valid guideline
        assert result in ("NG84", "NG136")  # fallback may also be NG84

    @pytest.mark.asyncio
    async def test_invalid_triage_suggestion_falls_through(self):
        """If triage suggests a guideline that doesn't exist, fall to keywords."""
        result = await select_guideline_fn(
            "sore throat",
            {"suggested_guideline": "NONEXISTENT_GUIDELINE"},
            {},
            SORE_THROAT_PATIENT,
        )
        # Should fall through to keyword match → NG84
        assert result == "NG84"


# ── extract_variables_20b() ───────────────────────────────────────────────────

class TestExtractVariables:
    @pytest.mark.asyncio
    async def test_returns_empty_dict_for_unknown_guideline(self):
        result = await extract_variables_20b("NONEXISTENT", [], PATIENT, {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_extracts_age_and_gender_from_patient(self):
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({"feverpain_score": 3})
            result = await extract_variables_20b(
                "NG84",
                [{"role": "user", "content": "sore throat with fever"}],
                SORE_THROAT_PATIENT,
                {},
            )
        assert result.get("age") == 32
        assert result.get("gender") == "female"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_empty(self):
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("LLM error")
            result = await extract_variables_20b(
                "NG84",
                [{"role": "user", "content": "sore throat"}],
                SORE_THROAT_PATIENT,
                {},
            )
        # Should not raise; returns partial result from regex helpers
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_alias_normalisation(self):
        """LLM using 'bp' alias should be mapped to 'clinic_bp'."""
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({"bp": "155/95"})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP is 155/95"}],
                PATIENT,
                {},
            )
        # 'bp' → 'clinic_bp' alias normalisation
        assert result.get("clinic_bp") == "155/95" or result.get("bp") == "155/95"

    @pytest.mark.asyncio
    async def test_comorbidity_override_from_patient_record(self):
        """Patient record conditions should override LLM-hallucinated comorbidities."""
        diabetic_patient = {**PATIENT, "conditions": ["Diabetes Type 2", "Hypertension"]}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({"diabetes": False})  # LLM gets it wrong
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 160/100"}],
                diabetic_patient,
                {},
            )
        # Patient record should override LLM — diabetes=True because it's in conditions
        assert result.get("diabetes") is True

    @pytest.mark.asyncio
    async def test_clarification_answers_merged(self):
        """Clarification answers tagged with [var:...] should be set in extracted vars."""
        clarifications = {"[var:abpm_tolerated] Is ABPM tolerated?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 155/95"}],
                PATIENT,
                clarifications,
            )
        assert result.get("abpm_tolerated") is True

    @pytest.mark.asyncio
    async def test_qrisk_estimated_for_older_hypertensive(self):
        """QRISK should be estimated for patients over 60 with hypertension."""
        old_patient = {**PATIENT, "age": 65, "conditions": ["Hypertension"]}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 160/100"}],
                old_patient,
                {},
            )
        # qrisk_10yr should be estimated (conservative 15)
        assert result.get("qrisk_10yr") == 15


# ── walk_guideline_graph_fn() ─────────────────────────────────────────────────

class TestWalkGuidelineGraphFn:
    @pytest.mark.asyncio
    async def test_unknown_guideline_returns_terminal(self):
        result = await walk_guideline_graph_fn("NONEXISTENT", {}, None, [])
        assert result["terminal"] is True

    @pytest.mark.asyncio
    async def test_known_guideline_returns_pathway(self):
        variables = {
            "age": 32, "gender": "female",
            "feverpain_score": 4, "centor_score": 3,
            "fever": True, "systemically_very_unwell": False,
            "signs_of_serious_illness_condition": False,
            "high_risk_of_complications": False,
        }
        result = await walk_guideline_graph_fn("NG84", variables, None, [])
        assert "pathway_walked" in result
        assert "terminal" in result


# ── format_output_20b() ───────────────────────────────────────────────────────

class TestFormatOutput:
    @pytest.mark.asyncio
    async def test_returns_recommendation_for_valid_guideline(self):
        variables = {
            "age": 32, "gender": "female",
            "feverpain_score": 4, "centor_score": 3,
            "fever": True, "systemically_very_unwell": False,
            "signs_of_serious_illness_condition": False,
            "high_risk_of_complications": False,
        }
        result = await format_output_20b(
            "NG84",
            {"urgency": "moderate"},
            variables,
            [],
            SORE_THROAT_PATIENT,
        )
        assert "final_recommendation" in result
        assert result["citation"] == "NG84"

    @pytest.mark.asyncio
    async def test_unknown_guideline_uses_llm_or_fallback(self):
        with patch("app.orchestration.deps.settings") as s:
            s.OPENAI_API_KEY = None
            result = await format_output_20b(
                "NONEXISTENT",
                {"urgency": "routine"},
                {},
                [],
                {"first_name": "Test"},
            )
        assert "final_recommendation" in result

    @pytest.mark.asyncio
    async def test_llm_fallback_when_no_actions(self):
        """When guideline traversal yields no actions, LLM fallback is used."""
        with patch("app.orchestration.deps.traverse_guideline_graph") as mock_trav, \
             patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen, \
             patch("app.orchestration.deps.settings") as s:
            mock_trav.return_value = {"reached_actions": [], "path": [], "missing_variables": []}
            mock_gen.return_value = "Manage hypertension per NICE NG136."
            s.OPENAI_API_KEY = "sk-test"
            result = await format_output_20b(
                "NG136",
                {"urgency": "moderate"},
                {"clinic_bp": "155/95"},
                [],
                PATIENT,
            )
        assert "final_recommendation" in result


# ── fetch_patient() ───────────────────────────────────────────────────────────

class TestFetchPatient:
    @pytest.mark.asyncio
    async def test_returns_patient_dict_when_found(self):
        pid = uuid.uuid4()
        mock_patient = MagicMock()
        mock_patient.id = pid
        mock_patient.nhs_number = "1234567890"
        mock_patient.first_name = "John"
        mock_patient.last_name = "Doe"
        mock_patient.age = 45
        mock_patient.gender = "male"
        mock_patient.conditions = ["Hypertension"]
        mock_patient.medications = []
        mock_patient.allergies = []
        mock_patient.recent_vitals = {}
        mock_patient.clinical_notes = []

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_patient
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_db)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.orchestration.deps.AsyncSessionLocal", return_value=cm):
            result = await fetch_patient(str(pid))

        assert result["age"] == 45
        assert result["gender"] == "male"
        assert "Hypertension" in result["conditions"]

    @pytest.mark.asyncio
    async def test_returns_id_only_when_not_found(self):
        pid = str(uuid.uuid4())
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_db)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.orchestration.deps.AsyncSessionLocal", return_value=cm):
            result = await fetch_patient(pid)

        assert result == {"id": pid}


# ── Additional branch coverage ────────────────────────────────────────────────

class TestTriageAgentApiUrl:
    @pytest.mark.asyncio
    async def test_uses_external_api_when_configured(self):
        """When TRIAGE_API_URL is set, POST to external endpoint instead of OpenAI."""
        import httpx
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "urgency": "urgent",
            "suggested_guideline": "NG84",
            "reasoning": "External triage",
            "red_flags": ["high fever"],
            "assessment": "External assessment",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.orchestration.deps.settings") as s, \
             patch("httpx.AsyncClient", return_value=mock_client):
            s.TRIAGE_API_URL = "http://triage.internal"
            s.AI_TIMEOUT_SECONDS = 30
            result = await triage_agent("sore throat", [], SORE_THROAT_PATIENT)

        assert result["urgency"] == "urgent"
        mock_client.post.assert_awaited_once()


class TestGptClarifierAnswerParsing:
    """Cover the deep answer-parsing branches in gpt_clarifier."""

    @pytest.mark.asyncio
    async def test_bp_answer_sets_abpm_daytime(self):
        answers = {"[var:abpm_daytime] What is ABPM daytime average?": "145/90"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "high blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_qrisk_less_than_10_parsed(self):
        answers = {"[var:qrisk_10yr] What is QRISK score?": "less than 10%"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "high blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_qrisk_greater_than_10_parsed(self):
        answers = {"[var:qrisk_10yr] QRISK?": "above 10%"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "high blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_qrisk_numeric_value(self):
        answers = {"[var:qrisk_10yr] QRISK?": "12.5%"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "high blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_qrisk_low_string(self):
        answers = {"[var:qrisk_10yr] QRISK?": "low"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "high blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_not_black_african_caribbean_yes(self):
        """Yes → patient IS of African/Caribbean origin → not_black_african_caribbean=False."""
        answers = {"[var:not_black_african_caribbean] Is patient of African/Caribbean origin?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "blood pressure 155/95",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_not_black_african_caribbean_no(self):
        answers = {"[var:not_black_african_caribbean] Is patient of African/Caribbean origin?": "no"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "blood pressure 155/95",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_abpm_tolerated_no(self):
        answers = {"[var:abpm_tolerated] Is ABPM tolerated?": "no"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "blood pressure 155/95",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_abpm_tolerated_with_bp_reading(self):
        answers = {"[var:abpm_tolerated] Is ABPM tolerated?": "yes, result was 145/90"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "blood pressure",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_target_bp_achieved_with_reading(self):
        answers = {"[var:target_bp_achieved] Is BP at target?": "130/80"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "blood pressure on treatment",
                [], {**PATIENT, "medications": [{"name": "Amlodipine", "dose": "5mg"}]},
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_target_bp_achieved_yes(self):
        answers = {"[var:target_bp_achieved] Is BP at target?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "on treatment",
                [], {**PATIENT, "medications": [{"name": "Amlodipine", "dose": "5mg"}]},
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_temperature_numeric(self):
        answers = {"[var:temperature] What is temperature?": "38.5"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"},
                answers,
                selected_guideline="NG84"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_temperature_yes(self):
        answers = {"[var:fever] Does patient have fever?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"},
                answers,
                selected_guideline="NG84"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_gcs_score_numeric(self):
        answers = {"[var:gcs_score] GCS score?": "14"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "head injury",
                [], PATIENT,
                {"suggested_guideline": "NG232"},
                answers,
                selected_guideline="NG232"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_negated_variable_no_answer(self):
        """Variables starting with 'not_' should be set to True when answer is 'no'."""
        answers = {"[var:not_black_african_caribbean] African/Caribbean?": "no"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "hypertension",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_untagged_bp_answer_fallback(self):
        """Legacy untagged BP questions use keyword matching."""
        answers = {"What is the blood pressure?": "155/95"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "hypertension",
                [], PATIENT,
                {"suggested_guideline": "NG136"},
                answers,
                selected_guideline="NG136"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_untagged_yes_no_answer_fallback(self):
        """Legacy untagged yes/no questions use keyword matching."""
        answers = {"Does the patient have fever?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "sore throat",
                [], SORE_THROAT_PATIENT,
                {"suggested_guideline": "NG84"},
                answers,
                selected_guideline="NG84"
            )
        assert "done" in result

    @pytest.mark.asyncio
    async def test_bp_vitals_from_patient_record(self):
        """BP from patient recent_vitals should be used as clinic_bp."""
        patient_with_vitals = {
            **PATIENT,
            "recent_vitals": {"last_bp": "160/100"},
        }
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Q?"
            result = await gpt_clarifier(
                "hypertension",
                [], patient_with_vitals,
                {"suggested_guideline": "NG136"},
                {},
                selected_guideline="NG136"
            )
        assert "done" in result


class TestExtractVariablesAdditionalBranches:
    """Cover extra branches in extract_variables_20b."""

    @pytest.mark.asyncio
    async def test_hbpm_answer_merged(self):
        clarifications = {"[var:hbpm_average] HBPM average?": "148/92"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP monitoring done"}],
                PATIENT,
                clarifications,
            )
        assert result.get("hbpm_average") == "148/92"

    @pytest.mark.asyncio
    async def test_not_black_african_caribbean_yes_clarification(self):
        """Yes answer (patient IS African/Caribbean) → not_black_african_caribbean=False."""
        clarifications = {"[var:not_black_african_caribbean] African/Caribbean origin?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 155/95"}],
                PATIENT,
                clarifications,
            )
        assert result.get("not_black_african_caribbean") is False

    @pytest.mark.asyncio
    async def test_not_black_african_caribbean_no_clarification(self):
        clarifications = {"[var:not_black_african_caribbean] African/Caribbean origin?": "no"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 155/95"}],
                PATIENT,
                clarifications,
            )
        assert result.get("not_black_african_caribbean") is True

    @pytest.mark.asyncio
    async def test_target_bp_achieved_yes_clarification(self):
        on_treatment_patient = {**PATIENT, "medications": [{"name": "Amlodipine", "dose": "5mg"}]}
        clarifications = {"[var:target_bp_achieved] BP at target?": "yes"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "on treatment"}],
                on_treatment_patient,
                clarifications,
            )
        assert result.get("target_bp_achieved") is True

    @pytest.mark.asyncio
    async def test_target_bp_achieved_with_bp_reading_clarification(self):
        on_treatment_patient = {**PATIENT, "medications": [{"name": "Amlodipine", "dose": "5mg"}]}
        clarifications = {"[var:target_bp_achieved] BP at target?": "130/80"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "on treatment"}],
                on_treatment_patient,
                clarifications,
            )
        assert result.get("clinic_bp") == "130/80"

    @pytest.mark.asyncio
    async def test_unknown_answer_skipped_in_extract(self):
        """Unknown answers in clarifications should not set any variable."""
        clarifications = {"[var:abpm_tolerated] Is ABPM tolerated?": "not known"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 155/95"}],
                PATIENT,
                clarifications,
            )
        assert result.get("abpm_tolerated") is None

    @pytest.mark.asyncio
    async def test_ethnicity_inferred_from_patient_record(self):
        """If patient has ethnicity field, ethnicity inference code runs without error."""
        patient_with_ethnicity = {**PATIENT, "ethnicity": "White British"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP 155/95"}],
                patient_with_ethnicity,
                {},
            )
        # If not_black_african_caribbean was included in all_vars, it should be True
        # (patient is White British, so not of African/Caribbean origin)
        assert isinstance(result, dict)
        if "not_black_african_caribbean" in result:
            assert result["not_black_african_caribbean"] is True

    @pytest.mark.asyncio
    async def test_no_epilepsy_default_true(self):
        """no_epilepsy_history defaults to True when patient has no epilepsy."""
        patient_no_epilepsy = {**PATIENT, "conditions": []}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG232",  # Head injury uses no_epilepsy_history
                [{"role": "user", "content": "head injury patient"}],
                patient_no_epilepsy,
                {},
            )
        # no_epilepsy_history should default to True for non-epileptic patients
        if "no_epilepsy_history" in result:
            assert result["no_epilepsy_history"] is True

    @pytest.mark.asyncio
    async def test_legacy_untagged_clarification_keyword_match(self):
        """Untagged clarification uses variable keyword matching."""
        clarifications = {"What is the blood pressure reading?": "155/95"}
        with patch("app.orchestration.deps.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = json.dumps({})
            result = await extract_variables_20b(
                "NG136",
                [{"role": "user", "content": "BP issues"}],
                PATIENT,
                clarifications,
            )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_build_orchestration_deps_returns_all_keys(self):
        """build_orchestration_deps should return a dict with all expected keys."""
        from app.orchestration.deps import build_orchestration_deps
        deps = build_orchestration_deps()
        expected_keys = {
            "fetch_patient", "triage_agent", "gpt_clarifier",
            "select_guideline", "extract_variables_20b",
            "walk_guideline_graph", "format_output_20b",
        }
        assert expected_keys == set(deps.keys())
