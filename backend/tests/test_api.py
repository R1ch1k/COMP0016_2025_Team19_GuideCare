"""
Integration tests for FastAPI HTTP endpoints.

Uses httpx AsyncClient with the FastAPI app and an in-memory SQLite DB
(overriding the real DB dependency via conftest).
"""

import pytest
from uuid import uuid4
from app.db.models import Diagnosis


# ── Patients API ─────────────────────────────────────────────────

class TestPatientsAPI:
    @pytest.mark.asyncio
    async def test_create_patient(self, async_client):
        resp = await async_client.post("/patients", json={
            "nhs_number": "API-001",
            "first_name": "Alice",
            "last_name": "Smith",
            "date_of_birth": "1990-05-10",
            "gender": "Female",
            "conditions": ["Asthma"],
            "allergies": ["Penicillin"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["nhs_number"] == "API-001"
        assert data["first_name"] == "Alice"
        assert data["age"] > 0
        assert "id" in data

    @pytest.mark.asyncio
    async def test_list_patients(self, async_client):
        # Create a patient first
        await async_client.post("/patients", json={
            "nhs_number": "API-LIST-001",
            "first_name": "Bob",
            "last_name": "Jones",
            "date_of_birth": "1985-01-01",
        })
        resp = await async_client.get("/patients")
        assert resp.status_code == 200
        patients = resp.json()
        assert isinstance(patients, list)
        assert len(patients) >= 1

    @pytest.mark.asyncio
    async def test_get_patient(self, async_client):
        # Create then fetch
        create_resp = await async_client.post("/patients", json={
            "nhs_number": "API-GET-001",
            "first_name": "Carol",
            "last_name": "Davis",
            "date_of_birth": "1975-12-25",
        })
        patient_id = create_resp.json()["id"]

        resp = await async_client.get(f"/patients/{patient_id}")
        assert resp.status_code == 200
        assert resp.json()["first_name"] == "Carol"

    @pytest.mark.asyncio
    async def test_get_patient_not_found(self, async_client):
        resp = await async_client.get(f"/patients/{uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_patient_context(self, async_client):
        create_resp = await async_client.post("/patients", json={
            "nhs_number": "API-CTX-001",
            "first_name": "Dave",
            "last_name": "Wilson",
            "date_of_birth": "1980-06-15",
            "gender": "Male",
            "conditions": ["Diabetes"],
        })
        patient_id = create_resp.json()["id"]

        resp = await async_client.get(f"/patients/{patient_id}/context")
        assert resp.status_code == 200
        ctx = resp.json()
        assert ctx["name"] == "Dave Wilson"
        assert "Diabetes" in ctx["conditions"]

    @pytest.mark.asyncio
    async def test_import_csv(self, async_client):
        csv_content = (
            "nhs_number,first_name,last_name,date_of_birth,gender,conditions,allergies\n"
            "IMP-001,Eve,Brown,1992-03-20,Female,Asthma,Penicillin\n"
            "IMP-002,Frank,Green,1988-07-04,Male,\"Diabetes, Hypertension\",\n"
        )
        resp = await async_client.post(
            "/patients/import",
            files={"file": ("patients.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["imported"] == 2
        assert data["total_rows"] == 2

    @pytest.mark.asyncio
    async def test_import_unsupported_format(self, async_client):
        resp = await async_client.post(
            "/patients/import",
            files={"file": ("data.txt", b"some text", "text/plain")},
        )
        assert resp.status_code == 400


# ── Conversations API ────────────────────────────────────────────

class TestConversationsAPI:
    @pytest.mark.asyncio
    async def test_create_conversation(self, async_client):
        # Create patient first
        p_resp = await async_client.post("/patients", json={
            "nhs_number": "CONV-API-001",
            "first_name": "Grace",
            "last_name": "Hall",
            "date_of_birth": "1995-01-01",
        })
        patient_id = p_resp.json()["id"]

        resp = await async_client.post("/conversations", json={
            "patient_id": patient_id,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "in_progress"
        assert data["messages"] == []

    @pytest.mark.asyncio
    async def test_get_conversation(self, async_client):
        p_resp = await async_client.post("/patients", json={
            "nhs_number": "CONV-API-002",
            "first_name": "Henry",
            "last_name": "King",
            "date_of_birth": "1990-01-01",
        })
        patient_id = p_resp.json()["id"]

        c_resp = await async_client.post("/conversations", json={
            "patient_id": patient_id,
        })
        conv_id = c_resp.json()["id"]

        resp = await async_client.get(f"/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, async_client):
        resp = await async_client.get(f"/conversations/{uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_conversation_nonexistent_patient(self, async_client):
        resp = await async_client.post("/conversations", json={
            "patient_id": str(uuid4()),
        })
        assert resp.status_code == 404


# ── Diagnoses API ────────────────────────────────────────────────

class TestDiagnosesAPI:
    @pytest.mark.asyncio
    async def test_list_diagnoses_empty(self, async_client):
        resp = await async_client.get("/diagnoses")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_export_json(self, async_client):
        resp = await async_client.get("/diagnoses/export?format=json")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_export_csv(self, async_client):
        resp = await async_client.get("/diagnoses/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_get_diagnosis_not_found(self, async_client):
        resp = await async_client.get(f"/diagnoses/{uuid4()}")
        assert resp.status_code == 404


# ── Patient PATCH + diagnoses endpoint ───────────────────────────

class TestPatientsAPIPatch:
    @pytest.mark.asyncio
    async def test_update_patient(self, async_client):
        create_resp = await async_client.post("/patients", json={
            "nhs_number": "PATCH-001",
            "first_name": "Pat",
            "last_name": "Update",
            "date_of_birth": "1990-01-01",
            "conditions": ["Asthma"],
        })
        assert create_resp.status_code == 201
        patient_id = create_resp.json()["id"]

        resp = await async_client.patch(f"/patients/{patient_id}", json={
            "conditions": ["Asthma", "Hypertension"],
            "allergies": ["Penicillin"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Hypertension" in data["conditions"]
        assert "Penicillin" in data["allergies"]
        assert data["first_name"] == "Pat"  # unchanged fields preserved

    @pytest.mark.asyncio
    async def test_update_patient_not_found(self, async_client):
        resp = await async_client.patch(f"/patients/{uuid4()}", json={
            "conditions": ["Asthma"],
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_patient_context_not_found(self, async_client):
        resp = await async_client.get(f"/patients/{uuid4()}/context")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_patient_diagnoses_empty(self, async_client):
        create_resp = await async_client.post("/patients", json={
            "nhs_number": "PDIAG-EMPTY-001",
            "first_name": "NoDiag",
            "last_name": "Patient",
            "date_of_birth": "1985-05-05",
        })
        patient_id = create_resp.json()["id"]
        resp = await async_client.get(f"/patients/{patient_id}/diagnoses")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_get_patient_diagnoses_not_found(self, async_client):
        resp = await async_client.get(f"/patients/{uuid4()}/diagnoses")
        assert resp.status_code == 404


# ── Diagnoses API with data ───────────────────────────────────────

class TestDiagnosesAPIWithData:
    async def _create_patient_with_diagnosis(self, async_client, db_session, nhs: str):
        """Helper: create a patient via API, insert a diagnosis via DB."""
        p_resp = await async_client.post("/patients", json={
            "nhs_number": nhs,
            "first_name": "Diag",
            "last_name": "Test",
            "date_of_birth": "1980-01-01",
        })
        assert p_resp.status_code == 201
        patient_id = p_resp.json()["id"]

        diag = Diagnosis(
            patient_id=patient_id,
            selected_guideline="NG136",
            final_recommendation="Offer antihypertensive treatment.",
            urgency="moderate",
            pathway_walked=["n1(yes)", "n5(action)"],
            extracted_variables={"age": 45, "clinic_bp": "160/100"},
        )
        db_session.add(diag)
        await db_session.commit()
        await db_session.refresh(diag)
        return patient_id, str(diag.id)

    @pytest.mark.asyncio
    async def test_list_diagnoses_with_data(self, async_client, db_session):
        await self._create_patient_with_diagnosis(async_client, db_session, "DLIST-001")
        resp = await async_client.get("/diagnoses")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["selected_guideline"] is not None
        assert "patient_name" in data[0]

    @pytest.mark.asyncio
    async def test_get_diagnosis_found(self, async_client, db_session):
        _, diag_id = await self._create_patient_with_diagnosis(
            async_client, db_session, "DGET-001"
        )
        resp = await async_client.get(f"/diagnoses/{diag_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_guideline"] == "NG136"
        assert data["final_recommendation"] == "Offer antihypertensive treatment."
        assert "patient_name" in data

    @pytest.mark.asyncio
    async def test_export_json_with_data(self, async_client, db_session):
        await self._create_patient_with_diagnosis(async_client, db_session, "DEXP-001")
        resp = await async_client.get("/diagnoses/export?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert "nhs_number" in data[0]
        assert "final_recommendation" in data[0]

    @pytest.mark.asyncio
    async def test_export_csv_with_data(self, async_client, db_session):
        await self._create_patient_with_diagnosis(async_client, db_session, "DCSV-001")
        resp = await async_client.get("/diagnoses/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")
        content = resp.text
        assert "diagnosis_id" in content  # header row
        assert "NG136" in content  # our test data
