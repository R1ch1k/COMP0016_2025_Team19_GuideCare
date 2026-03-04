"""
Integration tests for FastAPI HTTP endpoints.

Uses httpx AsyncClient with the FastAPI app and an in-memory SQLite DB
(overriding the real DB dependency via conftest).
"""

import pytest
from uuid import uuid4


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
