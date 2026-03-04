"""
Unit tests for CRUD operations — uses in-memory SQLite via conftest fixtures.
"""

import pytest
from datetime import date
from uuid import uuid4

from app.crud import (
    compute_age,
    create_patient,
    get_patient,
    list_patients,
    start_conversation,
    get_conversation,
    append_message_to_conversation,
)
from app.schemas import PatientCreate


# ── compute_age ───────────────────────────────────────────────────

class TestComputeAge:
    @pytest.mark.asyncio
    async def test_age_calculation(self):
        today = date.today()
        dob = date(today.year - 30, today.month, today.day)
        assert await compute_age(dob) == 30

    @pytest.mark.asyncio
    async def test_birthday_not_yet(self):
        today = date.today()
        # Birthday hasn't happened yet this year
        if today.month < 12:
            dob = date(today.year - 25, today.month + 1, 1)
            assert await compute_age(dob) == 24
        else:
            dob = date(today.year - 25, 1, 1)
            assert await compute_age(dob) == 25


# ── create_patient / get_patient ──────────────────────────────────

class TestPatientCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get(self, db_session):
        payload = PatientCreate(
            nhs_number="TEST-001",
            first_name="Test",
            last_name="Patient",
            date_of_birth=date(1990, 6, 15),
            gender="Male",
            conditions=["Asthma"],
            allergies=["Penicillin"],
        )
        patient = await create_patient(db_session, payload)
        assert patient.id is not None
        assert patient.first_name == "Test"
        assert patient.nhs_number == "TEST-001"
        assert patient.age > 0

        fetched = await get_patient(db_session, patient.id)
        assert fetched is not None
        assert fetched.nhs_number == "TEST-001"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, db_session):
        result = await get_patient(db_session, str(uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_list_patients(self, db_session):
        # Create two patients
        for i in range(2):
            await create_patient(
                db_session,
                PatientCreate(
                    nhs_number=f"LIST-{i}",
                    first_name=f"Patient{i}",
                    last_name="Test",
                    date_of_birth=date(1985, 1, 1),
                ),
            )
        patients = await list_patients(db_session)
        # At least 2 (may include patients from other tests in same session)
        assert len(patients) >= 2

    @pytest.mark.asyncio
    async def test_list_patients_no_limit(self, db_session):
        patients = await list_patients(db_session)
        assert len(patients) >= 1


# ── conversations ─────────────────────────────────────────────────

class TestConversationCRUD:
    @pytest.mark.asyncio
    async def test_start_conversation(self, db_session):
        patient = await create_patient(
            db_session,
            PatientCreate(
                nhs_number="CONV-001",
                first_name="Conv",
                last_name="Patient",
                date_of_birth=date(1980, 3, 20),
            ),
        )
        conv = await start_conversation(db_session, patient.id)
        assert conv.id is not None
        assert conv.status == "in_progress"
        assert conv.messages == []

    @pytest.mark.asyncio
    async def test_get_conversation(self, db_session):
        patient = await create_patient(
            db_session,
            PatientCreate(
                nhs_number="CONV-002",
                first_name="Conv2",
                last_name="Patient",
                date_of_birth=date(1980, 3, 20),
            ),
        )
        conv = await start_conversation(db_session, patient.id)
        fetched = await get_conversation(db_session, conv.id)
        assert fetched is not None
        assert fetched.status == "in_progress"

    @pytest.mark.asyncio
    async def test_get_conversation_nonexistent(self, db_session):
        result = await get_conversation(db_session, str(uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_append_message(self, db_session):
        patient = await create_patient(
            db_session,
            PatientCreate(
                nhs_number="CONV-003",
                first_name="Msg",
                last_name="Patient",
                date_of_birth=date(1990, 1, 1),
            ),
        )
        conv = await start_conversation(db_session, patient.id)
        msg = {"role": "user", "content": "I have a sore throat"}

        updated = await append_message_to_conversation(db_session, conv, msg)
        assert len(updated.messages) == 1
        assert updated.messages[0]["content"] == "I have a sore throat"

        # Append another
        msg2 = {"role": "assistant", "content": "What is the FeverPAIN score?"}
        updated = await append_message_to_conversation(db_session, updated, msg2)
        assert len(updated.messages) == 2
