import pytest
from datetime import date
from app.schemas import PatientCreate
from app.crud import create_patient, get_patient

@pytest.mark.asyncio
async def test_create_and_get_patient(db_session):
    payload = PatientCreate(
        nhs_number="TEST123",
        first_name="Jane",
        last_name="Doe",
        date_of_birth=date(1990, 1, 1),
    )

    p = await create_patient(db_session, payload)
    fetched = await get_patient(db_session, p.id)

    assert fetched is not None
    assert fetched.id == p.id
    assert fetched.nhs_number == "TEST123"
