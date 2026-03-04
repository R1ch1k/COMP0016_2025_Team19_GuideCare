from app.db.session import AsyncSessionLocal
from app.db.models import Patient
from datetime import date
from sqlalchemy import select
import uuid
import logging

logger = logging.getLogger("seed")

SAMPLE_PATIENTS = [
    {
        "nhs_number": "NHS-1001",
        "first_name": "Alex",
        "last_name": "Morgan",
        "date_of_birth": date(1979, 6, 12),
        "gender": "male",
        "conditions": ["Type 2 diabetes", "Hypertension"],
        "medications": [{"name": "Metformin", "dose": "500mg"}],
        "allergies": ["Penicillin"],
        "recent_vitals": {"last_bp": "145/90", "last_bp_date": "2025-01-10"},
        "clinical_notes": [{"note": "A1C trending upward", "date": "2025-01-09"}]
    },
    {
        "nhs_number": "NHS-1002",
        "first_name": "Jordan",
        "last_name": "Lee",
        "date_of_birth": date(1962, 4, 5),
        "gender": "female",
        "conditions": ["Hypertension"],
        "medications": [{"name": "Amlodipine", "dose": "5mg"}],
        "allergies": [],
        "recent_vitals": {"last_bp": "160/100", "last_bp_date": "2025-01-08"},
        "clinical_notes": [{"note": "Medication review scheduled", "date": "2025-01-08"}]
    },
    {
        "nhs_number": "NHS-1003",
        "first_name": "Samantha",
        "last_name": "Chen",
        "date_of_birth": date(1992, 11, 21),
        "gender": "female",
        "conditions": ["Asthma"],
        "medications": [{"name": "Salbutamol", "dose": "100mcg"}],
        "allergies": ["Sulfa drugs"],
        "recent_vitals": {"last_bp": "120/78", "last_bp_date": "2025-01-02"},
        "clinical_notes": [{"note": "Follow-up in 6 months", "date": "2025-01-02"}]
    }
]

async def seed_if_empty():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Patient).limit(1))
        if result.scalars().first():
            logger.info("Seed: patients already present, skipping")
            return
        for p in SAMPLE_PATIENTS:
            age = calculate_age(p["date_of_birth"])
            patient = Patient(
                id=uuid.uuid4(),
                nhs_number=p["nhs_number"],
                first_name=p["first_name"],
                last_name=p["last_name"],
                date_of_birth=p["date_of_birth"],
                age=age,
                gender=p["gender"],
                conditions=p["conditions"],
                medications=p["medications"],
                allergies=p["allergies"],
                recent_vitals=p["recent_vitals"],
                clinical_notes=p["clinical_notes"]
            )
            session.add(patient)
        await session.commit()
        logger.info("Seed: inserted sample patients")

def calculate_age(dob):
    from datetime import date
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
