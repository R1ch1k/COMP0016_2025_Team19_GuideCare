import csv
import io
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Diagnosis, Patient
from app.db.session import get_async_session

router = APIRouter()


@router.get("")
async def list_diagnoses(db: AsyncSession = Depends(get_async_session)):
    """List all diagnoses with patient name."""
    result = await db.execute(
        select(Diagnosis, Patient.first_name, Patient.last_name)
        .join(Patient, Diagnosis.patient_id == Patient.id)
        .order_by(Diagnosis.diagnosed_at.desc())
    )
    rows = result.all()
    out = []
    for diag, first, last in rows:
        out.append({
            "id": str(diag.id),
            "patient_id": str(diag.patient_id),
            "patient_name": f"{first} {last}",
            "conversation_id": str(diag.conversation_id) if diag.conversation_id else None,
            "selected_guideline": diag.selected_guideline,
            "extracted_variables": diag.extracted_variables,
            "pathway_walked": diag.pathway_walked,
            "final_recommendation": diag.final_recommendation,
            "urgency": diag.urgency,
            "status": diag.status,
            "diagnosed_at": diag.diagnosed_at.isoformat() if diag.diagnosed_at else None,
        })
    return out


@router.get("/export")
async def export_diagnoses(
    format: str = "json",
    db: AsyncSession = Depends(get_async_session),
):
    """Export all diagnoses as JSON or CSV."""
    result = await db.execute(
        select(Diagnosis, Patient.first_name, Patient.last_name, Patient.nhs_number)
        .join(Patient, Diagnosis.patient_id == Patient.id)
        .order_by(Diagnosis.diagnosed_at.desc())
    )
    rows = result.all()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "diagnosis_id", "patient_name", "nhs_number", "guideline",
            "urgency", "recommendation", "pathway", "diagnosed_at",
        ])
        for diag, first, last, nhs in rows:
            writer.writerow([
                str(diag.id),
                f"{first} {last}",
                nhs,
                diag.selected_guideline or "",
                diag.urgency or "",
                diag.final_recommendation or "",
                " -> ".join(diag.pathway_walked or []),
                diag.diagnosed_at.isoformat() if diag.diagnosed_at else "",
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=diagnoses.csv"},
        )

    # Default: JSON
    out = []
    for diag, first, last, nhs in rows:
        out.append({
            "id": str(diag.id),
            "patient_name": f"{first} {last}",
            "nhs_number": nhs,
            "selected_guideline": diag.selected_guideline,
            "extracted_variables": diag.extracted_variables,
            "pathway_walked": diag.pathway_walked,
            "final_recommendation": diag.final_recommendation,
            "urgency": diag.urgency,
            "diagnosed_at": diag.diagnosed_at.isoformat() if diag.diagnosed_at else None,
        })
    return out


@router.get("/{diagnosis_id}")
async def get_diagnosis(diagnosis_id: UUID, db: AsyncSession = Depends(get_async_session)):
    """Get a single diagnosis by ID."""
    diag = await db.get(Diagnosis, diagnosis_id)
    if not diag:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    # Fetch patient name
    patient = await db.get(Patient, diag.patient_id)
    patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Unknown"

    return {
        "id": str(diag.id),
        "patient_id": str(diag.patient_id),
        "patient_name": patient_name,
        "conversation_id": str(diag.conversation_id) if diag.conversation_id else None,
        "selected_guideline": diag.selected_guideline,
        "extracted_variables": diag.extracted_variables,
        "pathway_walked": diag.pathway_walked,
        "final_recommendation": diag.final_recommendation,
        "urgency": diag.urgency,
        "status": diag.status,
        "diagnosed_at": diag.diagnosed_at.isoformat() if diag.diagnosed_at else None,
    }
