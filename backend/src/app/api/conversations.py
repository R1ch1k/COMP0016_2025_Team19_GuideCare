from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas import ConversationCreate, ConversationOut
from app import crud

router = APIRouter()


@router.post("", response_model=ConversationOut, status_code=201)
async def start_conversation(payload: ConversationCreate, db: AsyncSession = Depends(get_async_session)):
    patient = await crud.get_patient(db, payload.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return await crud.start_conversation(db, payload.patient_id, payload.selected_guideline)


@router.get("/{conv_id}", response_model=ConversationOut)
async def get_conversation(conv_id: UUID, db: AsyncSession = Depends(get_async_session)):
    conv = await crud.get_conversation(db, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv
