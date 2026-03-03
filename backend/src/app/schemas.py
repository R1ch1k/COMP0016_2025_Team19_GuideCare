from datetime import date, datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import ConfigDict  # type: ignore

    class ORMBase(BaseModel):
        model_config = ConfigDict(from_attributes=True)

except Exception:
    # Pydantic v1 fallback
    class ORMBase(BaseModel):
        class Config:
            orm_mode = True


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None


class PatientCreate(BaseModel):
    nhs_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    recent_vitals: dict = Field(default_factory=dict)
    clinical_notes: List[dict] = Field(default_factory=list)


class PatientUpdate(BaseModel):
    """Partial update schema — only provided fields are updated."""
    conditions: Optional[List[str]] = None
    medications: Optional[List[Medication]] = None
    allergies: Optional[List[str]] = None
    recent_vitals: Optional[dict] = None
    clinical_notes: Optional[List[dict]] = None


class PatientOut(ORMBase):
    id: UUID
    nhs_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    age: int
    gender: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    medications: List[Any] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    recent_vitals: dict = Field(default_factory=dict)
    clinical_notes: List[Any] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ConversationCreate(BaseModel):
    patient_id: UUID
    selected_guideline: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str
    meta: dict = Field(default_factory=dict)
    timestamp: Optional[datetime] = None


class ConversationOut(ORMBase):
    id: UUID
    patient_id: UUID
    messages: List[Any] = Field(default_factory=list)
    selected_guideline: Optional[str] = None
    extracted_variables: dict = Field(default_factory=dict)
    final_recommendation: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime


class DiagnosisOut(ORMBase):
    id: UUID
    patient_id: UUID
    conversation_id: Optional[UUID] = None
    selected_guideline: Optional[str] = None
    extracted_variables: dict = Field(default_factory=dict)
    pathway_walked: List[Any] = Field(default_factory=list)
    final_recommendation: Optional[str] = None
    urgency: Optional[str] = None
    status: str
    diagnosed_at: datetime
    # Joined fields (optional, populated by list endpoint)
    patient_name: Optional[str] = None
