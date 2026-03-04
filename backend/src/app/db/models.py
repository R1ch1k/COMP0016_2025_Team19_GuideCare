import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Date, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nhs_number = Column(String(64), unique=True, nullable=False)
    first_name = Column(String(128), nullable=False)
    last_name = Column(String(128), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(32), nullable=True)
    conditions = Column(JSONB, default=list)
    medications = Column(JSONB, default=list)
    allergies = Column(JSONB, default=list)
    recent_vitals = Column(JSONB, default=dict)
    clinical_notes = Column(JSONB, default=list)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    messages = Column(JSONB, default=list)  # list of message objects
    selected_guideline = Column(String(64), nullable=True)
    extracted_variables = Column(JSONB, default=dict)
    final_recommendation = Column(Text, nullable=True)
    status = Column(String(32), default="in_progress", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    selected_guideline = Column(String(64), nullable=True)
    extracted_variables = Column(JSONB, default=dict)
    pathway_walked = Column(JSONB, default=list)
    final_recommendation = Column(Text, nullable=True)
    urgency = Column(String(32), nullable=True)
    status = Column(String(32), default="completed", nullable=False)
    diagnosed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
