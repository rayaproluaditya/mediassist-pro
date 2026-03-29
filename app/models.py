"""
MediAssist Pro — Pydantic data models for API request / response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Inbound ──────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         description="Patient's free-text message")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

    model_config = {
        "json_schema_extra": {
            "example": {"message": "I have had a headache for two days."}
        }
    }


# ── Outbound ─────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    response: str
    emergency: bool = False
    emergency_type: Optional[str] = None
    red_flags: List[str] = []
    rag_sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Symptom(BaseModel):
    name: str
    description: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    location: Optional[str] = None
    timestamp: Optional[datetime] = None


class PatientData(BaseModel):
    chief_complaint: Optional[str] = None
    symptoms: List[Symptom] = []
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    quality: Optional[str] = None
    location: Optional[str] = None
    timing: Optional[str] = None
    context: Optional[str] = None
    associated_symptoms: List[str] = []
    medical_history: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []
    vitals: Dict[str, Any] = {}


class SummaryResponse(BaseModel):
    patient_data: PatientData
    completion_percentage: float
    conversation_count: int
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    service: str
    agent_ready: bool
    rag_ready: bool
    version: str


class ResetResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
