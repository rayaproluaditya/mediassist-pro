# app/models/summary.py

from datetime import datetime

from pydantic import BaseModel

from .patient import PatientData


class SummaryResponse(BaseModel):
    patient_data: PatientData
    completion_percentage: float
    conversation_count: int
    last_updated: datetime = datetime.utcnow()