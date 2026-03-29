# app/models/patient.py

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel


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