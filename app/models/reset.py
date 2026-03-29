# app/models/reset.py

from datetime import datetime
from pydantic import BaseModel, Field


class ResetResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)