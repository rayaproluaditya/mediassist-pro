# app/models/message.py

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         description="Patient's free-text message")
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {"message": "I have had a headache for two days."}
        }
    }


class MessageResponse(BaseModel):
    response: str
    emergency: bool = False
    emergency_type: Optional[str] = None
    red_flags: List[str] = []
    rag_sources: List[str] = []
    timestamp: datetime = datetime.utcnow()