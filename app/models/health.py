# app/models/health.py

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    agent_ready: bool
    rag_ready: bool
    version: str