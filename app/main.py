"""
MediAssist Pro — FastAPI Backend
=================================
Exposes four endpoints:
  GET  /              → service info
  GET  /health        → readiness probe
  POST /api/v1/message   → chat with the intake agent
  GET  /api/v1/summary   → structured patient record
  POST /api/v1/reset     → clear session
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    HealthResponse,
    MessageRequest,
    MessageResponse,
    ResetResponse,
    SummaryResponse,
)
from app.agents.intake_agent import MedicalIntakeAgent
from app.vector_db.medical_retriever import MedicalGuidelinesRetriever

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_retriever: MedicalGuidelinesRetriever | None = None
_agent: MedicalIntakeAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    global _retriever, _agent

    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting %s v%s …", settings.APP_NAME, settings.APP_VERSION)

    # Validate required config (raises on missing GROQ_API_KEY)
    settings.validate()

    # Initialise RAG retriever (non-fatal if ChromaDB is unreachable)
    try:
        _retriever = MedicalGuidelinesRetriever()
        logger.info("RAG retriever ready.")
    except Exception as exc:
        logger.warning("RAG retriever unavailable: %s — continuing without RAG.", exc)
        _retriever = None

    # Initialise agent
    _agent = MedicalIntakeAgent(retriever=_retriever)
    logger.info("Medical intake agent ready.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down %s …", settings.APP_NAME)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=dict, tags=["Info"])
async def root():
    """Service information & endpoint catalogue."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "safety_protocols": "active",
        "endpoints": {
            "POST /api/v1/message": "Send a message to the intake agent",
            "GET  /api/v1/summary": "Retrieve structured patient summary",
            "POST /api/v1/reset":   "Clear the current session",
            "GET  /health":         "Container health probe",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Used by Docker health-check and the Streamlit frontend connection probe.
    Returns 200 only when the agent is initialised.
    """
    return HealthResponse(
        status="healthy" if _agent else "initialising",
        service=settings.APP_NAME,
        agent_ready=_agent is not None,
        rag_ready=_retriever is not None,
        version=settings.APP_VERSION,
    )


@app.post("/api/v1/message", response_model=MessageResponse, tags=["Agent"])
async def process_message(request: MessageRequest):
    """
    Send a patient message to the intake agent.

    The agent will:
    1. Screen for emergency keywords (returns 911 prompt if detected).
    2. Retrieve relevant medical guidelines via RAG.
    3. Call the LLM with full safety constraints.
    4. Validate the response and add disclaimers as needed.
    5. Return the response along with metadata.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialised.")

    try:
        result = await _agent.process_message(request.message)
        return MessageResponse(**result)
    except Exception as exc:
        logger.exception("Error processing message: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/summary", response_model=SummaryResponse, tags=["Agent"])
async def get_summary():
    """Return the structured patient data collected so far in this session."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialised.")

    try:
        summary = _agent.get_summary()
        return SummaryResponse(**summary)
    except Exception as exc:
        logger.exception("Error building summary: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/reset", response_model=ResetResponse, tags=["Agent"])
async def reset_conversation():
    """Clear the current session's conversation history and patient record."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialised.")

    try:
        result = _agent.reset()
        return ResetResponse(**result)
    except Exception as exc:
        logger.exception("Error resetting session: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
