"""
MediAssist Pro — Centralised Configuration
All settings are loaded from environment variables / .env file.
"""

import os
from typing import Set
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "MediAssist Pro"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "AI-powered medical intake agent for preliminary symptom analysis. "
        "Not a diagnostic tool."
    )
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Groq LLM ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = 0.2   # low temperature → deterministic, safe output
    GROQ_MAX_TOKENS: int = 600

    # ── ChromaDB / RAG ───────────────────────────────────────────────────────
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "chromadb")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_COLLECTION: str = "medical_guidelines"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_TOP_K: int = 4

    # ── Local persist path (used when ChromaDB HTTP client is unavailable) ───
    CHROMA_PERSIST_DIR: str = "data/embeddings"

    # ── Guidelines data path ─────────────────────────────────────────────────
    GUIDELINES_PATH: str = "data/guidelines"

    # ── Safety — forbidden diagnostic phrases ────────────────────────────────
    FORBIDDEN_PHRASES: Set[str] = frozenset({
        "you have",
        "diagnosed with",
        "suffering from",
        "you are experiencing",
        "this means you",
        "it is definitely",
    })

    # ── Safety — emergency keywords (trigger 911 alert) ──────────────────────
    EMERGENCY_KEYWORDS: Set[str] = frozenset({
        "chest pain",
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "severe bleeding",
        "uncontrolled bleeding",
        "loss of consciousness",
        "passed out",
        "fainted",
        "seizure",
        "stroke",
        "heart attack",
        "severe allergic reaction",
        "anaphylaxis",
        "suicidal",
        "want to die",
        "head injury",
        "poisoning",
        "overdose",
        "choking",
        "face drooping",
        "arm weakness",
        "sudden numbness",
        "sudden vision loss",
    })

    # ── Safety — red-flag symptoms (urgent care, not 911) ────────────────────
    RED_FLAGS: Set[str] = frozenset({
        "fever above 103",
        "fever above 104",
        "stiff neck with fever",
        "sudden severe headache",
        "worst headache of my life",
        "vision changes",
        "weakness on one side",
        "slurred speech",
        "confusion",
        "high fever",
    })

    def validate(self) -> bool:
        """Raise ValueError if any required secret is missing."""
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Get a free key at https://console.groq.com"
            )
        return True

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton — import this everywhere
settings = Settings()
