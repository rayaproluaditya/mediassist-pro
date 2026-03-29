"""
MediAssist Pro — Medical Intake Agent
=======================================
Orchestration layer that ties together:
  • Safety guardrails (emergency detection, response validation)
  • RAG retriever     (ChromaDB-backed medical guidelines)
  • LLM calls         (Groq / llama-3.3-70b-versatile)
  • Patient data extraction & structured summary

Design principles:
  - All LLM calls are routed through a single _call_llm() method.
  - Safety checks happen before AND after LLM inference.
  - The agent never stores PII beyond the current session object.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from groq import Groq

from app.config import settings
from app.core.safety_guardrails import SafetyGuardrails
from app.agents.prompt_templates import (
    get_system_prompt,
    build_user_turn,
    get_emergency_response,
)

logger = logging.getLogger(__name__)


class MedicalIntakeAgent:
    """
    Stateful (per-session) medical intake agent.

    Usage:
        agent = MedicalIntakeAgent(retriever=<MedicalGuidelinesRetriever>)
        result = await agent.process_message("I have a headache")
        summary = agent.get_summary()
        agent.reset()
    """

    def __init__(self, retriever=None) -> None:
        self._client = Groq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL_NAME
        self._retriever = retriever
        self._safety = SafetyGuardrails()

        # Conversation memory
        self.conversation_history: List[Dict[str, str]] = []

        # Structured patient record
        self.patient_data: Dict[str, Any] = self._empty_patient_record()

    # ── Public API ────────────────────────────────────────────────────────────

    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Main entry point.  Returns a dict with keys:
            response      (str)
            emergency     (bool)
            emergency_type(str | None)
            red_flags     (list[str])
            rag_sources   (list[str])
        """

        # 1 ─ Emergency gate (pre-LLM)
        is_emergency, emergency_type = self._safety.check_emergency(message)
        if is_emergency:
            return {
                "response": get_emergency_response(emergency_type),
                "emergency": True,
                "emergency_type": emergency_type,
                "red_flags": [],
                "rag_sources": [],
            }

        # 2 ─ Red-flag detection
        red_flags = self._safety.check_red_flags(message)

        # 3 ─ RAG retrieval
        guidelines_context = self._retrieve_context(message)
        rag_sources = self._retriever.get_sources(message) if self._retriever else []

        # 4 ─ Build prompt & call LLM
        user_turn = build_user_turn(
            message=message,
            conversation_history=self.conversation_history,
            guidelines_context=guidelines_context,
        )
        response = self._call_llm(user_turn)

        # 5 ─ Post-generation safety check
        is_valid, warning = self._safety.validate_response(response)
        if not is_valid:
            logger.warning("Safety validation failed (%s) — sanitising response.", warning)
            response = self._safety.strip_diagnostic_language(response)
            response = self._safety.add_disclaimer(response)

        # 6 ─ Append red-flag advisory if needed
        if red_flags:
            flag_str = ", ".join(red_flags)
            response += (
                f"\n\n⚠️ **Clinical Note:** The following symptom(s) warrant "
                f"prompt in-person evaluation: *{flag_str}*. "
                "Please seek care soon."
            )

        # 7 ─ Update conversation history
        self.conversation_history.append({
            "user": message,
            "assistant": response,
        })

        # 8 ─ Extract structured patient fields
        self._extract_patient_info(message)

        return {
            "response": response,
            "emergency": False,
            "emergency_type": None,
            "red_flags": red_flags,
            "rag_sources": rag_sources,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return the current structured patient record plus metadata."""
        required = ["chief_complaint", "symptoms", "duration", "severity"]
        filled = sum(1 for f in required if self.patient_data.get(f))
        completion = round((filled / len(required)) * 100, 1)

        return {
            "patient_data": self.patient_data,
            "completion_percentage": completion,
            "conversation_count": len(self.conversation_history),
        }

    def reset(self) -> Dict[str, str]:
        """Clear session state."""
        self.conversation_history = []
        self.patient_data = self._empty_patient_record()
        return {"status": "reset", "message": "Conversation and patient data cleared."}

    # ── Private helpers ──────────────────────────────────────────────────────

    def _call_llm(self, user_content: str) -> str:
        """Single wrapper around the Groq completion API."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                "I'm sorry, I encountered a technical issue. "
                "Please try again in a moment. "
                "This is not a diagnosis — please consult a healthcare provider."
            )

    def _retrieve_context(self, message: str) -> str:
        if not self._retriever:
            return ""
        try:
            return self._retriever.get_context(message)
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return ""

    def _extract_patient_info(self, message: str) -> None:
        """
        Lightweight rule-based extraction to populate patient_data.
        This runs after every turn so the sidebar summary stays fresh.
        """
        lower = message.lower()

        # ── Chief complaint (first substantive message) ──────────────────────
        if not self.patient_data["chief_complaint"] and len(message) > 5:
            if any(kw in lower for kw in
                   ["have", "feel", "experiencing", "suffering", "pain",
                    "ache", "hurt", "sick", "nausea", "tired", "fever"]):
                self.patient_data["chief_complaint"] = message.split(".")[0].strip()

        # ── Known symptom keywords ────────────────────────────────────────────
        known_symptoms = [
            "headache", "fever", "cough", "pain", "nausea", "fatigue",
            "dizziness", "rash", "swelling", "vomiting", "diarrhoea",
            "diarrhea", "chills", "shortness of breath", "sore throat",
        ]
        for sym in known_symptoms:
            if sym in lower:
                entry = {"name": sym, "description": message}
                if not any(s.get("name") == sym for s in self.patient_data["symptoms"]):
                    self.patient_data["symptoms"].append(entry)

        # ── Duration ─────────────────────────────────────────────────────────
        if not self.patient_data["duration"]:
            m = re.search(
                r"(\d+)\s*(hour|day|week|month|minute)s?",
                lower,
            )
            if m:
                self.patient_data["duration"] = f"{m.group(1)} {m.group(2)}(s)"

        # ── Severity (numeric scale) ──────────────────────────────────────────
        if not self.patient_data["severity"]:
            m = re.search(r"\b([1-9]|10)\s*(out of|/|on a scale)\s*10\b", lower)
            if m:
                self.patient_data["severity"] = f"{m.group(1)}/10"
            else:
                for word in ["mild", "moderate", "severe", "excruciating"]:
                    if word in lower:
                        self.patient_data["severity"] = word
                        break

        # ── Quality descriptor ────────────────────────────────────────────────
        if not self.patient_data["quality"]:
            for q in ["sharp", "dull", "throbbing", "burning", "stabbing",
                      "aching", "pressure", "squeezing", "cramping"]:
                if q in lower:
                    self.patient_data["quality"] = q
                    break

        # ── Onset (when) ─────────────────────────────────────────────────────
        if not self.patient_data["onset"]:
            m = re.search(
                r"(started|began|since|this\s+morning|yesterday|last\s+\w+)"
                r"[\w\s,]*",
                lower,
            )
            if m:
                self.patient_data["onset"] = m.group(0).strip()[:80]

    @staticmethod
    def _empty_patient_record() -> Dict[str, Any]:
        return {
            "chief_complaint": None,
            "symptoms": [],
            "onset": None,
            "duration": None,
            "severity": None,
            "quality": None,
            "location": None,
            "timing": None,
            "context": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "vitals": {},
        }
