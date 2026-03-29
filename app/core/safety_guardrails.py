"""
MediAssist Pro — Safety Guardrails
===========================================
Single source of truth for:
  • Emergency detection  → 911 / ER referral
  • Red-flag detection   → urgent-care prompt
  • Response validation  → no-diagnosis check
  • Disclaimer injection → legal safety net
"""

from __future__ import annotations

import re
from typing import List, Tuple

from app.config import settings


DISCLAIMER = (
    "\n\n---\n"
    "*⚠️ This is not a diagnosis and does not constitute medical advice. "
    "Please consult a qualified healthcare professional for personalised care. "
    "In an emergency, call 911 (US) or your local emergency number immediately.*"
)

EMERGENCY_TEMPLATE = """\
⚠️ **URGENT MEDICAL ALERT** ⚠️

I have detected symptoms that may be a **medical emergency**: *{emergency_type}*.

**Please take ONE of these actions right now:**
1. Call **911** (US) or your local emergency number
2. Go to the **nearest Emergency Room**
3. Ask someone nearby to help you immediately

**Do NOT wait.** This message is not a diagnosis — it is a safety precaution \
based on the information you provided.

I am unable to assist further until you have spoken with emergency services.
"""


class SafetyGuardrails:
    """
    Stateless safety layer.  All methods are pure functions operating on text.
    Instantiate once and reuse across requests.
    """

    def __init__(self) -> None:
        self._emergency_kw: frozenset[str] = settings.EMERGENCY_KEYWORDS
        self._red_flags: frozenset[str] = settings.RED_FLAGS
        self._forbidden: frozenset[str] = settings.FORBIDDEN_PHRASES

    # ── Emergency detection ──────────────────────────────────────────────────

    def check_emergency(self, text: str) -> Tuple[bool, str | None]:
        """
        Returns (is_emergency, matched_keyword_or_None).
        Case-insensitive substring match against the emergency keyword list.
        """
        lower = text.lower()
        for kw in self._emergency_kw:
            if kw in lower:
                return True, kw
        return False, None

    def get_emergency_response(self, emergency_type: str) -> str:
        return EMERGENCY_TEMPLATE.format(emergency_type=emergency_type)

    # ── Red-flag detection ───────────────────────────────────────────────────

    def check_red_flags(self, text: str) -> List[str]:
        """Return a list of all red-flag phrases found in *text*."""
        lower = text.lower()
        return [flag for flag in self._red_flags if flag in lower]

    # ── Response validation ──────────────────────────────────────────────────

    def validate_response(self, response: str) -> Tuple[bool, str | None]:
        """
        Ensure the LLM response:
          1. Does not contain forbidden diagnostic phrases.
          2. Contains a safety disclaimer.
        Returns (is_valid, warning_message_or_None).
        """
        lower = response.lower()

        for phrase in self._forbidden:
            if phrase in lower:
                return False, f"Forbidden phrase detected: '{phrase}'"

        has_disclaimer = (
            "not a diagnosis" in lower
            or "consult a healthcare" in lower
            or "consult a qualified" in lower
            or "medical advice" in lower
        )
        if not has_disclaimer:
            return False, "Response is missing required safety disclaimer"

        return True, None

    # ── Disclaimer helpers ───────────────────────────────────────────────────

    def add_disclaimer(self, response: str) -> str:
        """Append the standard disclaimer if it is not already present."""
        if "not a diagnosis" not in response.lower():
            return response + DISCLAIMER
        return response

    def strip_diagnostic_language(self, response: str) -> str:
        """
        Best-effort replacement of forbidden diagnostic phrases with safe
        alternatives.  Used as a last resort before returning a response.
        """
        replacements = {
            r"\byou have\b": "your symptoms may be consistent with",
            r"\bdiagnosed with\b": "possibly associated with",
            r"\bsuffering from\b": "experiencing symptoms that may relate to",
            r"\byou are experiencing\b": "you have described symptoms that could relate to",
            r"\bthis means you\b": "this may suggest",
            r"\bit is definitely\b": "it is possible that",
        }
        for pattern, replacement in replacements.items():
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        return response
