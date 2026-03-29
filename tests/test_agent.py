"""
tests/test_agent.py
===================
Pytest unit tests for MediAssist Pro core components.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ── Safety Guardrails ─────────────────────────────────────────────────────────

class TestSafetyGuardrails:
    """Tests for app.core.safety_guardrails.SafetyGuardrails"""

    @pytest.fixture
    def guardrails(self):
        from app.core.safety_guardrails import SafetyGuardrails
        return SafetyGuardrails()

    def test_emergency_chest_pain(self, guardrails):
        is_em, kw = guardrails.check_emergency("I have severe chest pain")
        assert is_em is True
        assert "chest pain" in kw

    def test_emergency_difficulty_breathing(self, guardrails):
        is_em, kw = guardrails.check_emergency("I have difficulty breathing")
        assert is_em is True

    def test_emergency_suicidal(self, guardrails):
        is_em, kw = guardrails.check_emergency("I have been feeling suicidal")
        assert is_em is True

    def test_no_emergency_headache(self, guardrails):
        is_em, kw = guardrails.check_emergency("I have a mild headache")
        assert is_em is False
        assert kw is None

    def test_red_flag_worst_headache(self, guardrails):
        flags = guardrails.check_red_flags("I have the worst headache of my life")
        assert len(flags) > 0

    def test_validate_response_passes_with_disclaimer(self, guardrails):
        resp = "Your symptoms may be consistent with a viral infection. This is not a diagnosis. Please consult a qualified healthcare provider."
        is_valid, warning = guardrails.validate_response(resp)
        assert is_valid is True
        assert warning is None

    def test_validate_response_fails_on_forbidden_phrase(self, guardrails):
        resp = "You have influenza."
        is_valid, warning = guardrails.validate_response(resp)
        assert is_valid is False

    def test_validate_response_fails_missing_disclaimer(self, guardrails):
        resp = "Your symptoms could be many things."
        is_valid, warning = guardrails.validate_response(resp)
        assert is_valid is False

    def test_add_disclaimer_appends(self, guardrails):
        resp = "Your symptoms may be consistent with a cold."
        updated = guardrails.add_disclaimer(resp)
        assert "not a diagnosis" in updated.lower()

    def test_add_disclaimer_idempotent(self, guardrails):
        resp = "Not a diagnosis. Please consult a qualified healthcare provider."
        # Adding disclaimer to text that already contains it should not duplicate
        updated = guardrails.add_disclaimer(resp)
        assert updated.lower().count("not a diagnosis") == 1

    def test_strip_diagnostic_language(self, guardrails):
        dirty = "You have diabetes and you are experiencing high blood sugar."
        clean = guardrails.strip_diagnostic_language(dirty)
        assert "you have" not in clean.lower()

    def test_emergency_case_insensitive(self, guardrails):
        is_em, _ = guardrails.check_emergency("I AM HAVING A SEIZURE")
        assert is_em is True


# ── Patient info extraction ───────────────────────────────────────────────────

class TestPatientExtraction:
    """Tests for the rule-based extraction inside MedicalIntakeAgent"""

    @pytest.fixture
    def agent(self):
        with patch("app.agents.intake_agent.Groq"):
            from app.agents.intake_agent import MedicalIntakeAgent
            a = MedicalIntakeAgent(retriever=None)
            return a

    def test_duration_extraction(self, agent):
        agent._extract_patient_info("I have had a headache for 3 days")
        assert agent.patient_data["duration"] is not None
        assert "3" in agent.patient_data["duration"]

    def test_severity_scale_extraction(self, agent):
        agent._extract_patient_info("Pain is about 7 out of 10")
        assert agent.patient_data["severity"] == "7/10"

    def test_symptom_keyword_extraction(self, agent):
        agent._extract_patient_info("I have a headache and fever")
        names = [s["name"] for s in agent.patient_data["symptoms"]]
        assert "headache" in names
        assert "fever" in names

    def test_quality_descriptor_extraction(self, agent):
        agent._extract_patient_info("The pain is throbbing and sharp")
        assert agent.patient_data["quality"] in ("throbbing", "sharp")

    def test_reset_clears_data(self, agent):
        agent._extract_patient_info("fever for 2 days")
        agent.reset()
        assert agent.patient_data["chief_complaint"] is None
        assert agent.patient_data["symptoms"] == []
        assert agent.conversation_history == []


# ── Integration: process_message with mocked LLM ──────────────────────────────

class TestAgentIntegration:

    @pytest.fixture
    def agent_with_mock_llm(self):
        with patch("app.agents.intake_agent.Groq") as MockGroq:
            mock_choice = MagicMock()
            mock_choice.message.content = (
                "Thank you for sharing that. On a scale of 0 to 10, "
                "how severe is the pain? This is not a diagnosis. "
                "Please consult a qualified healthcare provider."
            )
            MockGroq.return_value.chat.completions.create.return_value.choices = [mock_choice]
            from app.agents.intake_agent import MedicalIntakeAgent
            return MedicalIntakeAgent(retriever=None)

    @pytest.mark.asyncio
    async def test_normal_message_returns_response(self, agent_with_mock_llm):
        result = await agent_with_mock_llm.process_message("I have a headache")
        assert result["emergency"] is False
        assert len(result["response"]) > 10

    @pytest.mark.asyncio
    async def test_emergency_message_bypasses_llm(self, agent_with_mock_llm):
        result = await agent_with_mock_llm.process_message("I have chest pain")
        assert result["emergency"] is True
        assert "911" in result["response"] or "emergency" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_conversation_history_grows(self, agent_with_mock_llm):
        await agent_with_mock_llm.process_message("I have a headache")
        await agent_with_mock_llm.process_message("It's been for two days")
        assert len(agent_with_mock_llm.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_summary_completion_increases(self, agent_with_mock_llm):
        summary_before = agent_with_mock_llm.get_summary()["completion_percentage"]
        await agent_with_mock_llm.process_message("I have a fever for 3 days")
        summary_after = agent_with_mock_llm.get_summary()["completion_percentage"]
        assert summary_after >= summary_before
