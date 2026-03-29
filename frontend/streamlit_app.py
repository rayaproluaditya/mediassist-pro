"""
MediAssist Pro — Streamlit Frontend
=====================================
Connects to the FastAPI backend and renders a clinical-grade chat UI.
"""

from __future__ import annotations

import time
import requests
import streamlit as st
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
API_BASE    = f"{BACKEND_URL}/api/v1"
HEALTH_URL  = f"{BACKEND_URL}/health"

st.set_page_config(
    page_title="MediAssist Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── global ── */
  .block-container { padding-top: 1.5rem; }

  /* ── emergency banner ── */
  .emergency-banner {
    background: #fff0f0;
    border-left: 6px solid #d32f2f;
    padding: 1.1rem 1.4rem;
    border-radius: 6px;
    margin: 0.8rem 0;
    animation: pulse-red 1.8s ease-in-out infinite;
  }
  @keyframes pulse-red {
    0%,100% { opacity:1; }
    50%      { opacity:.75; }
  }

  /* ── disclaimer banner ── */
  .disclaimer-banner {
    background: #fffde7;
    border-left: 5px solid #f9a825;
    padding: 0.75rem 1rem;
    border-radius: 5px;
    font-size: 0.82rem;
    margin-bottom: 0.8rem;
  }

  /* ── safety status box ── */
  .safety-ok {
    background: #f1f8e9;
    border-left: 4px solid #558b2f;
    padding: 0.6rem 0.9rem;
    border-radius: 5px;
    font-size: 0.83rem;
  }

  /* ── progress label ── */
  .progress-label {
    font-size: 0.78rem;
    color: #555;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"           not in st.session_state: st.session_state.messages           = []
if "backend_connected"  not in st.session_state: st.session_state.backend_connected  = False
if "emergency_active"   not in st.session_state: st.session_state.emergency_active   = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 MediAssist Pro")
st.caption("AI-powered preliminary medical intake assistant  •  Not a diagnostic tool")

st.markdown("""
<div class="disclaimer-banner">
⚠️ <strong>Medical Disclaimer:</strong>
This assistant collects structured symptom information for review by a healthcare professional.
It does <em>not</em> provide diagnoses or medical advice.
<strong>In an emergency, call 911 (US) or your local emergency number immediately.</strong>
</div>
""", unsafe_allow_html=True)

# ── Backend connection probe ──────────────────────────────────────────────────
if not st.session_state.backend_connected:
    probe = st.empty()
    probe.info("🔌 Connecting to backend …")
    for attempt in range(40):
        try:
            r = requests.get(HEALTH_URL, timeout=3)
            if r.status_code == 200 and r.json().get("agent_ready"):
                st.session_state.backend_connected = True
                probe.success("✅ Connected!")
                time.sleep(0.8)
                probe.empty()
                break
        except Exception:
            pass
        time.sleep(1)
        if attempt % 5 == 4:
            probe.info(f"⏳ Waiting for backend … ({attempt+1}s)")
    else:
        probe.error("❌ Could not reach the backend. Is Docker running?")
        st.stop()

# ── Sidebar — patient summary ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Summary")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset", use_container_width=True):
            try:
                r = requests.post(f"{API_BASE}/reset", timeout=10)
                if r.status_code == 200:
                    st.session_state.messages         = []
                    st.session_state.emergency_active = False
                    st.success("Session cleared")
                    st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
    with col2:
        if st.button("📊 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    try:
        r = requests.get(f"{API_BASE}/summary", timeout=8)
        if r.status_code == 200:
            data    = r.json()
            patient = data.get("patient_data", {})
            pct     = data.get("completion_percentage", 0)
            n_turns = data.get("conversation_count", 0)

            if patient.get("chief_complaint"):
                st.info(f"**Chief Complaint**\n{patient['chief_complaint']}")
            else:
                st.caption("No chief complaint yet.")

            if patient.get("symptoms"):
                st.markdown("**Symptoms collected**")
                for s in patient["symptoms"][:8]:
                    st.markdown(f"• {s.get('name','?').title()}")

            cols = st.columns(2)
            if patient.get("duration"):
                cols[0].metric("Duration", patient["duration"])
            if patient.get("severity"):
                cols[1].metric("Severity", patient["severity"])
            if patient.get("onset"):
                st.caption(f"🕐 Onset: {patient['onset']}")
            if patient.get("quality"):
                st.caption(f"✏️ Quality: {patient['quality']}")

            if pct > 0:
                st.markdown("---")
                st.markdown('<p class="progress-label">Data collection progress</p>',
                            unsafe_allow_html=True)
                st.progress(int(pct) / 100)
                st.caption(f"{pct:.0f}% complete  •  {n_turns} exchange(s)")
    except Exception:
        st.caption("Start a conversation to see the summary.")

    st.markdown("---")
    st.markdown("""
<div class="safety-ok">
✅ <strong>Safety layer active</strong><br>
• No diagnosis provided<br>
• Emergency detection enabled<br>
• All data for clinician review only
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🔧 System Status"):
        try:
            hr = requests.get(HEALTH_URL, timeout=3).json()
            st.json(hr)
        except Exception:
            st.caption("Health endpoint unreachable.")

# ── Chat area ─────────────────────────────────────────────────────────────────
st.markdown("### 💬 Intake Conversation")

# Emergency banner (persists until reset)
if st.session_state.emergency_active:
    st.markdown("""
<div class="emergency-banner">
🚨 <strong>EMERGENCY ALERT ACTIVE</strong> — Please call <strong>911</strong>
or go to the nearest Emergency Room immediately. Do not wait.
</div>
""", unsafe_allow_html=True)

# Welcome state
if not st.session_state.messages:
    st.info(
        "👋 Welcome to MediAssist Pro.  \n"
        "Describe your symptoms in your own words and I'll guide you through "
        "a structured intake conversation."
    )

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rag_sources"):
            with st.expander("📚 Guideline sources", expanded=False):
                for src in msg["rag_sources"]:
                    st.caption(f"• {src}")

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Describe your symptoms …",
    disabled=st.session_state.emergency_active,
)

if user_input:
    # Render user bubble
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Analysing …"):
            try:
                resp = requests.post(
                    f"{API_BASE}/message",
                    json={"message": user_input},
                    timeout=60,
                )
                if resp.status_code == 200:
                    result  = resp.json()
                    reply   = result.get("response", "Sorry, I couldn't process that.")
                    sources = result.get("rag_sources", [])

                    st.markdown(reply)

                    if sources:
                        with st.expander("📚 Guideline sources", expanded=False):
                            for src in sources:
                                st.caption(f"• {src}")

                    # Emergency handling
                    if result.get("emergency"):
                        st.session_state.emergency_active = True
                        st.markdown("""
<div class="emergency-banner">
🚨 <strong>EMERGENCY DETECTED</strong><br>
Call <strong>911</strong> or go to the nearest Emergency Room immediately.
</div>
""", unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role":        "assistant",
                        "content":     reply,
                        "rag_sources": sources,
                    })
                else:
                    err = f"Backend error {resp.status_code}: {resp.text[:200]}"
                    st.error(err)
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot reach the backend. Is Docker running?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:11px;'>"
    "MediAssist Pro — Preliminary Symptom Collection Tool<br>"
    "Not a substitute for professional medical advice. "
    "In emergencies call 911."
    "</div>",
    unsafe_allow_html=True,
)
