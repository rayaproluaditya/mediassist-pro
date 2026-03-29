# 🏥 MediAssist Pro

> **AI-powered preliminary medical intake agent with RAG, fine-tuning, and strict safety guardrails.**

[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docs.docker.com/compose/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)](https://trychroma.com)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Setup Instructions](#setup-instructions)
5. [Design Choices](#design-choices)
6. [Fine-Tuning](#fine-tuning)
7. [API Reference](#api-reference)
8. [Safety Protocols](#safety-protocols)
9. [Testing](#testing)

---

## Overview

MediAssist Pro is a **medically-grounded conversational intake agent** that:

- Collects structured symptom data using the clinical **OPQRST framework**
- Grounds every response with **CDC / WHO / clinical guidelines** via RAG
- Enforces **strict safety guardrails** (no diagnosis, emergency escalation)
- Produces a structured **patient intake summary** ready for clinician review
- Runs end-to-end with a single `docker-compose up` command

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                             │
│                    http://localhost:8501                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Streamlit Frontend  (port 8501)                   │
│  • Chat UI with emergency banners                               │
│  • Live patient summary sidebar                                 │
│  • Connection probe / health check                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │  REST  /api/v1/message
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI Backend  (port 8001 → 8000)                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MedicalIntakeAgent (LangChain)              │   │
│  │                                                         │   │
│  │  1. SafetyGuardrails.check_emergency()   ← PRE-LLM      │   │
│  │  2. MedicalGuidelinesRetriever.get_context() ← RAG      │   │
│  │  3. Groq LLM  (llama-3.3-70b-versatile)                 │   │
│  │  4. SafetyGuardrails.validate_response() ← POST-LLM     │   │
│  │  5. Patient data extraction                             │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │  HTTP client                           │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              ChromaDB  (port 8000)                              │
│  • Collection: medical_guidelines                               │
│  • Embedding:  all-MiniLM-L6-v2  (384-dim)                     │
│  • Persistent volume: chromadb_data                            │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow (per message)

```
Patient message
      │
      ▼
[1] Emergency keyword scan   ──→  YES → Return 911 alert immediately
      │ NO
      ▼
[2] RAG: retrieve top-4 guideline chunks from ChromaDB
      │
      ▼
[3] Build prompt:
      system prompt (OPQRST rules + safety constraints)
      + last 4 conversation turns
      + retrieved guidelines context
      + patient message
      │
      ▼
[4] Groq API  →  LLM response  (llama-3.3-70b, temp=0.2)
      │
      ▼
[5] Post-generation safety check:
      • No forbidden phrases?
      • Disclaimer present?
      • Strip / rewrite if needed
      │
      ▼
[6] Append red-flag advisory (if applicable)
      │
      ▼
[7] Update conversation history + extract patient fields
      │
      ▼
Return MessageResponse {response, emergency, rag_sources, red_flags}
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/mediassist-pro.git
cd mediassist-pro

# 2. Set your Groq API key (free at https://console.groq.com)
cp .env.example .env
# Edit .env and set:  GROQ_API_KEY=gsk_...

# 3. Start everything
docker-compose up --build

# 4. Open the app
open http://localhost:8501
```

That's it. ChromaDB is auto-seeded with medical guidelines on first start.

---

## Setup Instructions

### Prerequisites

| Tool | Version |
|------|---------|
| Docker | ≥ 24 |
| Docker Compose | ≥ 2.20 |
| Groq API key | Free at [console.groq.com](https://console.groq.com) |

### Step-by-step

**1. Clone and configure**
```bash
git clone https://github.com/your-org/mediassist-pro.git
cd mediassist-pro
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_your_key_here
MODEL_NAME=llama-3.3-70b-versatile
```

**2. Build and start**
```bash
docker-compose up --build
```

Services start in order: ChromaDB → Backend → Frontend.
First build downloads the `all-MiniLM-L6-v2` model (~90 MB) — takes ~2 min.

**3. Verify services**
```bash
curl http://localhost:8001/health
# {"status":"healthy","agent_ready":true,"rag_ready":true,...}

curl http://localhost:8000/api/v1/heartbeat
# {"nanosecond heartbeat":...}
```

**4. Open UI**
```
http://localhost:8501
```

**5. Optional: manually seed ChromaDB**
```bash
docker-compose exec backend python scripts/seed_vectordb.py \
    --host chromadb --port 8000 --guidelines data/guidelines
```

**6. Stop**
```bash
docker-compose down        # stop
docker-compose down -v     # stop + delete ChromaDB volume
```

---

## Design Choices

### LLM: Groq + llama-3.3-70b-versatile

| Choice | Rationale |
|--------|-----------|
| **Groq** | Sub-second inference latency; generous free tier for prototyping |
| **llama-3.3-70b** | Strong instruction-following, good clinical vocabulary, open-weight |
| **Temperature 0.2** | Deterministic, consistent clinical responses; reduces hallucination risk |

**Trade-off:** Groq is not HIPAA-compliant out of the box. For production, swap to Azure OpenAI (BAA available) or a self-hosted model.

### Vector DB: ChromaDB

| Choice | Rationale |
|--------|-----------|
| **ChromaDB** | Easy Docker deployment; native Python client; good OSS community |
| **all-MiniLM-L6-v2** | Fast CPU inference; 384-dim vectors; strong semantic search for medical text |
| **Top-k = 4** | Balances context richness vs prompt length |

**Trade-off:** For very large corpora (>1M chunks), Qdrant or Weaviate with HNSW indexing would scale better.

### Orchestration: LangChain

| Choice | Rationale |
|--------|-----------|
| **LangChain** | Mature RAG abstractions; Chroma integration; easy to swap LLM providers |
| **Custom agent loop** | Simpler than LangGraph for single-turn intake; full control over safety checks |

**Trade-off:** LlamaIndex offers better multi-document retrieval but more complex setup.

### Fine-tuning: LoRA on TinyLlama-1.1B

| Choice | Rationale |
|--------|-----------|
| **TinyLlama 1.1B** | Fits on a single T4 GPU; fast iteration; deployable on CPU |
| **LoRA r=16** | ~0.5% of parameters trained; preserves base knowledge |
| **SFT (instruction fine-tuning)** | Direct alignment to OPQRST + empathetic clinical tone |

The fine-tuned adapter is used as a fallback when Groq is unavailable.

---

## Fine-Tuning

### Dataset preparation
```bash
python fine_tuning/scripts/generate_medical_dataset.py \
    --output fine_tuning/data \
    --train-split 0.85
# Outputs: train.json (21 samples), validation.json (4 samples)
```

### Training
```bash
# Requires GPU; set up a venv first
pip install -r requirements.txt

python fine_tuning/scripts/finetune_medical_agent.py \
    --train  fine_tuning/data/train.json \
    --val    fine_tuning/data/validation.json \
    --output fine_tuning/checkpoints/medical_agent_model
```

### Key hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | TinyLlama-1.1B-Chat | Swap to Mistral-7B for better quality |
| LoRA rank (r) | 16 | Higher = more capacity but more memory |
| LoRA alpha | 32 | Scaling factor: alpha/r = 2 |
| Epochs | 3 | Increase for large datasets |
| Learning rate | 2e-4 | Cosine schedule with 50 warmup steps |
| Batch size | 4 × 4 grad accum = 16 effective | |
| Max sequence len | 512 tokens | |

### Training metrics
See `fine_tuning/logs/` for loss curves. Typical final eval loss: ~0.8–1.2 on this dataset size.

The notebook `medical_agent_model.ipynb` contains the full interactive training run with cell-by-cell outputs.

---

## API Reference

### `POST /api/v1/message`

Send a patient message.

**Request:**
```json
{ "message": "I have had a headache for two days." }
```

**Response:**
```json
{
  "response": "I'm sorry to hear that. Can you tell me where the pain is located?...\n\nThis is not a diagnosis. Please consult a qualified healthcare provider.",
  "emergency": false,
  "emergency_type": null,
  "red_flags": [],
  "rag_sources": ["CDC / Headache Protocol"],
  "timestamp": "2024-06-01T12:00:00"
}
```

### `GET /api/v1/summary`

Get the structured patient record.

**Response:**
```json
{
  "patient_data": {
    "chief_complaint": "headache for two days",
    "symptoms": [{"name": "headache", "description": "..."}],
    "duration": "2 day(s)",
    "severity": null
  },
  "completion_percentage": 50.0,
  "conversation_count": 2
}
```

### `POST /api/v1/reset`

Clear the session.

### `GET /health`

Readiness probe.

---

## Safety Protocols

### Emergency escalation (pre-LLM)
Triggers on: chest pain, difficulty breathing, seizure, stroke signs, suicidal ideation, poisoning, overdose, and 20+ other keywords. Returns 911 alert; bypasses LLM entirely.

### No-diagnosis enforcement (post-LLM)
Forbidden phrases: "you have", "diagnosed with", "suffering from", "this means you", "it is definitely". Auto-stripped and replaced with safe language if detected.

### Mandatory disclaimer
Every response must contain "This is not a diagnosis" or equivalent. Added automatically if missing.

### Red-flag advisory
Phrases like "worst headache of my life" or "high fever" trigger a clinical advisory appended to the response urging prompt in-person evaluation.

---

## Testing

```bash
# Run unit tests (no Docker needed)
pip install -r requirements.txt
pytest tests/ -v --cov=app

# Expected output: 15 tests passing
```

---

## Project Structure

```
mediassist-pro/
├── app/
│   ├── agents/
│   │   ├── intake_agent.py        # Core orchestration
│   │   └── prompt_templates.py    # System & user prompts
│   ├── core/
│   │   └── safety_guardrails.py   # Emergency & validation logic
│   ├── vector_db/
│   │   └── medical_retriever.py   # ChromaDB RAG retriever
│   ├── config.py                  # All settings (env-driven)
│   ├── models.py                  # Pydantic schemas
│   └── main.py                    # FastAPI app
├── frontend/
│   └── streamlit_app.py           # Streamlit UI
├── fine_tuning/
│   ├── scripts/
│   │   ├── generate_medical_dataset.py
│   │   └── finetune_medical_agent.py
│   └── data/                      # train/val JSON
├── data/
│   └── guidelines/                # Medical guideline JSONs (seeded to ChromaDB)
├── scripts/
│   └── seed_vectordb.py           # Manual ChromaDB seeder
├── tests/
│   └── test_agent.py              # Pytest unit tests
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
