# EEBC 2021 Agentic AI Compliance System

Multi-agent RAG system for checking EEBC 2021 (Energy Efficiency Building Code) compliance, performing ETTV/RTTV calculations, and providing corrective solution advice.

---

## Architecture

```
User Query
    ↓
Orchestrator Agent  (LLM routing + conversation memory)
    ↓ routes to ↓
┌─────────────────────────────────────────────────────┐
│  Compliance Checker │ ETTV/RTTV Calculator           │
│  Solution Advisor   │ Envelope Specialist (Sec 4)    │
│  Lighting Spec (5)  │ HVAC Specialist (Sec 6)        │
│  SWH Specialist (7) │ Electrical Specialist (Sec 8)  │
│  EEBC Expert (fallback)                              │
└─────────────────────────────────────────────────────┘
    ↓
RAG (ChromaDB · HuggingFace all-MiniLM-L6-v2 · MMR k=20)
    ↓
EEBC 2021 Vector Store
    ↓
Groq LLaMA-3.3-70B → Final Answer
```

---

## Quick Start

### 1. Clone / Unzip the project

```
eebc_system/
├── backend/
│   ├── app.py
│   ├── orchestrator.py
│   ├── rag_core.py
│   ├── models.py
│   ├── ingest.py
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── package.json
    ├── public/
    └── src/
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 3. Ingest EEBC 2021 PDF

```bash
# Create Data folder and place the EEBC 2021 PDF inside it
mkdir Data
# Copy your EEBC_2021.pdf into Data/

# Run ingestor (first run downloads the embedding model ~90 MB)
python ingest.py
```

The ingestor will:
- Load the PDF with PyMuPDF
- Split into 1000-char chunks with 200-char overlap
- Embed with all-MiniLM-L6-v2
- Store in ChromaDB at `./DB/chroma_langchain_db`
- Rename processed files with `_` prefix

### 4. Start Backend

```bash
python app.py
# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

### 5. Frontend Setup

```bash
cd ../frontend
npm install
```

Create `.env` in `frontend/`:
```
REACT_APP_API_URL=http://localhost:8000
```

```bash
npm start
# Opens http://localhost:3000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/ask` | Main agentic endpoint (recommended) |
| DELETE | `/agent/memory` | Clear conversation memory |
| GET | `/agent/list` | List all agents |
| POST | `/upload` | Upload PDF to vector DB |
| POST | `/ingest/eebc` | Trigger bulk ingest from ./Data |
| GET | `/vector/status` | Vector store document count |
| POST | `/tools/rag` | Direct single-agent call (legacy) |
| GET | `/health` | Health check |

### POST /agent/ask
```json
{
  "question": "My wall U-value is 0.9 W/m²·K. Is it EEBC 2021 compliant?",
  "session_id": "session-001"
}
```
Response includes: `answer`, `agents_used`, `routing_method`, `routing_reasoning`, `multi_agent`, `timestamp`

### POST /tools/rag (direct agent)
```json
{
  "question": "Calculate ETTV: Uw=0.5, Aw=120, TDeq=15, Uf=3.0, Af=30, SC=0.6, SF=180, At=150",
  "agent_type": "ETTV/RTTV Calculator"
}
```

Available `agent_type` values:
- `Compliance Checker`
- `ETTV/RTTV Calculator`
- `Solution Advisor`
- `Envelope Specialist`
- `Lighting Specialist`
- `HVAC Specialist`
- `Service Water Heating Specialist`
- `Electrical Power Specialist`
- `EEBC Expert`

---

## Agents Detail

### Compliance Checker
Analyses building parameters, returns COMPLIANT ✅ or NON-COMPLIANT ❌ with specific clause citation.

### ETTV/RTTV Calculator ⭐ Dedicated Agent
Full step-by-step ETTV and RTTV calculations using the exact EEBC 2021 formulas:

**ETTV** = [ (Uw × Aw × TDeq) + (Uf × Af × ΔT) + (SC × Af × SF) ] / At  
**Limit**: ≤ 50 W/m²

**RTTV** = [ (Ur × Ar × TDeqr) + (Us × As × ΔTs) + (SCs × As × SFs) ] / Ar_total  
**Limit**: ≤ 25 W/m²

### Solution Advisor
Recommends corrective actions prioritised by impact + qualitative ROI.

### Section Specialists (4–8)
Envelope, Lighting, HVAC, SWH, Electrical — each scoped to their EEBC section.

### EEBC Expert
General fallback for any EEBC 2021 question not matching a specialist.

---

## Deployment

### Render (Backend)

1. Push `backend/` to GitHub
2. Create a new **Web Service** on Render
3. Set environment:
   - `GROQ_API_KEY` = your key
   - `PORT` = 10000 (auto-set by Render)
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`

> **Important**: The ChromaDB is ephemeral on Render's free tier. Use a persistent disk or mount, or re-ingest on startup.

### Vercel / Netlify (Frontend)

```bash
cd frontend
npm run build
# Deploy the build/ folder
```
Set `REACT_APP_API_URL` to your Render backend URL.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "app.py"]
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | HF embedding model |
| `PORT` | No | `8000` | Server port |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins |

---

## Project Files

```
backend/
├── app.py           Main FastAPI application
├── orchestrator.py  Multi-agent routing + memory
├── rag_core.py      RAG chain + all agent prompts
├── models.py        Groq LLM + HF embeddings initialisation
├── ingest.py        PDF ingestion script
├── requirements.txt Python dependencies
└── .env.example    Environment template

frontend/
├── src/
│   ├── App.js       Main React app (5 tabs)
│   ├── App.css      Dark technical design system
│   ├── index.js     Entry point
│   └── index.css    Global CSS variables
├── public/
│   └── index.html
└── package.json
```
