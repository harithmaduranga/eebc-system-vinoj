"""
FastAPI Backend — EEBC 2021 Agentic RAG System
================================================
Endpoints:
  POST /agent/ask          ← Orchestrated agentic endpoint (recommended)
  DELETE /agent/memory     ← Clear conversation memory
  GET  /agent/list         ← List all available agents
  POST /upload             ← Upload EEBC PDF to vector store
  POST /ingest/eebc        ← Trigger ingestion from ./Data folder
  GET  /vector/status      ← Check vector store document count
  POST /tools/rag          ← Legacy direct-agent endpoint
  GET  /health
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rag_core import initialize_rag, query_rag, refresh_vector_store
from orchestrator import get_orchestrator, AGENT_REGISTRY
from models import Models

load_dotenv()

app = FastAPI(
    title="EEBC 2021 Agentic RAG API",
    description="Orchestrator-driven multi-agent RAG system for EEBC 2021 compliance",
    version="3.0.0",
    redirect_slashes=False,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024

# ── CORS ───────────────────────────────────────────────────────────────────────
# allow_credentials=True is incompatible with allow_origins=["*"] per the CORS
# spec — browsers silently drop the response. Use explicit origins + credentials
# only when CORS_ORIGINS is set; fall back to wildcard without credentials.
_cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if _cors_origins_env and _cors_origins_env != "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _cors_origins_env.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
        allow_headers=["*"],
    )


# ── Pydantic models ────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class AgentResponse(BaseModel):
    question: str
    answer: str
    agents_used: List[str]
    routing_method: str
    routing_reasoning: str
    multi_agent: bool
    timestamp: str
    status: str = "success"


class RAGRequest(BaseModel):
    question: str
    agent_type: str = "EEBC Expert"


class RAGResponse(BaseModel):
    question: str
    answer: str
    agent_type: str
    timestamp: str
    status: str = "success"


class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    chunks_created: int
    upload_id: Optional[str] = None


class ComplianceUploadCheckResponse(BaseModel):
    filename: str
    status: str
    message: str
    chunks_created: int
    upload_id: str
    compliance_answer: str
    solution_answer: str
    timestamp: str


class ClearMemoryRequest(BaseModel):
    session_id: Optional[str] = "default"


# ── Root & Health ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "EEBC 2021 Agentic RAG API v3.0",
        "status": "running",
        "agents": list(AGENT_REGISTRY.keys()),
        "endpoints": {
            "orchestrator":  "POST /agent/ask",
            "clear_memory":  "DELETE /agent/memory",
            "agent_list":    "GET  /agent/list",
            "upload":        "POST /upload",
            "ingest":        "POST /ingest/eebc",
            "vector_status": "GET  /vector/status",
            "legacy_rag":    "POST /tools/rag",
            "health":        "GET  /health",
            "api_docs":      "GET  /docs",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "EEBC 2021 Agentic RAG",
        "version": "3.0.0",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "timestamp": datetime.now().isoformat(),
    }


# ── Agentic endpoints ──────────────────────────────────────────────────────────
@app.post("/agent/ask", response_model=AgentResponse)
async def agent_ask(request: AgentRequest):
    """
    Main agentic endpoint.
    Analyses the question → routes to best specialist(s) → synthesises answer.
    Remembers the last 10 conversation turns per session.
    """
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            question=request.question,
            session_id=request.session_id,
        )
        return AgentResponse(
            question=request.question,
            answer=result["answer"],
            agents_used=result["agents_used"],
            routing_method=result["routing_method"],
            routing_reasoning=result["routing_reasoning"],
            multi_agent=result["multi_agent"],
            timestamp=result["timestamp"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {str(e)}")


@app.delete("/agent/memory")
async def clear_agent_memory(request: ClearMemoryRequest):
    """Clear the orchestrator's conversation memory for a fresh session."""
    try:
        get_orchestrator().clear_memory()
        return {"status": "success", "message": "Conversation memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/list")
async def list_agents():
    """Return all available specialist agents and their descriptions."""
    return {
        "agents": [
            {"name": name, "description": meta["description"], "keywords": meta["keywords"]}
            for name, meta in AGENT_REGISTRY.items()
        ]
    }


# ── Vector store endpoints ─────────────────────────────────────────────────────
@app.get("/vector/status")
async def vector_status():
    """Check how many document chunks are in the vector store."""
    try:
        models_obj = Models()
        vs = Chroma(
            collection_name=models_obj.collection_name,
            embedding_function=models_obj.embeddings_hf,
            persist_directory="./DB/chroma_langchain_db",
        )
        count = vs._collection.count()
        return {
            "status": "ok",
            "document_chunks": count,
            "collection": models_obj.collection_name,
            "persist_directory": "./DB/chroma_langchain_db",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e), "document_chunks": 0}


@app.post("/ingest/eebc")
async def ingest_eebc():
    """
    Trigger ingestion of all PDFs in the ./Data folder.
    Rename processed files with underscore prefix to avoid re-ingestion.
    """
    data_folder = Path("./Data")
    data_folder.mkdir(exist_ok=True)

    pdf_files = [f for f in data_folder.glob("*.pdf") if not f.name.startswith("_")]
    if not pdf_files:
        return {"status": "no_files", "message": "No unprocessed PDFs found in ./Data folder"}

    models_obj = Models()
    vs = Chroma(
        collection_name=models_obj.collection_name,
        embedding_function=models_obj.embeddings_hf,
        persist_directory="./DB/chroma_langchain_db",
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    results = []
    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            uuids = [str(uuid4()) for _ in chunks]
            vs.add_documents(documents=chunks, ids=uuids)

            # Mark as processed
            new_path = pdf_path.parent / ("_" + pdf_path.name)
            pdf_path.rename(new_path)

            results.append({"file": pdf_path.name, "chunks": len(chunks), "status": "ok"})
        except Exception as e:
            results.append({"file": pdf_path.name, "status": "error", "detail": str(e)})

    refresh_vector_store()
    return {"status": "done", "files_processed": len(pdf_files), "results": results}


# ── Upload endpoint ────────────────────────────────────────────────────────────
def _ingest_uploaded_pdf(file: UploadFile) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    upload_dir = Path("./Uploads")
    upload_dir.mkdir(exist_ok=True)
    safe_filename = Path(file.filename).name
    upload_id = str(uuid4())
    file_path = upload_dir / f"{upload_id}_{safe_filename}"

    try:
        bytes_written = 0
        with open(file_path, "wb") as buf:
            while chunk := file.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                buf.write(chunk)

        models_obj = Models()
        vs = Chroma(
            collection_name=models_obj.collection_name,
            embedding_function=models_obj.embeddings_hf,
            persist_directory="./DB/chroma_langchain_db",
        )

        loader = PyMuPDFLoader(str(file_path))
        raw_docs = loader.load()
        if not raw_docs:
            raise HTTPException(status_code=400, detail="No content found in PDF")

        for doc in raw_docs:
            doc.metadata.update(
                {
                    "upload_id": upload_id,
                    "original_filename": safe_filename,
                    "uploaded_at": datetime.now().isoformat(),
                }
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)
        uuids = [str(uuid4()) for _ in chunks]
        vs.add_documents(documents=chunks, ids=uuids)
        refresh_vector_store()

        return UploadResponse(
            filename=safe_filename,
            status="success",
            message="PDF uploaded and ingested successfully",
            chunks_created=len(chunks),
            upload_id=upload_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the RAG vector store."""
    return _ingest_uploaded_pdf(file)


@app.post("/compliance/upload-check", response_model=ComplianceUploadCheckResponse)
async def upload_and_check_compliance(file: UploadFile = File(...)):
    """Upload a PDF, ingest it, and run compliance analysis against only that PDF."""
    uploaded = _ingest_uploaded_pdf(file)
    metadata_filter = {"upload_id": uploaded.upload_id}
    query = (
        f'Analyze the uploaded building document "{uploaded.filename}" for EEBC 2021 compliance. '
        "Use only the uploaded document context. Identify compliant and non-compliant parameters, "
        "cite relevant EEBC clauses where available, and clearly flag missing information."
    )
    solution_query = (
        f'For the uploaded building document "{uploaded.filename}", recommend corrections for any '
        "non-compliant or incomplete items found in the compliance analysis."
    )

    compliance_answer = query_rag(query, "Compliance Checker", metadata_filter=metadata_filter)
    solution_answer = query_rag(solution_query, "Solution Advisor", metadata_filter=metadata_filter)

    return ComplianceUploadCheckResponse(
        filename=uploaded.filename,
        status="success",
        message="PDF uploaded, ingested, and checked successfully",
        chunks_created=uploaded.chunks_created,
        upload_id=uploaded.upload_id,
        compliance_answer=compliance_answer,
        solution_answer=solution_answer,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ── Legacy direct-agent endpoint ───────────────────────────────────────────────
@app.post("/tools/rag", response_model=RAGResponse)
async def rag_direct(request: RAGRequest):
    """Legacy endpoint — calls a single named agent directly (no orchestration)."""
    if request.agent_type not in AGENT_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{request.agent_type}'. Use GET /agent/list for valid names.",
        )
    try:
        answer = query_rag(request.question, request.agent_type)
        return RAGResponse(
            question=request.question,
            answer=answer,
            agent_type=request.agent_type,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ── Server entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    import sys

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("=" * 60)
    print("  EEBC 2021 Agentic RAG System v3.0")
    print("=" * 60)
    print(f"  Groq Key    : {'✅ Configured' if os.getenv('GROQ_API_KEY') else '❌ MISSING'}")
    print(f"  Environment : {'Production (Render)' if os.getenv('RENDER') else 'Development'}")

    print("\n[1/2] Initialising RAG system...")
    if not initialize_rag():
        print("ERROR: RAG initialization failed")
        sys.exit(1)
    print("      RAG system ready ✅")

    print("[2/2] Warming up Orchestrator...")
    try:
        get_orchestrator()
        print("      Orchestrator ready ✅")
    except Exception as e:
        print(f"ERROR: Orchestrator failed: {e}")
        sys.exit(1)

    port = int(os.getenv("PORT", 8000))
    print(f"\nAPI Docs : http://localhost:{port}/docs")
    print(f"Starting on port {port}...\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
