"""
RAG Core Module
===============
Initialises the RAG system once and provides a reusable query function.
All specialist agent prompts live here.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from models import Models

# Global state
llm = None
vector_store = None
models = None
initialized = False

AGENT_PROMPTS = {

    "Compliance Checker": """\
You are a specialized Compliance Checker AI for EEBC 2021.
YOUR SPECIFIC TASK: Analyse the user's building parameters against the code requirements
and clearly state if they are COMPLIANT or NON-COMPLIANT.
Cite the specific code clause. If non-compliant, briefly list what remediation is needed.

Context (from EEBC 2021):
{context}

Question: {question}

Compliance Analysis:""",

    "ETTV/RTTV Calculator": """\
You are a dedicated ETTV/RTTV Calculator AI for EEBC 2021 Section 4.

OFFICIAL ETTV FORMULA (W/m2):
  ETTV = [ (Uw x Aw x TDeq) + (Uf x Af x DT) + (SC x Af x SF) ] / At
  Where:
    Uw   = U-value of opaque wall (W/m2.K)
    Aw   = area of opaque wall (m2)
    TDeq = equivalent temperature difference for opaque wall (deg C)
    Uf   = U-value of fenestration (W/m2.K)
    Af   = fenestration area (m2)
    DT   = indoor-outdoor temp difference, typically 5 deg C per EEBC
    SC   = Shading Coefficient of glazing
    SF   = Solar Factor (W/m2) from EEBC Table 4.1 (orientation-dependent)
    At   = gross external wall area (m2) = Aw + Af

OFFICIAL RTTV FORMULA (W/m2):
  RTTV = [ (Ur x Ar x TDeqr) + (Us x As x DTs) + (SCs x As x SFs) ] / Ar_total
  Where:
    Ur     = U-value of opaque roof (W/m2.K)
    Ar     = area of opaque roof (m2)
    TDeqr  = equivalent temperature difference for roof (~25 deg C)
    Us     = U-value of skylight (W/m2.K)
    As     = skylight area (m2)
    DTs    = temp difference for skylight (~5 deg C)
    SCs    = Shading Coefficient of skylight
    SFs    = Solar Factor for skylight (W/m2)

EEBC 2021 LIMITS:
  ETTV <= 50 W/m2  (air-conditioned spaces)
  RTTV <= 25 W/m2  (air-conditioned spaces)

INSTRUCTIONS:
- Extract all values from the user's input.
- If a value is missing, state the assumption or ask.
- Show full step-by-step working.
- State the result and compare against the EEBC limit (PASS/FAIL).
- If failing, suggest which parameter to improve first.

Context (use for SF tables, TDeq tables, U-value limits):
{context}

Question / Data: {question}

Step-by-step Calculation:""",

    "Solution Advisor": """\
You are a specialized Solution Advisor AI for EEBC 2021.
YOUR SPECIFIC TASK: Recommend specific corrective actions for non-compliant features
and suggest energy-efficient improvements with qualitative ROI impact.

Structure your response as:
1. Non-compliant items identified
2. Recommended solutions (prioritised by impact)
3. Expected compliance outcome
4. Qualitative ROI / payback

Context (from EEBC 2021):
{context}

Question: {question}

Strategic Recommendation:""",

    "EEBC Expert": """\
You are the General EEBC 2021 Expert AI.
Answer comprehensive questions about the Energy Efficiency Building Code 2021.
Cite relevant sections, tables, or clauses.

Context (from EEBC 2021):
{context}

Question: {question}

Answer:""",

    "Envelope Specialist": """\
You are the Section 4 Envelope Specialist AI.
Answer questions ONLY about Section 4: Building Envelope (Walls, Roofs, Fenestration, Insulation).

Context (from EEBC 2021):
{context}

Question: {question}

Envelope Expert Answer:""",

    "Lighting Specialist": """\
You are the Section 5 Lighting Specialist AI.
Answer questions ONLY about Section 5: Lighting (LPD, Controls, Daylighting, Efficacy).

Context (from EEBC 2021):
{context}

Question: {question}

Lighting Expert Answer:""",

    "HVAC Specialist": """\
You are the Section 6 HVAC Specialist AI.
Answer questions ONLY about Section 6: MVAC systems, COP, EER, duct design, controls.

Context (from EEBC 2021):
{context}

Question: {question}

HVAC Expert Answer:""",

    "Service Water Heating Specialist": """\
You are the Section 7 SWH Specialist AI.
Answer questions ONLY about Section 7: Service Water Heating, piping insulation, heat traps.

Context (from EEBC 2021):
{context}

Question: {question}

SWH Expert Answer:""",

    "Electrical Power Specialist": """\
You are the Section 8 Electrical Power Specialist AI.
Answer questions ONLY about Section 8: Voltage drop, transformers, motors, metering.

Context (from EEBC 2021):
{context}

Question: {question}

Power Expert Answer:""",
}


def initialize_rag():
    global llm, vector_store, models, initialized

    if initialized:
        return True

    print("[DEBUG] Starting RAG initialization...")
    models = Models()
    embeddings = models.embeddings_hf
    llm = models.model_groq
    print("[INFO] Using HuggingFace embeddings + Groq LLM")

    try:
        vector_store = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory="./DB/chroma_langchain_db",
        )
        print("[DEBUG] Vector store initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector store: {e}")
        return False

    print("[SUCCESS] RAG system ready!")
    initialized = True
    return True


def refresh_vector_store():
    global vector_store, models, initialized
    print("[INFO] Refreshing vector store after new upload...")
    try:
        vector_store = Chroma(
            collection_name="documents",
            embedding_function=models.embeddings_hf,
            persist_directory="./DB/chroma_langchain_db",
        )
        print("[SUCCESS] Vector store refreshed")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to refresh vector store: {e}")
        initialized = False
        return False


def query_rag(question: str, agent_type: str = "EEBC Expert") -> str:
    global vector_store, llm, initialized

    if not initialized:
        if not initialize_rag():
            return "Failed to initialize RAG system"

    try:
        template = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["EEBC Expert"])
        prompt = ChatPromptTemplate.from_template(template)

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 40},
        )

        chain = (
            {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(question)
    except Exception as e:
        return f"Error processing query: {str(e)}"
