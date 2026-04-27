"""
Orchestrator Agent — The Brain of the Agentic RAG System
=========================================================
Responsibilities:
  1. Analyse the user's intent using the LLM
  2. Decide which specialist agent(s) to call and in what order
  3. Optionally run multiple agents and synthesise a combined answer
  4. Maintain a short-term conversation memory within a session
  5. Return a rich AgentResponse with full trace / audit information

Agents:
  - Compliance Checker   : COMPLIANT / NON-COMPLIANT analysis
  - ETTV/RTTV Calculator : Full thermal transfer value calculations
  - Solution Advisor     : Corrective actions for non-compliant items
  - Envelope Specialist  : Section 4 – walls, roofs, fenestration
  - Lighting Specialist  : Section 5 – LPD, controls, daylighting
  - HVAC Specialist      : Section 6 – MVAC systems & efficiency
  - SWH Specialist       : Section 7 – service water heating
  - Electrical Specialist: Section 8 – voltage drop, transformers
  - EEBC Expert          : General fallback for any EEBC question
"""

import json
import re
from datetime import datetime
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from rag_core import query_rag, initialize_rag

# ── Agent registry ─────────────────────────────────────────────────────────────
AGENT_REGISTRY = {
    "Compliance Checker": {
        "description": (
            "Checks if building parameters are COMPLIANT or NON-COMPLIANT with EEBC 2021. "
            "Cites the specific clause that determines status."
        ),
        "keywords": [
            "compliant", "compliance", "comply", "violation", "meet the requirement",
            "pass", "fail", "allowed", "permitted", "exceed", "within limit",
            "non-compliant", "check compliance", "verify",
        ],
    },
    "ETTV/RTTV Calculator": {
        "description": (
            "Dedicated agent for Envelope Thermal Transfer Value (ETTV) and "
            "Roof Thermal Transfer Value (RTTV) calculations. Applies the official "
            "EEBC 2021 formulas, extracts U-values, SC values and area data, "
            "and returns step-by-step numeric results."
        ),
        "keywords": [
            "ettv", "rttv", "thermal transfer value", "envelope thermal",
            "roof thermal", "u-value", "u value", "shading coefficient",
            "sc value", "sc=", "wall area", "window area", "roof area",
            "heat gain", "thermal performance", "eetv", "eer",
            "calculate ettv", "calculate rttv", "compute ettv", "compute rttv",
            "formula", "coefficient", "w/m2",
        ],
    },
    "Solution Advisor": {
        "description": (
            "Recommends corrective actions and energy-efficient improvements "
            "for non-compliant features. Includes qualitative ROI estimates."
        ),
        "keywords": [
            "improve", "recommendation", "solution", "fix", "correct", "better",
            "upgrade", "retrofit", "energy saving", "roi", "return on investment",
            "what should", "how can", "suggestion", "remediate", "address",
        ],
    },
    "Envelope Specialist": {
        "description": "Expert on Section 4: walls, roofs, fenestration, insulation, opaque areas.",
        "keywords": [
            "wall", "roof", "window", "fenestration", "insulation", "opaque",
            "glazing", "facade", "envelope", "u value", "section 4",
            "external wall", "external roof",
        ],
    },
    "Lighting Specialist": {
        "description": "Expert on Section 5: LPD, controls, daylighting, efficacy.",
        "keywords": [
            "lighting", "lpd", "lux", "lumens", "daylight", "lamp", "fixture",
            "control", "dimming", "occupancy sensor", "section 5", "efficacy",
            "lighting power density",
        ],
    },
    "HVAC Specialist": {
        "description": "Expert on Section 6: MVAC systems, equipment efficiency, controls.",
        "keywords": [
            "hvac", "air conditioning", "ventilation", "mvac", "cop", "eer",
            "chiller", "ahu", "fan", "duct", "cooling load", "section 6",
            "air-conditioning", "refrigerant",
        ],
    },
    "Service Water Heating Specialist": {
        "description": "Expert on Section 7: water heating equipment, piping insulation.",
        "keywords": [
            "water heater", "hot water", "swh", "heat trap", "piping insulation",
            "solar water", "heat pump water", "section 7", "water heating",
        ],
    },
    "Electrical Power Specialist": {
        "description": "Expert on Section 8: voltage drop, transformers, motors, metering.",
        "keywords": [
            "electrical", "voltage drop", "transformer", "motor", "metering",
            "power factor", "wiring", "cable", "section 8", "switchboard",
        ],
    },
    "EEBC Expert": {
        "description": "General EEBC 2021 questions that don't fit a specialist category.",
        "keywords": [],  # fallback — always matches last
    },
}

# ── Routing system prompt ──────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """You are an intelligent routing orchestrator for an EEBC 2021 AI compliance system.

Your job: read the user's question and decide which specialist agent(s) should answer it.

Available agents:
{agent_descriptions}

Rules:
- Return ONLY a valid JSON object — no explanation, no markdown.
- Format: {{"agents": ["Agent Name 1", "Agent Name 2"], "reasoning": "short reason", "multi_agent": true/false}}
- Choose multiple agents ONLY when the question genuinely spans 2+ domains.
- Always include at least one agent.
- Use "EEBC Expert" as the fallback for general questions.
- For ETTV/RTTV calculations always choose "ETTV/RTTV Calculator".
- For compliance checking that is non-compliant, also include "Solution Advisor".
- "multi_agent" must be true if you chose more than one agent, false otherwise.
"""

# ── Conversation memory ────────────────────────────────────────────────────────
class ConversationMemory:
    """Simple in-process conversation history (per session)."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content, "ts": datetime.now().isoformat()})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_context_string(self) -> str:
        if not self.history:
            return ""
        lines = []
        for h in self.history[-6:]:
            prefix = "User" if h["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {h['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history = []


# ── Orchestrator ───────────────────────────────────────────────────────────────
class OrchestratorAgent:
    """
    Central agent that:
      - routes questions to the right specialist(s)
      - can call multiple agents and synthesise results
      - tracks conversation memory
    """

    def __init__(self):
        from models import Models
        self._models = Models()
        self._llm = self._models.model_groq
        self._memory = ConversationMemory()
        initialize_rag()
        print("[Orchestrator] ✅ Initialized")

    def _build_agent_descriptions(self) -> str:
        return "\n".join(
            f"- {name}: {meta['description']}"
            for name, meta in AGENT_REGISTRY.items()
        )

    def _keyword_route(self, question: str) -> list[str]:
        q_lower = question.lower()
        matched = []
        for agent_name, meta in AGENT_REGISTRY.items():
            if agent_name == "EEBC Expert":
                continue
            if any(kw in q_lower for kw in meta["keywords"]):
                matched.append(agent_name)
        return matched if matched else ["EEBC Expert"]

    def _llm_route(self, question: str, conversation_context: str) -> dict:
        system = ROUTER_SYSTEM_PROMPT.format(
            agent_descriptions=self._build_agent_descriptions()
        )
        user_content = question
        if conversation_context:
            user_content = f"[Recent conversation]\n{conversation_context}\n\n[Current question]\n{question}"

        try:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=user_content),
            ]
            response = self._llm.invoke(messages)
            raw = re.sub(r"```json|```", "", response.content.strip()).strip()
            result = json.loads(raw)

            valid_agents = [a for a in result.get("agents", []) if a in AGENT_REGISTRY]
            if not valid_agents:
                valid_agents = self._keyword_route(question)

            return {
                "agents": valid_agents,
                "reasoning": result.get("reasoning", "LLM routing"),
                "multi_agent": len(valid_agents) > 1,
                "method": "llm",
            }
        except Exception as e:
            print(f"[Orchestrator] LLM routing failed ({e}), using keyword fallback")
            agents = self._keyword_route(question)
            return {
                "agents": agents,
                "reasoning": "Keyword-based routing (LLM fallback)",
                "multi_agent": len(agents) > 1,
                "method": "keyword",
            }

    def _call_agent(self, agent_name: str, question: str) -> str:
        print(f"[Orchestrator] 🤖 Calling agent: {agent_name}")
        return query_rag(question, agent_name)

    def _synthesise(self, question: str, agent_responses: dict) -> str:
        if len(agent_responses) == 1:
            return list(agent_responses.values())[0]

        combined = "\n\n".join(
            f"=== {agent} ===\n{answer}"
            for agent, answer in agent_responses.items()
        )

        synthesis_prompt = f"""You are a senior EEBC 2021 expert synthesising answers from multiple specialist agents.

Merge their responses into ONE clear, non-repetitive professional answer.
Do not mention the agent names in the final answer.

User question: {question}

Specialist responses:
{combined}

Synthesised answer:"""

        try:
            response = self._llm.invoke([HumanMessage(content=synthesis_prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"[Orchestrator] Synthesis failed ({e}), returning concatenated answers")
            return combined

    def run(self, question: str, session_id: Optional[str] = None) -> dict:
        print(f"\n[Orchestrator] 📩 Question: {question[:80]}...")

        context = self._memory.get_context_string()
        routing = self._llm_route(question, context)
        agents_to_call = routing["agents"]
        print(f"[Orchestrator] 🗺️  Routing → {agents_to_call} (via {routing['method']})")

        agent_responses = {}
        for agent_name in agents_to_call:
            enriched_q = question
            if context:
                enriched_q = (
                    f"[Context from previous conversation]\n{context}\n\n"
                    f"[Current question]\n{question}"
                )
            agent_responses[agent_name] = self._call_agent(agent_name, enriched_q)

        final_answer = self._synthesise(question, agent_responses)

        self._memory.add("user", question)
        self._memory.add("assistant", final_answer)

        return {
            "answer": final_answer,
            "agents_used": agents_to_call,
            "routing_method": routing["method"],
            "routing_reasoning": routing["reasoning"],
            "multi_agent": routing["multi_agent"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def clear_memory(self):
        self._memory.clear()
        print("[Orchestrator] 🗑️  Memory cleared")


# ── Singleton ──────────────────────────────────────────────────────────────────
_orchestrator_instance: Optional[OrchestratorAgent] = None


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance
