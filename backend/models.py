"""
Models Module
=============
Centralised initialisation of all AI models used across the system.
  - Groq LLM  (LLaMA-3 70B via Groq API)
  - HuggingFace Embeddings  (all-MiniLM-L6-v2)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class Models:
    """
    Singleton-style model container.
    Import once, reuse everywhere.
    """

    _embeddings_instance = None  # class-level cache to avoid reloading HF model

    def __init__(self):
        # ── Groq LLM ──────────────────────────────────────────────────────────
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )

        self.model_groq = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=groq_api_key,
            temperature=0.1,          # low temp → deterministic code answers
            max_tokens=4096,
        )

        # ── HuggingFace Embeddings ─────────────────────────────────────────────
        # Cache at class level so only one download/load happens per process
        if Models._embeddings_instance is None:
            Models._embeddings_instance = HuggingFaceEmbeddings(
                model_name=os.getenv(
                    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self.embeddings_hf = Models._embeddings_instance
