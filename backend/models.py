"""
Models Module
=============
Centralised initialisation of models used by the system.

Render's free/low-memory instances cannot reliably hold PyTorch +
sentence-transformers in 512 MB RAM. The default embedding provider is
therefore a small deterministic hashing embedder. Set
EMBEDDING_PROVIDER=huggingface locally if you want MiniLM embeddings.
"""

import hashlib
import math
import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

load_dotenv()


def get_embedding_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower()


def get_vector_collection_name() -> str:
    configured = os.getenv("CHROMA_COLLECTION", "").strip()
    if configured:
        return configured
    return "documents_hf" if get_embedding_provider() == "huggingface" else "documents_hash_v1"


class HashingEmbeddings(Embeddings):
    """Low-memory embedding function compatible with Chroma/LangChain."""

    _token_re = re.compile(r"[a-z0-9]+")

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        tokens = self._token_re.findall(text.lower())
        vector = [0.0] * self.dimensions

        for token in tokens:
            self._add_token(vector, token, 1.0)

        for left, right in zip(tokens, tokens[1:]):
            self._add_token(vector, f"{left}_{right}", 0.5)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def _add_token(self, vector: List[float], token: str, weight: float) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % self.dimensions
        sign = 1.0 if number & 1 else -1.0
        vector[index] += sign * weight


class Models:
    """Singleton-style model container."""

    _embeddings_instance = None
    _llm_instance = None

    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )

        if Models._llm_instance is None:
            Models._llm_instance = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                api_key=groq_api_key,
                temperature=0.1,
                max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048")),
            )
        self.model_groq = Models._llm_instance

        if Models._embeddings_instance is None:
            provider = get_embedding_provider()
            if provider == "huggingface":
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError as exc:
                    raise ImportError(
                        "EMBEDDING_PROVIDER=huggingface requires langchain-huggingface "
                        "and sentence-transformers to be installed."
                    ) from exc

                Models._embeddings_instance = HuggingFaceEmbeddings(
                    model_name=os.getenv(
                        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                    ),
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            else:
                dimensions = int(os.getenv("HASH_EMBEDDING_DIM", "384"))
                Models._embeddings_instance = HashingEmbeddings(dimensions=dimensions)

        self.embeddings_hf = Models._embeddings_instance
        self.collection_name = get_vector_collection_name()
