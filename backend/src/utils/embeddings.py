"""Embeddings through NVIDIA NIM's OpenAI-compatible /embeddings endpoint.

This is the only embedding provider. It uses the same API key and base URL as the LLM, and serves every
feature that embeds text (menu semantic search, BI example retrieval, seeding scripts).

Duck-typed like LangChain embeddings: embed_query(text) and embed_documents(texts).
NIM retrieval models are asymmetric, so queries and documents are sent with different `input_type`s.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


class NimEmbeddings:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        expected_dim: Optional[int] = None,
        allow_truncate: bool = False,
        batch_size: int = 32,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        client: Any = None,
    ):
        self.model = model
        self.dimensions = dimensions          # asked for in the request; dropped if the model rejects it
        self.expected_dim = expected_dim      # what the database column holds
        self.allow_truncate = allow_truncate  # only safe for Matryoshka-trained models
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.client = client

    # ── public API ────────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(list(texts), "passage")

    # ── internals ─────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._call(texts[start:start + self.batch_size], input_type))
        return [self._fit(v) for v in vectors]

    def _call(self, chunk: List[str], input_type: str) -> List[List[float]]:
        attempt = 0
        while True:
            kwargs = {
                "model": self.model,
                "input": chunk,
                "encoding_format": "float",
                "extra_body": {"input_type": input_type, "truncate": "END"},
            }
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            try:
                response = self.client.embeddings.create(**kwargs)
                break
            except Exception as e:
                if self.dimensions and "dimension" in str(e).lower():
                    logger.warning(f"{self.model} does not accept a 'dimensions' parameter; using its native size")
                    self.dimensions = None
                    continue
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        return [list(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]

    def _fit(self, vector: List[float]) -> List[float]:
        n = self.expected_dim
        if not n or len(vector) == n:
            return vector
        if len(vector) > n and self.allow_truncate:
            vector = vector[:n]
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            return [x / norm for x in vector]
        raise ValueError(
            f"Embedding model '{self.model}' returned {len(vector)} dimensions but the database expects {n}. "
            "Choose a model and size that match (EMBEDDING_NIM_MODEL / EMBEDDING_DIM) or change the vector size "
            "in supabase_db/schema.sql and re-embed everything."
        )
