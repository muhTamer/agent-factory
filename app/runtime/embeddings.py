# app/runtime/embeddings.py
"""
Thin wrapper around OpenAI / Azure OpenAI embeddings API.

Returns a callable ``embed_fn(texts) -> List[List[float]]`` that can be
injected into DomainAgentEngine / RAGFiniteStateMachine for dense retrieval.

Supports a **separate** Azure OpenAI resource for embeddings via:
  AZURE_OPENAI_EMBEDDING_ENDPOINT   (falls back to AZURE_OPENAI_ENDPOINT)
  AZURE_OPENAI_EMBEDDING_KEY        (falls back to AZURE_OPENAI_API_KEY)
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT (falls back to model param)
  AZURE_OPENAI_EMBEDDING_API_VERSION (falls back to AZURE_OPENAI_API_VERSION)
"""
from __future__ import annotations

import logging
import math
import os
from typing import Callable, List

log = logging.getLogger(__name__)

# Azure OpenAI has a per-request token limit; batching keeps us safe.
_BATCH_SIZE = 100


def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector for dot-product similarity."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        return vec
    return [v / norm for v in vec]


def _get_embedding_client():
    """
    Return an OpenAI client configured for the embedding endpoint.

    Checks embedding-specific env vars first, then falls back to the
    main Azure/OpenAI credentials from ``app.llm_client``.
    """
    embed_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    embed_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")

    # If embedding-specific endpoint is set, create a dedicated client
    if embed_endpoint:
        from openai import AzureOpenAI

        api_key = embed_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv(
            "AZURE_OPENAI_EMBEDDING_API_VERSION",
            os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        log.info(
            "Using dedicated embedding endpoint: %s (api_version=%s)",
            embed_endpoint,
            api_version,
        )
        return AzureOpenAI(
            azure_endpoint=embed_endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
        )

    # Fall back to the shared client (same resource as chat completions)
    from app.llm_client import get_client

    return get_client()


def get_embed_fn(
    model: str = "text-embedding-3-small",
    batch_size: int = _BATCH_SIZE,
) -> Callable[[List[str]], List[List[float]]]:
    """
    Factory that returns an embedding callable.

    The returned function signature is::

        embed_fn(texts: List[str]) -> List[List[float]]

    Automatically batches large inputs to avoid Azure token limits.

    Deployment resolution order:
      1. AZURE_OPENAI_EMBEDDING_DEPLOYMENT env var
      2. ``model`` parameter (default: "text-embedding-3-small")
    """
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or model

    def _embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = _get_embedding_client()
        all_vecs: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(input=batch, model=deployment)
            vecs = [item.embedding for item in response.data]
            all_vecs.extend(_normalize(v) for v in vecs)

        return all_vecs

    return _embed
