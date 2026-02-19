# app/runtime/embeddings.py
"""
Thin wrapper around OpenAI / Azure OpenAI embeddings API.

Returns a callable ``embed_fn(texts) -> List[List[float]]`` that can be
injected into RAGFiniteStateMachine for dense retrieval.
"""
from __future__ import annotations

import math
from typing import Callable, List

# Azure OpenAI has a per-request token limit; batching keeps us safe.
_BATCH_SIZE = 100


def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector for dot-product similarity."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        return vec
    return [v / norm for v in vec]


def get_embed_fn(
    model: str = "text-embedding-3-small",
    batch_size: int = _BATCH_SIZE,
) -> Callable[[List[str]], List[List[float]]]:
    """
    Factory that returns an embedding callable.

    The returned function signature is::

        embed_fn(texts: List[str]) -> List[List[float]]

    Automatically batches large inputs to avoid Azure token limits.
    Uses ``app.llm_client.get_client()`` so it inherits Azure / OpenAI
    credentials from the environment automatically.
    """
    from app.llm_client import get_client

    def _embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = get_client()
        all_vecs: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(input=batch, model=model)
            vecs = [item.embedding for item in response.data]
            all_vecs.extend(_normalize(v) for v in vecs)

        return all_vecs

    return _embed
