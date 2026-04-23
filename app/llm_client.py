# app/llm_client.py
"""
Unified LLM client for Azure OpenAI or OpenAI API.
Usage:
    from app.llm_client import get_client, chat_json, LLM_MODEL, LLM_TEMPERATURE
"""

import os
import json
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

# Load .env automatically
load_dotenv()

# ── Global LLM configuration ─────────────────────────────────────────
# All components should import these instead of hardcoding values.
LLM_MODEL: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

_cached_client = None


def get_client():
    """Return a cached OpenAI client object (Azure or regular).

    The client is created once and reused across all calls so HTTP
    connections are pooled (TLS handshake + DNS lookup happen only once).
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    if endpoint and api_key:
        # Azure mode
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        _cached_client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
        )
    elif api_key:
        # Regular OpenAI mode
        base_url = os.getenv("OPENAI_BASE_URL")
        _cached_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    else:
        raise RuntimeError("No OpenAI or Azure OpenAI credentials found.")
    return _cached_client


def chat_json(messages, model=None, temperature=None, timeout=None):
    """
    Send a chat completion request and expect JSON response.
    Returns a Python dict.

    ``model`` defaults to :data:`LLM_MODEL`.
    ``temperature`` defaults to :data:`LLM_TEMPERATURE` (0.2).
    ``timeout`` overrides the client-level default for this single call
    (useful for heavy generation steps like blueprint planning).
    """
    client = get_client()
    deployment = model or LLM_MODEL
    if temperature is None:
        temperature = LLM_TEMPERATURE

    create_kwargs = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        # Some models (o-series, reasoning models) reject non-default temperature.
        # Retry once with temperature removed if the error mentions it.
        if "temperature" in str(exc).lower() and temperature != 1.0:
            create_kwargs.pop("temperature", None)
            response = client.chat.completions.create(**create_kwargs)
        else:
            raise

    msg = response.choices[0].message.content
    try:
        return json.loads(msg) if msg else {}
    except json.JSONDecodeError:
        return {"raw": msg}
