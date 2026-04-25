# app/llm_client.py
"""
Unified LLM client for Azure OpenAI or OpenAI API.
Usage:
    from app.llm_client import get_client, chat_json, LLM_MODEL, LLM_TEMPERATURE
"""

import os
import json
import time
import logging
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

# Load .env automatically
load_dotenv()

# ── Global LLM configuration ─────────────────────────────────────────
# All components should import these instead of hardcoding values.
LLM_MODEL: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "1.0"))

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


_llm_log = logging.getLogger("llm_perf")

# Auto-incrementing call counter per request for waterfall tracing
_call_seq = 0


def chat_json(messages, model=None, temperature=None, timeout=None, max_tokens=None):
    """
    Send a chat completion request and expect JSON response.
    Returns a Python dict.

    ``model`` defaults to :data:`LLM_MODEL`.
    ``temperature`` defaults to :data:`LLM_TEMPERATURE` (0.2).
    ``timeout`` overrides the client-level default for this single call
    (useful for heavy generation steps like blueprint planning).
    ``max_tokens`` caps generation length — set this to reduce latency
    on calls that only need short responses (e.g. classifiers).
    """
    global _call_seq
    _call_seq += 1
    seq = _call_seq

    client = get_client()
    deployment = model or LLM_MODEL
    if temperature is None:
        temperature = LLM_TEMPERATURE

    # Identify caller for the trace log
    import traceback

    caller = "unknown"
    for frame in traceback.extract_stack()[-3:-1]:
        caller = f"{os.path.basename(frame.filename)}:{frame.lineno}"

    sys_preview = ""
    if messages and messages[0].get("role") == "system":
        sys_preview = messages[0]["content"][:80].replace("\n", " ")

    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    _llm_log.info(
        "[LLM #%d] START  caller=%s model=%s prompt=%d chars sys='%s...'",
        seq,
        caller,
        deployment,
        prompt_chars,
        sys_preview,
    )
    t0 = time.time()

    create_kwargs = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        exc_str = str(exc).lower()
        # Some models (o-series, reasoning models) reject non-default temperature.
        # Retry once with temperature removed if the error mentions it.
        if "temperature" in exc_str and temperature != 1.0:
            create_kwargs.pop("temperature", None)
            response = client.chat.completions.create(**create_kwargs)
        # Some models/API versions reject max_tokens (want max_completion_tokens).
        elif "max_tokens" in exc_str or "max_completion_tokens" in exc_str:
            create_kwargs.pop("max_tokens", None)
            response = client.chat.completions.create(**create_kwargs)
        else:
            elapsed = time.time() - t0
            _llm_log.warning(
                "[LLM #%d] FAIL   %.1fs caller=%s err=%s", seq, elapsed, caller, exc
            )
            raise

    elapsed = time.time() - t0
    usage = response.usage
    _llm_log.info(
        "[LLM #%d] DONE   %.1fs caller=%s tokens=%s/%s",
        seq,
        elapsed,
        caller,
        usage.prompt_tokens if usage else "?",
        usage.completion_tokens if usage else "?",
    )

    msg = response.choices[0].message.content
    try:
        return json.loads(msg) if msg else {}
    except json.JSONDecodeError:
        return {"raw": msg}
