# app/llm_client.py
"""
Unified LLM client for Azure OpenAI or OpenAI API.
Usage:
    from app.llm_client import get_client, chat_json
"""

import os
import json
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

# Load .env automatically
load_dotenv()


def get_client():
    """Return an OpenAI client object (Azure or regular)."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    if endpoint and api_key:
        # Azure mode
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        return AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version, timeout=timeout
        )
    elif api_key:
        # Regular OpenAI mode
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    else:
        raise RuntimeError("No OpenAI or Azure OpenAI credentials found.")


def chat_json(messages, model=None, temperature=1.0, timeout=None):
    """
    Send a chat completion request and expect JSON response.
    Returns a Python dict.

    ``timeout`` overrides the client-level default for this single call
    (useful for heavy generation steps like blueprint planning).
    """
    client = get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model or "gpt-5-mini"

    create_kwargs = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    if timeout is not None:
        create_kwargs["timeout"] = timeout

    response = client.chat.completions.create(**create_kwargs)
    msg = response.choices[0].message.content
    try:
        return json.loads(msg) if msg else {}
    except json.JSONDecodeError:
        return {"raw": msg}
