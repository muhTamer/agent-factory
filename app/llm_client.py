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
    if endpoint and api_key:
        # Azure mode
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    elif api_key:
        # Regular OpenAI mode
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        raise RuntimeError("No OpenAI or Azure OpenAI credentials found.")


def chat_json(messages, model=None, temperature=1.0):
    """
    Send a chat completion request and expect JSON response.
    Returns a Python dict.
    """
    client = get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model or "gpt-5-mini"

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    msg = response.choices[0].message.content
    try:
        return json.loads(msg) if msg else {}
    except json.JSONDecodeError:
        return {"raw": msg}
