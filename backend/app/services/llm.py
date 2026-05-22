from openai import OpenAI, AzureOpenAI
from fastapi import HTTPException

from app.config import settings

# Lazy client — created on first use, not at import time
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    # Prefer Azure OpenAI if credentials are set
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        _client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        return _client

    # Fall back to regular OpenAI
    if settings.openai_api_key:
        _client = OpenAI(api_key=settings.openai_api_key)
        return _client

    raise HTTPException(
        status_code=500,
        detail="No LLM credentials set. Add AZURE_OPENAI_* or OPENAI_API_KEY to .env",
    )


def _get_model() -> str:
    """Return the deployment/model name depending on which client is active."""
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return settings.azure_openai_deployment
    return settings.llm_model


def ask_llm(messages: list[dict]) -> str:
    """
    Send messages to the OpenAI chat API and return the response text.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        The LLM's response as a string.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=_get_model(),
        messages=messages,
        temperature=0.3,  # low = more factual, less creative
    )

    return response.choices[0].message.content
