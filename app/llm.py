"""Gateway LLM OpenAI-compatible: tenta o Ollama local e cai no OpenRouter (híbrido).

O contrato é o mesmo consumido por ``OpenAICompatibleLLM`` no código existente:
``POST {base}/chat/completions`` → resposta OpenAI (``choices[0].message.content``).
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import get_settings


class LLMUnavailable(RuntimeError):
    pass


def _ollama_chat(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    url = f"{settings.ollama_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=settings.llm_timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def _openrouter_chat(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMUnavailable("openrouter not configured")
    body = dict(payload)
    body["model"] = settings.openrouter_model
    url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=settings.llm_timeout) as client:
        response = client.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()


def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    """Local (Ollama) primeiro; fallback OpenRouter em erro/timeout."""
    settings = get_settings()
    # Garante um modelo no payload local (o cliente pode mandar o dele).
    local_payload = dict(payload)
    local_payload.setdefault("model", settings.llm_model)
    try:
        return _ollama_chat(local_payload)
    except Exception as local_exc:  # noqa: BLE001 - cai no fallback
        try:
            return _openrouter_chat(payload)
        except Exception as remote_exc:  # noqa: BLE001 - reporta ambos
            raise LLMUnavailable(
                f"local LLM failed ({local_exc!r}); fallback failed ({remote_exc!r})"
            ) from remote_exc
