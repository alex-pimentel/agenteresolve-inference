"""Gateway LLM OpenAI-compatible: tenta o Ollama local e cai no OpenRouter (híbrido).

O contrato é o mesmo consumido por ``OpenAICompatibleLLM`` no código existente:
``POST {base}/chat/completions`` → resposta OpenAI (``choices[0].message.content``).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .config import get_settings


class LLMUnavailable(RuntimeError):
    pass


# Cache curto da lista de modelos do Ollama: evita um /api/tags por requisição, mas
# ainda assimila um ``ollama pull`` feito logo depois do boot.
_MODELS_TTL = 60.0
_models_cache: tuple[float, set[str]] | None = None


def _ollama_models() -> set[str]:
    global _models_cache
    now = time.monotonic()
    if _models_cache is not None and now - _models_cache[0] < _MODELS_TTL:
        return _models_cache[1]
    settings = get_settings()
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            names = {str(m.get("name")) for m in response.json().get("models", []) if m.get("name")}
    except Exception:  # noqa: BLE001 - sem a lista, caímos no modelo padrão
        return set()
    _models_cache = (now, names)
    return names


def _local_model(requested: Any) -> str:
    """Modelo a rodar no Ollama local.

    O cliente fala o nome "de plataforma" (ex.: ``openai/gpt-4o-mini``), que não
    existe no Ollama e devolveria 404. Só honramos o modelo pedido quando o Ollama
    realmente o tem (ex.: o modelo dedicado de tradução); caso contrário usamos o
    configurado em ``LLM_MODEL``.
    """
    settings = get_settings()
    default = settings.llm_model
    name = str(requested) if requested else ""
    if not name or name == default:
        return default
    return name if name in _ollama_models() else default


def _ollama_chat(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    url = f"{settings.ollama_url.rstrip('/')}/v1/chat/completions"
    body = dict(payload)
    body["model"] = _local_model(body.get("model"))
    # Modelos de raciocínio (ex.: qwen3.5) gastam o orçamento inteiro "pensando" e
    # devolvem ``content`` vazio. O ``think=false`` (native do Ollama) é ignorado no
    # endpoint OpenAI-compatible; o que funciona é ``reasoning_effort``.
    body.pop("think", None)
    body.setdefault("reasoning_effort", "none")
    with httpx.Client(timeout=settings.llm_timeout) as client:
        response = client.post(url, json=body)
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
    """Local (Ollama) primeiro; fallback OpenRouter em erro/timeout.

    O modelo pedido pelo cliente é resolvido para um modelo local disponível (ver
    :func:`_local_model`); o ``payload`` original segue intacto para o fallback.
    """
    try:
        return _ollama_chat(payload)
    except Exception as local_exc:  # noqa: BLE001 - cai no fallback
        try:
            return _openrouter_chat(payload)
        except Exception as remote_exc:  # noqa: BLE001 - reporta ambos
            raise LLMUnavailable(
                f"local LLM failed ({local_exc!r}); fallback failed ({remote_exc!r})"
            ) from remote_exc
