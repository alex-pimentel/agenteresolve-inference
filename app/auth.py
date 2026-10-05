"""Autenticação por Bearer (INFERENCE_KEY).

Desenhado como dependência plugável: hoje compara com uma chave única; no futuro pode
delegar ao sistema de chaves de API por cliente (BYOK) sem mudar os endpoints.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from .config import get_settings


def require_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = settings.inference_key
    if not expected:
        # Sem chave configurada: modo dev (não recomendado em produção).
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
