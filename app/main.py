"""API de inferência do Agenteresolve (Oracle).

Endpoints:
  GET  /health
  POST /v1/chat/completions   (LLM gateway: Ollama local → fallback OpenRouter)
  POST /ocr /caption /detect /transcribe /enhance /synthesize /embeddings /extract

Auth: Bearer INFERENCE_KEY (exceto /health).
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status

from . import __version__, providers
from .auth import require_key
from .config import get_settings
from .llm import LLMUnavailable, chat_completions
from .providers import Unavailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")


def _ensure_ollama_model() -> None:
    """Garante que o modelo local esteja no Ollama (pull no boot; não fatal)."""
    settings = get_settings()
    base = settings.ollama_url.rstrip("/")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{base}/api/tags")
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        if settings.llm_model not in names:
            logger.info("pulling ollama model %s ...", settings.llm_model)
            with httpx.Client(timeout=1800.0) as client:
                client.post(f"{base}/api/pull", json={"name": settings.llm_model})
    except Exception as exc:  # noqa: BLE001 - melhor esforço
        logger.warning("ollama model ensure skipped: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_ollama_model()
    yield


app = FastAPI(title="agenteresolve-inference", version=__version__, lifespan=lifespan)

# Limita concorrência de operações pesadas (CPU modesto no Oracle).
_sem = threading.Semaphore(max(1, get_settings().max_concurrency))


def _read(file: UploadFile) -> bytes:
    return file.file.read()


def _run(fn, *args, **kwargs):
    with _sem:
        return fn(*args, **kwargs)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_model": settings.llm_model,
        "ollama_url": settings.ollama_url,
        "fallback": bool(settings.openrouter_api_key),
        "auth": bool(settings.inference_key),
        "max_concurrency": settings.max_concurrency,
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
def llm(payload: dict = Body(...)) -> dict:
    try:
        return _run(chat_completions, payload)
    except LLMUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/ocr", dependencies=[Depends(require_key)])
def ocr(file: UploadFile = File(...), lang: str | None = Form(default=None)) -> dict:
    try:
        return _run(providers.ocr, _read(file), lang=lang)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/caption", dependencies=[Depends(require_key)])
def caption(file: UploadFile = File(...)) -> dict:
    try:
        return _run(providers.caption, _read(file))
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/detect", dependencies=[Depends(require_key)])
def detect(file: UploadFile = File(...), labels: str | None = Form(default=None)) -> dict:
    wanted = [label.strip() for label in labels.split(",") if label.strip()] if labels else None
    try:
        return _run(providers.detect, _read(file), labels=wanted)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/transcribe", dependencies=[Depends(require_key)])
def transcribe(file: UploadFile = File(...), lang: str | None = Form(default=None)) -> dict:
    try:
        return _run(providers.transcribe, _read(file), lang=lang)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/enhance", dependencies=[Depends(require_key)])
def enhance(file: UploadFile = File(...), mode: str = Form(default="denoise")) -> Response:
    try:
        content, media_type = _run(providers.enhance, _read(file), mode=mode)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return Response(content=content, media_type=media_type)


@app.post("/synthesize", dependencies=[Depends(require_key)])
def synthesize(payload: dict = Body(...)) -> Response:
    try:
        content, media_type = _run(
            providers.synthesize,
            str(payload.get("text", "")),
            voice=payload.get("voice"),
            lang=payload.get("lang"),
            speed=float(payload.get("speed", 1.0)),
        )
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return Response(content=content, media_type=media_type)


@app.post("/embeddings", dependencies=[Depends(require_key)])
def embeddings(payload: dict = Body(...)) -> dict:
    inputs = payload.get("input")
    texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
    if not texts:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "'input' is required")
    return _run(providers.embeddings, texts)


@app.post("/extract", dependencies=[Depends(require_key)])
def extract(file: UploadFile = File(...), lang: str | None = Form(default=None)) -> dict:
    content_type = file.content_type or "application/octet-stream"
    try:
        return _run(providers.extract, _read(file), content_type=content_type, lang=lang)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
