"""API de inferência do Agenteresolve (Oracle).

Endpoints:
  GET  /health
  POST /v1/chat/completions   (LLM gateway: Ollama local → fallback OpenRouter)
  POST /ocr /caption /detect /transcribe /enhance /synthesize /embeddings /extract

Auth: Bearer INFERENCE_KEY (exceto /health).
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
import wave
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
    """Garante que os modelos locais estejam no Ollama (pull no boot; não fatal)."""
    settings = get_settings()
    base = settings.ollama_url.rstrip("/")
    wanted = {m for m in (settings.llm_model, settings.translate_model) if m}
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{base}/api/tags")
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        for model in wanted - names:
            logger.info("pulling ollama model %s ...", model)
            with httpx.Client(timeout=3600.0) as client:
                client.post(f"{base}/api/pull", json={"name": model})
    except Exception as exc:  # noqa: BLE001 - melhor esforço
        logger.warning("ollama model ensure skipped: %s", exc)


# PNG 1x1 transparente: entrada mínima para carregar os modelos de visão no boot.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _silent_wav(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """WAV silencioso mínimo (mono, 16 bits) para aquecer o Whisper."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def _warmup() -> None:
    """Carrega os modelos locais no boot, antes de aceitar tráfego.

    Medido em 2026-10-17: a **primeira** chamada de ``tts`` depois de um deploy levou
    302 s (timeout no cliente) e a segunda, 7 s. O custo era do carregamento do modelo,
    não da síntese — e caía no colo de quem chegasse primeiro, inclusive da rotina
    diária, que então acusaria uma falha que não existe. Estes endpoints não têm
    fallback remoto, então aquecer no boot é a única forma de tirar o custo do caminho
    do usuário. Melhor esforço: falha de aquecimento não derruba o serviço.
    """
    settings = get_settings()
    if not settings.warmup:
        logger.info("warmup desligado (WARMUP=0)")
        return
    audio = _silent_wav()
    steps: tuple[tuple[str, object], ...] = (
        ("tts", lambda: providers.synthesize("ok")),
        ("embeddings", lambda: providers.embeddings(["ok"])),
        ("ocr", lambda: providers.ocr(_TINY_PNG)),
        ("caption", lambda: providers.caption(_TINY_PNG)),
        ("detect", lambda: providers.detect(_TINY_PNG)),
        ("transcribe", lambda: providers.transcribe(audio)),
    )
    for name, step in steps:
        started = time.monotonic()
        try:
            step()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - aquecer é melhor esforço
            logger.warning("warmup %s falhou: %s", name, exc)
        else:
            logger.info("warmup %s ok em %.1fs", name, time.monotonic() - started)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_ollama_model()
    _warmup()
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


# --- Aliases para os backends standalone (nomes legados) --------------------------


def _anonymize_image(data: bytes) -> bytes:
    from PIL import Image, ImageFilter

    image = Image.open(io.BytesIO(data)).convert("RGB")
    try:
        detections = providers.detect(data)
    except Unavailable:
        detections = {"detections": []}
    for det in detections.get("detections", []):
        if det.get("label") != "person":
            continue
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        box = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        image.paste(image.crop(box).filter(ImageFilter.GaussianBlur(15)), box)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@app.post("/tts", dependencies=[Depends(require_key)])
def tts_legacy(
    text: str = Form(...),
    voice: str | None = Form(default=None),
    lang: str | None = Form(default=None),
    speed: float = Form(default=1.0),
) -> Response:
    try:
        content, media_type = _run(providers.synthesize, text, voice=voice, lang=lang, speed=speed)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return Response(content=content, media_type=media_type)


@app.post("/alttext", dependencies=[Depends(require_key)])
def alttext_legacy(file: UploadFile = File(...)) -> dict:
    try:
        return _run(providers.caption, _read(file))
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/objectcount", dependencies=[Depends(require_key)])
def objectcount_legacy(
    file: UploadFile = File(...), labels: str | None = Form(default=None)
) -> dict:
    wanted = [label.strip() for label in labels.split(",") if label.strip()] if labels else None
    try:
        return _run(providers.detect, _read(file), labels=wanted)
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/chat", dependencies=[Depends(require_key)])
def chat_legacy(
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    content = text or ""
    if file is not None:
        try:
            content = _run(providers.transcribe, _read(file))["text"]
        except Unavailable as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    try:
        return _run(chat_completions, {"messages": [{"role": "user", "content": content}]})
    except LLMUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@app.post("/anonymize", dependencies=[Depends(require_key)])
def anonymize(file: UploadFile = File(...)) -> Response:
    try:
        content = _run(_anonymize_image, _read(file))
    except Unavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return Response(content=content, media_type="image/png")
