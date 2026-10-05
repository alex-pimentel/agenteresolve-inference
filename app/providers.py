"""Capacidades locais (CPU) com import tardio.

Cada dependência pesada é importada dentro da função; se faltar (wheel ausente na
arquitetura, modelo não baixado), levanta :class:`Unavailable` → HTTP 503, sem derrubar
o serviço no boot.
"""

from __future__ import annotations

import io
import threading
from typing import Any

from .config import get_settings


class Unavailable(RuntimeError):
    """Capacidade sem backend disponível."""


_lock = threading.Lock()

# --- OCR --------------------------------------------------------------------------


def ocr(data: bytes, *, lang: str | None = None) -> dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise Unavailable(f"OCR dependency missing: {exc}") from exc

    settings = get_settings()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    raw = pytesseract.image_to_data(
        image, lang=lang or settings.tesseract_lang, output_type=pytesseract.Output.DICT
    )
    blocks: list[dict[str, Any]] = []
    for index, text in enumerate(raw.get("text", [])):
        if not str(text).strip():
            continue
        left, top = float(raw["left"][index]), float(raw["top"][index])
        width, height = float(raw["width"][index]), float(raw["height"][index])
        blocks.append(
            {
                "text": str(text),
                "bbox": [left, top, left + width, top + height],
                "confidence": float(raw["conf"][index]) / 100.0,
            }
        )
    return {"blocks": blocks}


# --- Vision -----------------------------------------------------------------------

_caption_pipe: Any = None
_yolo: Any = None


def caption(data: bytes) -> dict[str, Any]:
    global _caption_pipe
    try:
        from PIL import Image
        from transformers import pipeline
    except ImportError as exc:
        raise Unavailable(f"caption dependency missing: {exc}") from exc

    settings = get_settings()
    with _lock:
        if _caption_pipe is None:
            _caption_pipe = pipeline("image-to-text", model=settings.caption_model)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    output = _caption_pipe(image)
    if isinstance(output, list) and output:
        return {"caption": str(output[0].get("generated_text", "")).strip()}
    return {"caption": str(output).strip()}


def detect(data: bytes, *, labels: list[str] | None = None) -> dict[str, Any]:
    global _yolo
    try:
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise Unavailable(f"detect dependency missing: {exc}") from exc

    settings = get_settings()
    with _lock:
        if _yolo is None:
            _yolo = YOLO(settings.detect_model)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    result = _yolo(np.array(image), verbose=False)[0]
    names = result.names
    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        label = str(names[int(box.cls)])
        if labels and label not in labels:
            continue
        detections.append(
            {
                "label": label,
                "bbox": [float(v) for v in box.xyxy[0].tolist()],
                "score": float(box.conf),
            }
        )
    return {"detections": detections}


# --- Audio ------------------------------------------------------------------------

_whisper: Any = None


def transcribe(data: bytes, *, lang: str | None = None) -> dict[str, Any]:
    global _whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise Unavailable(f"transcribe dependency missing: {exc}") from exc

    settings = get_settings()
    with _lock:
        if _whisper is None:
            _whisper = WhisperModel(
                settings.whisper_model, device="cpu", compute_type=settings.whisper_compute_type
            )
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "audio")
        with open(path, "wb") as handle:
            handle.write(data)
        segments, info = _whisper.transcribe(path, language=lang)
        collected = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments
        ]
    return {
        "text": " ".join(s["text"] for s in collected).strip(),
        "segments": collected,
        "language": info.language,
    }


def enhance(data: bytes, *, mode: str = "denoise") -> tuple[bytes, str]:
    import os
    import shutil
    import subprocess  # nosec B404 - argv fixo, sem shell
    import tempfile

    filters = {
        "denoise": "highpass=f=80,afftdn,lowpass=f=8000",
        "voice-isolate": "highpass=f=100,afftdn,lowpass=f=6000",
        "normalize": "loudnorm=I=-16:TP=-1.5:LRA=11",
    }
    if mode not in filters:
        raise Unavailable(f"unknown enhance mode: {mode!r}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise Unavailable("ffmpeg not installed")
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        with open(in_path, "wb") as handle:
            handle.write(data)
        try:
            subprocess.run(  # nosec B603
                [ffmpeg, "-y", "-i", in_path, "-af", filters[mode], out_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise Unavailable(f"ffmpeg failed: {exc}") from exc
        with open(out_path, "rb") as handle:
            return handle.read(), "audio/wav"


# --- TTS --------------------------------------------------------------------------


def synthesize(text: str, *, voice: str | None = None, lang: str | None = None, speed: float = 1.0) -> tuple[bytes, str]:
    try:
        import piper  # type: ignore
    except ImportError as exc:
        raise Unavailable(f"tts dependency missing: {exc}") from exc

    settings = get_settings()
    voice_name = voice or settings.piper_voice
    try:
        piper_voice = piper.PiperVoice.load(voice_name)
        stream = piper_voice.synthesize(text, length_scale=1.0 / max(speed, 0.1))
        buffer = io.BytesIO()
        for chunk in stream:
            buffer.write(chunk.audio_int16_bytes)
        return buffer.getvalue(), "audio/wav"
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(f"piper synthesis failed for {voice_name!r}: {exc}") from exc


# --- Embeddings -------------------------------------------------------------------

_embedder: Any = None
_DIM = 256


def _hashing_embeddings(texts: list[str]) -> list[list[float]]:
    import hashlib
    import math

    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * _DIM
        for token in text.lower().split():
            digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:2], "big") % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


def embeddings(texts: list[str]) -> dict[str, Any]:
    global _embedder
    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {"data": [{"embedding": v} for v in _hashing_embeddings(texts)]}

    with _lock:
        if _embedder is None:
            _embedder = SentenceTransformer(settings.embedding_model)
    vectors = _embedder.encode(texts, normalize_embeddings=True)
    return {"data": [{"embedding": [float(x) for x in vec]} for vec in vectors]}


# --- Document extraction ----------------------------------------------------------


def _pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None  # type: ignore
    text = ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception:  # noqa: BLE001
            text = ""
    if not text.strip():
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:  # noqa: BLE001
            text = ""
    return text.strip()


def _docx_text(data: bytes) -> str:
    try:
        import docx
    except ImportError as exc:
        raise Unavailable(f"docx dependency missing: {exc}") from exc
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs).strip()


def _pptx_text(data: bytes) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise Unavailable(f"pptx dependency missing: {exc}") from exc
    presentation = Presentation(io.BytesIO(data))
    lines: list[str] = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                lines.append(shape.text)
    return "\n".join(lines).strip()


def _xlsx_text(data: bytes) -> str:
    try:
        import openpyxl
    except ImportError as exc:
        raise Unavailable(f"xlsx dependency missing: {exc}") from exc
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                lines.append("\t".join(cells))
    return "\n".join(lines).strip()


def extract(data: bytes, *, content_type: str, lang: str | None = None) -> dict[str, Any]:
    ctype = (content_type or "").lower()
    text = ""
    used_ocr = False
    if "pdf" in ctype:
        text = _pdf_text(data)
    elif "word" in ctype or "docx" in ctype or "wordprocessingml" in ctype:
        text = _docx_text(data)
    elif "presentation" in ctype or "pptx" in ctype:
        text = _pptx_text(data)
    elif "spreadsheet" in ctype or "xlsx" in ctype or "excel" in ctype:
        text = _xlsx_text(data)
    elif ctype.startswith("text/") or "json" in ctype:
        return {
            "text": data.decode("utf-8", errors="replace"),
            "used_ocr": False,
            "meta": {"content_type": ctype, "chars": len(data)},
        }
    else:
        # Tentativas por extensão desconhecida: PDF → DOCX → PPTX → XLSX
        for fn in (_pdf_text, _docx_text, _pptx_text, _xlsx_text):
            try:
                text = fn(data)
            except Exception:  # noqa: BLE001
                text = ""
            if text.strip():
                break

    if not text.strip():
        text = "\n".join(b["text"] for b in ocr(data, lang=lang)["blocks"]).strip()
        used_ocr = True
    return {"text": text, "used_ocr": used_ocr, "meta": {"content_type": ctype, "chars": len(text)}}
