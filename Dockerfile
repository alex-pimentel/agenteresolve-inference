# agenteresolve-inference — API de inferência (LLM gateway + capacidades) no Oracle.
#
# Build multi-arch (o Coolify builda no próprio Oracle = arm64).
# Deps "core" sempre; deps de ML em best-effort (se um wheel faltar na arquitetura, o
# serviço sobe e o endpoint correspondente responde 503 provider_unavailable).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-por \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && (pip install -r requirements-ml.txt || echo "AVISO: deps de ML parciais; endpoints afetados respondem 503")

# Voz do Piper (TTS) embutida na imagem (best-effort).
RUN python -m piper.download_voices en_US-lessac-medium --download-dir /app/voices \
    || echo "AVISO: voz do Piper não baixada; /synthesize responderá 503"

COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
