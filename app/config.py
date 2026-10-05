"""Configuração da API de inferência (tudo por env)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Auth ---------------------------------------------------------------
    # Bearer exigido em todos os endpoints (exceto /health). Se vazio, roda sem auth (dev).
    inference_key: str | None = None

    # LLM gateway (OpenAI-compatible) ------------------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:0.8b"
    # Modelo dedicado a tradução (melhor para tradução; mantido junto ao geral).
    translate_model: str = "translategemma:4b"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    llm_timeout: float = 180.0

    # Modelos locais ------------------------------------------------------
    tesseract_lang: str = "eng+por"
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    detect_model: str = "yolov8n.pt"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    piper_voice: str = "en_US-lessac-medium"

    # Limites -------------------------------------------------------------
    max_concurrency: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
