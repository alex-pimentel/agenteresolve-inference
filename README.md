# agenteresolve-inference

API de inferência do Agenteresolve, rodando no servidor **Oracle** (arm64, CPU). Serve
como ponto único de IA para os 15 backends e para os workers da plataforma.

- **LLM gateway** OpenAI-compatible (`POST /v1/chat/completions`): tenta o **Ollama** local
  e cai no **OpenRouter** (híbrido) quando o local falha e há chave configurada.
- **Capacidades** (CPU): `/ocr`, `/caption`, `/detect`, `/transcribe`, `/enhance`,
  `/synthesize`, `/embeddings`.
- **Extração de documentos**: `/extract` (PDF, DOCX, PPTX, XLSX) com fallback OCR.
- **Auth**: `Authorization: Bearer $INFERENCE_KEY` (exceto `/health`). Verificador plugável
  para o futuro sistema de chaves por cliente (BYOK).

## Endpoints

| Método | Rota | Corpo | Resposta |
|---|---|---|---|
| GET | `/health` | — | status/config |
| POST | `/v1/chat/completions` | JSON OpenAI | JSON OpenAI |
| POST | `/ocr` | multipart `file`, `lang?` | `{blocks:[{text,bbox,confidence}]}` |
| POST | `/caption` | multipart `file` | `{caption}` |
| POST | `/detect` | multipart `file`, `labels?` | `{detections:[{label,bbox,score}]}` |
| POST | `/transcribe` | multipart `file`, `lang?` | `{text,segments,language}` |
| POST | `/enhance` | multipart `file`, `mode?` | binário (wav) |
| POST | `/synthesize` | JSON `{text,voice?,lang?,speed?}` | binário (wav) |
| POST | `/embeddings` | JSON `{input}` | `{data:[{embedding}]}` |
| POST | `/extract` | multipart `file`, `lang?` | `{text,used_ocr,meta}` |

## Env

| Variável | Padrão | Descrição |
|---|---|---|
| `INFERENCE_KEY` | — | Bearer exigido (se vazio, roda sem auth — dev) |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | endpoint do Ollama |
| `LLM_MODEL` | `qwen3.5:0.8b` | modelo local padrão |
| `OPENROUTER_API_KEY` | — | habilita o fallback híbrido |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | modelo do fallback |
| `TESSERACT_LANG` | `eng+por` | idiomas do OCR |
| `WHISPER_MODEL` | `base` | tamanho do faster-whisper |
| `MAX_CONCURRENCY` | `1` | concorrência das operações pesadas |

## Rodar localmente

```bash
pip install -r requirements.txt
# opcional (visão/embeddings/TTS):
pip install -r requirements-ml.txt
INFERENCE_KEY=dev uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Requer `ffmpeg` e `tesseract-ocr` no sistema. Modelos pesados carregam sob demanda.

## Deploy

App no Coolify no servidor Oracle (arm64), domínio `inference.agenteresolve.com.br`,
porta `8000`. O LLM local roda em um app Ollama separado (`OLLAMA_URL`). Ver
`website/docs/superpowers/specs/2026-10-05-infra-oracle-inference-design.md`.
