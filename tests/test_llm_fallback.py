"""Prova da cadeia local→remoto: o passo local não pode consumir o orçamento inteiro.

Contexto (2026-10-17): a API de inferência rodava num host de 2 vCPU e o passo local
(Ollama, 7B) levava >180 s num prompt grande. Era esse o valor que o cliente do worker
esperava antes de cortar, então o fallback gratuito **nunca** rodava: a ferramenta
estourava em ~300 s ou morria com "Broken pipe" quando o container era reciclado.
Estes testes fixam o contrato: estourou o orçamento local → resposta vem do remoto.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCAL_SLEEP = 5.0

CALLS: dict[str, int] = {"local": 0, "remote": 0}


class _Handler(BaseHTTPRequestHandler):
    """Servidor de apoio: ``/api/tags`` responde na hora; o chat local dorme."""

    role = "local"

    def log_message(self, *args):  # noqa: ANN002 - silencia o log do http.server
        return

    def do_GET(self):  # noqa: N802 - API do http.server
        if self.path.endswith("/api/tags"):
            self._json({"models": [{"name": "stub-local"}]})
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802 - API do http.server
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        CALLS[self.role] += 1
        if self.role == "local":
            time.sleep(LOCAL_SLEEP)
        self._json({"choices": [{"message": {"role": "assistant", "content": f"via-{self.role}"}}]})

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(role: str) -> tuple[ThreadingHTTPServer, str]:
    handler = type(f"Handler_{role}", (_Handler,), {"role": role})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class FallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.local_server, cls.local_url = _serve("local")
        cls.remote_server, cls.remote_url = _serve("remote")

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.local_server, cls.remote_server):
            server.shutdown()

    def _configure(self, local_budget: float) -> None:
        os.environ.update(
            {
                "OLLAMA_URL": self.local_url,
                "LLM_MODEL": "stub-local",
                "OPENROUTER_API_KEY": "stub-key",
                "OPENROUTER_BASE_URL": self.remote_url,
                "OPENROUTER_MODEL": "stub-remote",
                "LLM_LOCAL_TIMEOUT": str(local_budget),
                "LLM_TIMEOUT": "30",
            }
        )
        from app.config import get_settings

        get_settings.cache_clear()
        CALLS.update({"local": 0, "remote": 0})

    def _call(self) -> tuple[str, float]:
        from app.llm import chat_completions

        started = time.monotonic()
        out = chat_completions({"messages": [{"role": "user", "content": "oi"}]})
        return out["choices"][0]["message"]["content"], time.monotonic() - started

    def test_orcamento_curto_cai_no_remoto(self) -> None:
        self._configure(local_budget=1.0)
        content, elapsed = self._call()
        self.assertEqual(content, "via-remote", "deveria responder pelo remoto")
        self.assertLess(elapsed, LOCAL_SLEEP, "não pode esperar o local inteiro")
        self.assertEqual(CALLS["local"], 1, "o local ainda é tentado primeiro")
        self.assertEqual(CALLS["remote"], 1)

    def test_orcamento_longo_usa_o_local(self) -> None:
        self._configure(local_budget=20.0)
        content, _ = self._call()
        self.assertEqual(content, "via-local", "local rápido vence o remoto")
        self.assertEqual(CALLS["remote"], 0, "sem gastar o remoto à toa")


if __name__ == "__main__":
    unittest.main()
