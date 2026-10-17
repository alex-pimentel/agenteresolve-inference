"""Contrato do aquecimento no boot: melhor esforço, um passo não derruba os outros.

Contexto (2026-10-17): a primeira chamada de `tts` depois de um deploy levou 302 s e a
segunda, 7 s. Os modelos locais carregavam no colo de quem chegasse primeiro. O
aquecimento passou a rodar no ``lifespan``; estes testes fixam que ele tenta **todos**
os passos, isola a falha de cada um e pode ser desligado por env.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import unittest
from unittest import mock


class WarmupTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["WARMUP"] = "1"
        from app.config import get_settings

        get_settings.cache_clear()

    def test_tenta_todos_os_passos_mesmo_com_falha(self) -> None:
        from app import main, providers

        chamados: list[str] = []

        def make(name: str):
            def step(*args, **kwargs):
                chamados.append(name)
                if name == "ocr":  # um passo quebrado não pode abortar o resto
                    raise RuntimeError("modelo indisponível")

            return step

        with (
            mock.patch.object(providers, "synthesize", make("tts")),
            mock.patch.object(providers, "embeddings", make("embeddings")),
            mock.patch.object(providers, "ocr", make("ocr")),
            mock.patch.object(providers, "caption", make("caption")),
            mock.patch.object(providers, "detect", make("detect")),
            mock.patch.object(providers, "transcribe", make("transcribe")),
        ):
            main._warmup()

        self.assertEqual(
            chamados,
            ["tts", "embeddings", "ocr", "caption", "detect", "transcribe"],
            "todos os passos precisam ser tentados, em ordem",
        )

    def test_desligado_por_env(self) -> None:
        from app import main, providers

        os.environ["WARMUP"] = "0"
        from app.config import get_settings

        get_settings.cache_clear()
        try:
            with mock.patch.object(providers, "synthesize") as synth:
                main._warmup()
            synth.assert_not_called()
        finally:
            os.environ["WARMUP"] = "1"
            get_settings.cache_clear()

    def test_wav_silencioso_e_wav_valido(self) -> None:
        import io
        import wave

        from app.main import _silent_wav

        with wave.open(io.BytesIO(_silent_wav(0.1))) as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getframerate(), 16000)
            self.assertGreater(handle.getnframes(), 0)


if __name__ == "__main__":
    unittest.main()
