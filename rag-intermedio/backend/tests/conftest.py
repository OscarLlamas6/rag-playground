"""Las pruebas no necesitan servidores ni cuentas: embeddings y modelo "fake", almacén en memoria.

Tampoco necesitan el grupo `media`: Docling y faster-whisper se sustituyen por dobles (`fakes.py`). Las pruebas con los
verdaderos están en `test_media_real.py` bajo el marcador `media` (`task test:media`).
"""

import fakes
import pytest
from fastapi.testclient import TestClient

from app import main, media, pdf


@pytest.fixture
def docling(monkeypatch):
    double = fakes.FakeDocling()
    monkeypatch.setattr(pdf, "_run_docling", double)
    monkeypatch.setattr(pdf, "_native_hybrid_pieces", fakes.fake_hybrid_pieces)
    return double


@pytest.fixture
def whisper(monkeypatch):
    double = fakes.FakeWhisper()
    monkeypatch.setattr(media, "_transcribe_raw", double)
    monkeypatch.setattr(media, "_probe", lambda *args: None)           # sin ffprobe
    return double


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path, docling, whisper):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "fake")
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("CHUNKS_CACHE", str(tmp_path / "chunks.json"))   # que las pruebas no ensucien el caché real
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))            # nunca los medios reales de quien programa
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path / "media-cache"))
    for name in ("DOCLING_MAX_PAGES", "DOCLING_OCR", "DOCLING_TOKENIZER", "WHISPER_MODEL", "WHISPER_LANGUAGE", "WHISPER_PROMPT",
                 "FRAME_CAPTIONS", "MEDIA_WINDOW_SECONDS", "MEDIA_OVERLAP_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    main._service = None


@pytest.fixture
def client():
    return TestClient(main.app)
