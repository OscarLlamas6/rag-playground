"""Pruebas REALES con Docling y faster-whisper. No corren en la suite por defecto: `task test:media` (o `pytest -m media`).

Se omiten solas si falta el grupo `media`, ffmpeg o los medios. La primera vez descargan modelos de Hugging Face (minutos).
Para los medios: pon un audio o video en backend/data/media/ (carpeta ignorada por Git) o apunta MEDIA_TEST_FILE a uno.
"""

import os
import subprocess
from pathlib import Path

import pytest

from app import extras, media, pdf
from app.chunking import chunk_document
from app.config import get_settings

pytestmark = pytest.mark.media
DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture(autouse=True)
def real_models(monkeypatch, tmp_path):
    """Deshace los dobles de conftest: aquí se usan las funciones verdaderas (con una caché temporal)."""
    from fakes import REAL_HYBRID, REAL_RUN_DOCLING, REAL_TRANSCRIBE_RAW

    monkeypatch.setattr(pdf, "_run_docling", REAL_RUN_DOCLING)
    monkeypatch.setattr(pdf, "_native_hybrid_pieces", REAL_HYBRID)
    monkeypatch.setattr(media, "_transcribe_raw", REAL_TRANSCRIBE_RAW)


def need(*modules):
    missing = [m for m in modules if not extras.module_available(m)]
    if missing:
        pytest.skip(f"falta el grupo `media` ({', '.join(missing)}): task install:media")


def test_real_docling_ocr_on_the_scanned_pdf():
    need("docling")
    doc = pdf.read_pdf(DOCS / "acta_escaneada.pdf")
    text = " ".join(t for _, t in doc.sections).lower()
    assert doc.info["ocr"] is True and "acta de instalacion" in text and "ax-2200" in text.replace(" ", "")
    assert doc.section_meta[0]["page"] == 1


def test_real_docling_reads_the_text_pdf_without_ocr_and_keeps_the_table():
    need("docling")
    doc = pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    assert doc.info["ocr"] is False and doc.info["pages"] == 2
    text = "\n".join(t for _, t in doc.sections)
    assert any(" | " in line and "Zona Norte" in line for line in text.splitlines())


def test_real_hybrid_chunker_respects_tokens_and_keeps_pages():
    need("docling", "transformers")
    doc = pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    chunks = chunk_document(doc, strategy="docling_hybrid", max_tokens=64, tokenizer="sentence-transformers/all-MiniLM-L6-v2")
    assert len(chunks) >= 3 and {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["contextualized"] for c in chunks)


def _sample_media() -> Path | None:
    explicit = os.getenv("MEDIA_TEST_FILE")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    found = media.list_media(get_settings().media_dir)
    return found[0] if found else None


def test_real_whisper_gives_timestamps_and_caches(tmp_path, monkeypatch):
    need("faster_whisper", "av")
    sample = _sample_media()
    if sample is None:
        pytest.skip("no hay medios de prueba: copia uno a backend/data/media/ o define MEDIA_TEST_FILE")
    if not extras.ffmpeg_path():
        pytest.skip("falta ffmpeg")
    clip = tmp_path / "media" / "clip.wav"
    clip.parent.mkdir()
    subprocess.run([extras.ffmpeg_path(), "-nostdin", "-v", "error", "-ss", "30", "-t", "60", "-i", str(sample), "-vn", "-ac", "1",
                    "-ar", "16000", str(clip)], check=True)
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WHISPER_MODEL", "base")
    first = media.transcribe(clip)
    assert first.segments and not first.from_cache and first.duration == pytest.approx(60, abs=1)
    starts = [s.start for s in first.segments]
    assert starts == sorted(starts) and first.segments[-1].end <= 61
    again = media.transcribe(clip)
    assert again.from_cache and [s.text for s in again.segments] == [s.text for s in first.segments]
    doc = media.read_media(clip)
    assert doc.section_meta and all(0 <= m["start_s"] < m["end_s"] <= 61 for m in doc.section_meta)
