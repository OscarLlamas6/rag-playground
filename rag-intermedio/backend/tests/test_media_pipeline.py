"""PDF con OCR automático, chunking docling_hybrid, ventanas de transcripción, caché, endpoint de medios y degradación.

Todo con dobles de Docling y de Whisper (ver conftest.py y fakes.py): rápido, sin modelos, sin red.
"""

import sys
from pathlib import Path

import fakes
import pytest

from app import extras, media, pdf
from app.chunking import chunk_document, text_to_embed
from app.config import get_settings
from app.domain import Document
from app.locations import format_range, format_time, location
from app.media import Segment, group_segments
from app.pdf import PdfItem, items_to_sections, needs_ocr

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


# ───────── formato de tiempo y citas ─────────
def test_format_time_and_range():
    assert format_time(0) == "00:00" and format_time(59.9) == "00:59" and format_time(750) == "12:30"
    assert format_time(3723) == "1:02:03"
    assert format_range(750, 790.4) == "12:30–13:10"


def test_location_for_pages_time_and_nothing():
    assert location({"page": 12}) == "p. 12"
    assert location({"page": 12, "page_end": 13}) == "p. 12–13"
    assert location({"page": 4, "page_end": 4}) == "p. 4"
    assert location({"start_s": 750.0, "end_s": 790.0, "media": "x.mp3"}) == "12:30–13:10"
    assert location({"start_s": 0.0, "end_s": 5.0}) == "00:00–00:05"      # el segundo 0 también es una posición
    assert location({"source": "a.md"}) == ""


# ───────── decisión de OCR ─────────
def test_needs_ocr_when_pages_have_almost_no_text():
    assert needs_ocr([], pages_read=1)[0] is True
    items = [PdfItem("text", "x" * 200, p) for p in (1, 2, 3)]
    assert needs_ocr(items, pages_read=3)[0] is False
    mostly_images = [PdfItem("text", "x" * 200, 1)]                       # 1 de 4 páginas con texto: 3 vacías
    decision, reason = needs_ocr(mostly_images, pages_read=4)
    assert decision is True and "3 de 4" in reason
    one_cover = [PdfItem("text", "x" * 200, p) for p in (2, 3, 4)]        # solo la portada sin texto: no hace falta
    assert needs_ocr(one_cover, pages_read=4)[0] is False
    assert needs_ocr([PdfItem("text", "hola", 1)], pages_read=1, min_chars=3)[0] is False   # el umbral es configurable


def test_scanned_pdf_is_read_first_without_ocr_and_then_with_ocr(docling):
    doc = pdf.read_pdf(DOCS / "acta_escaneada.pdf")
    assert [c[1] for c in docling.calls] == [False, True]                 # primero rápido, luego OCR
    assert doc.info["ocr"] is True and doc.info["pages"] == 1
    assert any("OCR aplicado" in w or "se aplicó OCR" in w for w in doc.warnings)
    assert any("errores" in w for w in doc.warnings)
    assert doc.sections and doc.section_meta[0]["page"] == 1


def test_pdf_with_text_does_not_use_ocr(docling):
    doc = pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    assert docling.calls == [("boletin_mantenimiento.pdf", False, 30)]
    assert doc.info["ocr"] is False and doc.warnings == []


def test_ocr_modes_off_and_force(docling, monkeypatch):
    monkeypatch.setenv("DOCLING_OCR", "off")
    doc = pdf.read_pdf(DOCS / "acta_escaneada.pdf")
    assert [c[1] for c in docling.calls] == [False] and doc.sections == []
    assert len(doc.warnings) == 1 and "DOCLING_OCR=off" in doc.warnings[0]     # off: no reintenta, pero avisa de que no hay texto
    docling.calls.clear()
    monkeypatch.setenv("DOCLING_OCR", "force")
    pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    assert [c[1] for c in docling.calls] == [True]


def test_page_cap_is_reported(docling, monkeypatch):
    docling.total_pages["boletin_mantenimiento.pdf"] = 142
    doc = pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    assert doc.info["pages"] == 30 and doc.info["pages_total"] == 142
    assert any("primeras 30 de 142" in w and "DOCLING_MAX_PAGES" in w for w in doc.warnings)
    monkeypatch.setenv("DOCLING_MAX_PAGES", "0")                         # 0 = sin tope
    assert pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf").info["pages"] == 142


# ───────── páginas en secciones y fragmentos ─────────
def test_sections_keep_their_pages():
    items = [PdfItem("heading", "Uno", 1, 1), PdfItem("text", "a", 1), PdfItem("text", "b", 2),
             PdfItem("heading", "Sub", 3, 2), PdfItem("text", "c", 3), PdfItem("table", "x | y", 3)]
    sections, metas = items_to_sections(items, "Doc")
    assert [s[0] for s in sections] == ["Uno", "Uno › Sub"]
    assert metas == [{"page": 1, "page_end": 2}, {"page": 3}]
    assert sections[1][1] == "c\nx | y"


def test_table_markdown_becomes_lines():
    md = "| Zona | Fecha |\n|------|-------|\n| Norte | 4 |\n"
    assert pdf.table_to_lines(md) == "Zona | Fecha\nNorte | 4"


@pytest.mark.parametrize("strategy", ["sentences", "section", "parent_child"])
def test_every_pdf_chunk_carries_its_page(strategy):
    doc = pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    chunks = chunk_document(doc, strategy=strategy, max_words=20)
    assert chunks and all(c.metadata["page"] in (1, 2, 3) and c.metadata["type"] == "pdf" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


# ───────── docling_hybrid ─────────
def test_docling_hybrid_uses_the_chunker_and_vectorizes_the_contextualized_text(docling):
    doc = pdf.read_pdf(DOCS / "boletin_mantenimiento.pdf")
    chunks = chunk_document(doc, strategy="docling_hybrid", max_tokens=128)
    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert all(c.metadata["strategy_used"] == "docling_hybrid" and c.metadata["contextualized"] for c in chunks)
    assert text_to_embed(chunks[0]) == chunks[0].text and chunks[0].text.startswith("Informe boletin_mantenimiento.pdf")


def test_docling_hybrid_falls_back_to_sentences_for_documents_without_docling():
    from app.ingest import load_documents
    md = next(d for d in load_documents(get_settings().docs_dir) if d.type == "md")
    chunks = chunk_document(md, strategy="docling_hybrid", max_tokens=100)
    assert chunks and all(c.metadata["strategy_used"].startswith("sentences (respaldo") for c in chunks)
    assert max(len(c.text.split()) for c in chunks) <= 70                 # 100 tokens ≈ 70 palabras
    with pytest.raises(ValueError):
        chunk_document(md, strategy="docling_hybrid", max_tokens=8)


def test_hybrid_tokenizer_defaults_to_the_embedding_model(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-nomic-embed-text-v2-moe")
    assert get_settings().docling_tokenizer == "nomic-ai/nomic-embed-text-v2-moe"
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    assert get_settings().docling_tokenizer == "sentence-transformers/all-MiniLM-L6-v2"
    monkeypatch.setenv("DOCLING_TOKENIZER", "otro/tokenizer")
    assert get_settings().docling_tokenizer == "otro/tokenizer"


# ───────── ventanas de transcripción ─────────
def segs(n, step=5.0, length=4.0):
    return [Segment(i * step, i * step + length, f"s{i}") for i in range(n)]


def test_windows_group_segments_with_times_and_overlap():
    windows = group_segments(segs(40), window_s=50, overlap_s=10)          # 40 segmentos de 5 s = 200 s
    assert windows[0].start == 0 and windows[0].end <= 50 and windows[0].text.startswith("s0 s1")
    assert all(w.end - w.start <= 50 for w in windows)
    for before, after in zip(windows, windows[1:]):
        assert after.start > before.start                                  # siempre avanza
        assert after.start < before.end                                    # y se solapa con la anterior
        assert before.end - after.start <= 10 + 4                          # el solape no pasa de overlap_s (+ un segmento)
    assert windows[-1].end == 199.0 and "s39" in windows[-1].text


def test_windows_without_overlap_do_not_repeat_text():
    windows = group_segments(segs(40), window_s=50, overlap_s=0)
    words = [w for win in windows for w in win.text.split()]
    assert len(words) == len(set(words)) == 40


def test_window_edge_cases():
    assert group_segments([]) == []
    one = group_segments([Segment(10, 100, "largo")], window_s=50)           # un segmento más largo que la ventana
    assert (one[0].start, one[0].end, one[0].text) == (10, 100, "largo")
    assert group_segments([Segment(0, 1, "  "), Segment(1, 2, "hola")])[0].text == "hola"
    with pytest.raises(ValueError):
        group_segments(segs(3), window_s=10, overlap_s=10)
    with pytest.raises(ValueError):
        group_segments(segs(3), window_s=0)


# ───────── caché de transcripciones ─────────
@pytest.fixture
def mp3(tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "charla.mp3").write_bytes(b"ID3 audio de mentira")
    return folder / "charla.mp3"


def test_transcript_is_cached_by_file_model_and_language(mp3, whisper, monkeypatch):
    first = media.transcribe(mp3)
    again = media.transcribe(mp3)
    assert len(whisper.calls) == 1 and not first.from_cache and again.from_cache
    assert [s.text for s in again.segments] == [s.text for s in first.segments]
    monkeypatch.setenv("WHISPER_MODEL", "base")                              # otro modelo = otra transcripción
    media.transcribe(mp3)
    monkeypatch.setenv("WHISPER_LANGUAGE", "es")                             # otro idioma = otra transcripción
    media.transcribe(mp3)
    assert [c[1:] for c in whisper.calls] == [("small", None), ("base", None), ("base", "es")]
    mp3.write_bytes(b"ID3 otro contenido")                                    # otro archivo = otra transcripción
    media.transcribe(mp3)
    assert len(whisper.calls) == 4


def test_cache_key_changes_with_every_ingredient():
    keys = {media.cache_key("a" * 64, m, lang, p) for m in ("small", "base") for lang in (None, "es") for p in (None, "ChatGPT")}
    assert len(keys) == 8


def test_media_document_has_windows_with_times(mp3, whisper, monkeypatch):
    monkeypatch.setenv("MEDIA_WINDOW_SECONDS", "40")
    monkeypatch.setenv("MEDIA_OVERLAP_SECONDS", "0")
    doc = media.read_media(mp3)
    assert doc.type == "audio" and doc.info["duration_s"] == 120.0 and doc.info["from_cache"] is False
    assert doc.section_meta[0] == {"media": "charla.mp3", "start_s": 0.0, "end_s": 29.0}
    assert doc.sections[0][0] == "00:00–00:29"
    chunks = chunk_document(doc, strategy="parent_child")                     # con cualquier estrategia, una ventana = un fragmento
    assert len(chunks) == len(doc.sections) and chunks[1].metadata["start_s"] == 30.0
    assert text_to_embed(chunks[0]).startswith("charla. ")                    # el contexto es el título, no el rango de tiempo
    assert media.read_media(mp3).info["from_cache"] is True


def test_kinds_and_mimes():
    assert media.kind_of("a.MP4") == "video" and media.kind_of("a.flac") == "audio"
    assert media.mime_of("a.mp3") == "audio/mpeg" and media.mime_of("a.webm") == "video/webm"


# ───────── ingesta de punta a punta con medios ─────────
def test_ingest_with_audio_cites_the_time(client, mp3, whisper):
    report = client.post("/api/ingest", json={"strategy": "sentences", "max_words": 40}).json()
    assert report["documents"] == 6 and report["per_document"]["charla.mp3"]["type"] == "audio"
    assert report["per_document"]["acta_escaneada.pdf"]["ocr"] is True
    assert any("acta_escaneada.pdf" in w and "OCR" in w for w in report["warnings"])
    body = client.post("/api/ask", json={"question": "¿Qué hace el servidor3?", "mode": "bm25", "k": 2}).json()
    top = body["hits"][0]
    assert top["source"] == "charla.mp3" and top["media"] == "charla.mp3" and top["start_s"] is not None
    assert top["location"] == format_range(top["start_s"], top["end_s"])
    assert "posición: " + top["location"] in body["prompt"]
    client.post("/api/ingest", json={})                                       # reingestar: la transcripción sale de la caché
    assert len(whisper.calls) == 1


def test_ingest_reports_pdf_pages_and_page_cites(client):
    client.post("/api/ingest", json={"strategy": "section"})
    hits = client.post("/api/ask", json={"question": "tema2 equipo AX-200 boletin_mantenimiento", "mode": "bm25", "k": 3}).json()["hits"]
    assert any(h["type"] == "pdf" and h["location"].startswith("p. ") for h in hits)


def test_ingest_docling_hybrid_end_to_end(client):
    report = client.post("/api/ingest", json={"strategy": "docling_hybrid", "max_tokens": 128}).json()
    assert report["per_document"]["boletin_mantenimiento.pdf"]["strategy_used"] == "docling_hybrid"
    assert report["per_document"]["manual_soporte.md"]["strategy_used"].startswith("sentences (respaldo")
    assert client.post("/api/ingest", json={"strategy": "docling_hybrid", "max_tokens": 8}).status_code == 422


# ───────── endpoint de medios ─────────
def test_media_listing(client, mp3):
    items = client.get("/api/media").json()
    assert items == [{"name": "charla.mp3", "type": "audio", "mime": "audio/mpeg", "size_kb": 0, "duration_s": None, "duration": None,
                      "url": "/api/media/charla.mp3"}]


def test_media_file_supports_range_requests(client, mp3):
    data = bytes(range(256)) * 4
    mp3.write_bytes(data)
    full = client.get("/api/media/charla.mp3")
    assert full.status_code == 200 and full.content == data
    assert full.headers["accept-ranges"] == "bytes" and full.headers["content-type"].startswith("audio/mpeg")
    part = client.get("/api/media/charla.mp3", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == data[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(data)}"
    tail = client.get("/api/media/charla.mp3", headers={"Range": "bytes=1000-"})
    assert tail.status_code == 206 and tail.content == data[1000:]
    assert client.get("/api/media/charla.mp3", headers={"Range": "bytes=5000-6000"}).status_code == 416


@pytest.mark.parametrize("name", ["..%2F..%2Fetc%2Fpasswd", "%2Fetc%2Fpasswd", "..%5Csecreto.mp3", "a%00.mp3", "%2E%2E"])
def test_media_file_rejects_paths(client, mp3, name):
    assert client.get(f"/api/media/{name}").status_code in (400, 404)


def test_resolve_media_only_returns_files_from_the_folder(mp3, tmp_path):
    (mp3.parent / "notas.txt").write_text("no es un medio")
    (tmp_path / "fuera.mp3").write_bytes(b"x")
    assert media.resolve_media(mp3.parent, "charla.mp3") == mp3
    assert media.resolve_media(mp3.parent, "notas.txt") is None             # extensión no permitida
    assert media.resolve_media(mp3.parent, "inexistente.mp3") is None
    for bad in ("../fuera.mp3", "/etc/passwd", "a/b.mp3", "a\\b.mp3", "", ".."):
        with pytest.raises(ValueError):
            media.resolve_media(mp3.parent, bad)


def test_media_endpoint_errors(client, mp3):
    assert client.get("/api/media/no_existe.mp3").status_code == 404
    assert client.get("/api/media/notas.txt").status_code == 404


def test_pdf_file_endpoint(client):
    r = client.get("/api/pdf/boletin_mantenimiento.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    assert client.get("/api/pdf/manual_soporte.md").status_code == 404
    assert client.get("/api/pdf/..%2Fpyproject.toml").status_code in (400, 404)


# ───────── degradación elegante sin el grupo `media` ─────────
@pytest.fixture
def without_media(monkeypatch):
    """Simula una instalación base: Docling, faster-whisper y av no se pueden importar y se usan las funciones REALES."""
    for module in ("docling", "docling_core", "faster_whisper", "av", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(pdf, "_run_docling", fakes.REAL_RUN_DOCLING)
    monkeypatch.setattr(media, "_transcribe_raw", fakes.REAL_TRANSCRIBE_RAW)
    monkeypatch.setattr(pdf, "_native_hybrid_pieces", fakes.REAL_HYBRID)


def test_require_gives_the_install_instruction(without_media):
    with pytest.raises(extras.MediaDepsMissing, match="task install:media"):
        extras.require("Leer PDF", "docling")
    assert extras.capabilities()["docling"] is False and extras.capabilities()["faster_whisper"] is False


def test_base_ingest_still_works_and_skips_pdf_and_media_with_a_clear_warning(client, mp3, without_media):
    report = client.post("/api/ingest", json={}).json()
    assert report["documents"] == 3 and report["chunks"] > 30
    skipped = {n for n, d in report["per_document"].items() if d.get("skipped")}
    assert skipped == {"acta_escaneada.pdf", "boletin_mantenimiento.pdf", "charla.mp3"}
    assert all("task install:media" in w for w in report["warnings"] if w.split(":")[0] in skipped)
    answer = client.post("/api/ask", json={"question": "¿Cuánto cuesta el plan de 300 Mbps?"}).json()
    assert "39" in answer["answer"]
    status = client.get("/api/status").json()["media"]
    assert status["docling"] is False and status["faster_whisper"] is False


def test_hybrid_chunking_without_docling_is_a_503_not_a_crash(client, without_media, monkeypatch):
    doc = Document("x.pdf", "X", "pdf", [("A", "texto")], native=object())
    with pytest.raises(extras.MediaDepsMissing):
        chunk_document(doc, strategy="docling_hybrid")


def test_video_without_ffmpeg_explains_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "ffmpeg_path", lambda: None)
    with pytest.raises(extras.MediaError, match="ffmpeg"):
        media._extract_audio(tmp_path / "v.mp4", tmp_path)
