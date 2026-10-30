from pathlib import Path

import pytest

from app.config import get_settings
from app.documents import chunk_document, load_documents, read_markdown, text_to_embed


def docs():
    return load_documents(get_settings().docs_dir)


def test_loads_three_markdown_files_with_sections():
    documents = docs()
    assert [d.source for d in documents] == [
        "contratacion_y_planes.md", "facturacion_y_compensaciones.md", "soporte_tecnico.md"]
    assert all(len(d.sections) >= 2 for d in documents)
    assert "Planes de internet" in [t for t, _ in documents[0].sections]


def test_section_strategy_gives_one_chunk_per_section():
    doc = docs()[0]
    chunks = chunk_document(doc, strategy="section")
    assert len(chunks) == len(doc.sections)
    assert chunks[0].metadata == {"source": doc.source, "section": doc.sections[0][0], "n": 0}


def test_sentences_respects_max_words_and_overlaps():
    doc = docs()[0]
    chunks = chunk_document(doc, strategy="sentences", max_words=40, overlap=1)
    assert len(chunks) > len(doc.sections)
    # un fragmento puede pasarse solo si una única oración ya supera el tope
    assert all(len(c.text.split()) <= 40 or "." not in c.text[:-1] for c in chunks)
    # con solape, la última oración de un fragmento reaparece al inicio del siguiente (dentro de la misma sección)
    same_section = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["section"] == b.metadata["section"]]
    assert any(a.text.split(". ")[-1].rstrip(".") in b.text for a, b in same_section)


def test_fixed_windows_and_unique_ids():
    chunks = chunk_document(docs()[2], strategy="fixed", max_words=30, overlap=1)
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.text.split()) <= 30 for c in chunks)


def test_invalid_arguments():
    doc = docs()[0]
    with pytest.raises(ValueError):
        chunk_document(doc, strategy="magic")
    with pytest.raises(ValueError):
        chunk_document(doc, max_words=3)


def test_text_to_embed_prepends_section(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("# Título\n\n## Precios\n\nEl plan cuesta 10 dólares.\n", encoding="utf-8")
    chunk = chunk_document(read_markdown(f), strategy="section")[0]
    assert text_to_embed(chunk) == "Precios. El plan cuesta 10 dólares."
