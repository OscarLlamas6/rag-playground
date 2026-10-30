"""Las piezas por separado: tokenizador, BM25, RRF, lectores de formato y fragmentado."""

import pytest

from app.chunking import chunk_document, infer_audience
from app.config import get_settings
from app.domain import Chunk, Hit
from app.fusion import rrf
from app.ingest import load_documents
from app.lexical import BM25, normalize, tokenize


def docs():
    return {d.source: d for d in load_documents(get_settings().docs_dir)}


# ───────── BM25 ─────────
def test_tokenizer_keeps_codes_whole_and_in_parts_and_drops_stopwords():
    tokens = tokenize("¿Qué significa el código E-217 en el router 192.168.1.1?")
    assert "e-217" in tokens and "217" in tokens
    assert "192.168.1.1" in tokens
    assert "el" not in tokens and "que" not in tokens


def test_tokenizer_merges_word_forms_and_ignores_accents():
    assert tokenize("cancelación") == tokenize("cancelar") == tokenize("Cancelacion")


def test_bm25_finds_the_exact_code_and_rare_words_weigh_more():
    chunks = [Chunk("a", "Alarma OPT-1103 potencia baja", {}), Chunk("b", "Alarma OPT-1107 voltaje alto", {}),
              Chunk("c", "Alarma general del equipo", {})]
    bm25 = BM25(chunks, lambda c: c.text)
    assert [h.id for h in bm25.search("¿Qué hago con OPT-1107?", k=1)] == ["b"]
    assert bm25.idf["opt-1107"] > bm25.idf["alarm"]            # la palabra rara pesa más que la común


def test_bm25_applies_the_metadata_filter():
    chunks = [Chunk("a", "plazo de reparación", {"audience": "negocio"}), Chunk("b", "plazo de reparación", {"audience": "residencial"})]
    bm25 = BM25(chunks, lambda c: c.text)
    hits = bm25.search("plazo reparación", accept=lambda m: m["audience"] == "negocio")
    assert [h.id for h in hits] == ["a"]


def test_normalize():
    assert normalize("  Penalización   POR  cancelar ") == "penalizacion por cancelar"


# ───────── RRF ─────────
def test_rrf_matches_the_worked_example():
    mk = lambda i: Hit(i, i, {}, 0.0)  # noqa: E731
    fused = rrf({"bm25": [mk("A"), mk("C"), mk("B")], "vector": [mk("B"), mk("A"), mk("D")]}, k=4)
    assert [h.id for h, _ in fused] == ["A", "B", "C", "D"]
    scores = {h.id: round(h.score, 4) for h, _ in fused}
    assert scores == {"A": 0.0325, "B": 0.0323, "C": 0.0161, "D": 0.0159}
    assert fused[0][1] == {"bm25": 1, "vector": 2}          # posiciones de A en cada lista


def test_rrf_weights_favor_a_list():
    mk = lambda i: Hit(i, i, {}, 0.0)  # noqa: E731
    lists = {"bm25": [mk("A")], "vector": [mk("B")]}
    assert rrf(lists, weights={"bm25": 2.0})[0][0].id == "A"


# ───────── lectores ─────────
def test_three_formats_are_read_into_sections():
    d = docs()
    assert {x.type for x in d.values()} == {"md", "html", "docx"}
    assert all(x.sections and not x.warnings for x in d.values())


def test_html_reader_drops_menus_and_scripts():
    text = " ".join(t for _, t in docs()["guia_tecnica.html"].sections)
    assert "dataLayer" not in text and "Mi Andes" not in text.split("Inicio")[0]
    assert "E-101" in text


def test_docx_reader_keeps_headings_and_flattens_tables():
    sections = dict(docs()["condiciones_servicio.docx"].sections)
    assert any("Clientes Negocio ›" in title for title in sections)
    assert any(" | " in text for text in sections.values())            # las tablas pasan a "celda | celda"


def test_unsupported_format(tmp_path):
    from app.ingest import read_document
    (tmp_path / "a.xlsx").write_text("x")
    with pytest.raises(ValueError):
        read_document(tmp_path / "a.xlsx")


# ───────── fragmentado y metadatos ─────────
def test_audience_comes_from_the_section_title():
    assert infer_audience("Parte B · Clientes Negocio › Compensaciones") == "negocio"
    assert infer_audience("Parte A · Clientes residenciales › Uso aceptable") == "residencial"
    assert infer_audience("1. Planes de internet") == "general"


def test_parent_child_children_remember_their_section():
    doc = docs()["manual_soporte.md"]
    chunks = chunk_document(doc, "parent_child", max_words=35)
    assert all(len(c.text.split()) <= 35 for c in chunks)
    assert all(c.metadata["parent_text"] and c.metadata["parent_id"] in c.id for c in chunks)
    parents = {c.metadata["parent_id"] for c in chunks}
    assert len(parents) == len(doc.sections) and len(chunks) > len(parents)


def test_chunk_ids_are_ascii_even_with_accented_file_names():
    from app.chunking import ascii_slug
    from app.domain import Document
    assert ascii_slug("Política de crédito (v2)") == "Politica_de_credito_v2"
    doc = Document("Garantía técnica.md", "t", "md", [("Sección", "Una oración corta para probar los ids.")])
    for strategy in ("sentences", "section", "parent_child"):
        assert all(c.id.isascii() and c.metadata.get("parent_id", "").isascii() for c in chunk_document(doc, strategy))


def test_other_strategies_and_invalid_arguments():
    doc = docs()["manual_soporte.md"]
    assert len(chunk_document(doc, "section")) == len(doc.sections)
    sentences = chunk_document(doc, "sentences", max_words=40)
    assert all(len(c.text.split()) <= 40 for c in sentences) and "parent_id" not in sentences[0].metadata
    with pytest.raises(ValueError):
        chunk_document(doc, "nope")
