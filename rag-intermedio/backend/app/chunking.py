"""Fragmentar (chunking) y etiquetar con metadatos.

Dos presiones opuestas: un fragmento PEQUEÑO da un vector preciso pero poco contexto para responder; uno GRANDE da
contexto pero un vector "diluido". El patrón padre-hijo evita elegir: se BUSCA con hijos pequeños y se ENTREGA al
modelo el padre (la sección completa).

Estrategias:
    sentences     junta oraciones hasta `max_words`, con solape de `overlap` oraciones
    section       un fragmento por sección
    parent_child  hijos de `max_words` palabras que recuerdan el texto de su padre (la sección)

Cada fragmento lleva metadatos para citar y filtrar: `source`, `section`, `type` y `audience`.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .domain import Chunk, Document

STRATEGIES = ("sentences", "section", "parent_child")


def infer_audience(section_title: str) -> str:
    """Etiqueta de audiencia a partir del título: el Word separa 'Parte A · Clientes residenciales' y 'Parte B · Clientes Negocio'.

    El metadato se asigna en la INGESTA y es lo que luego permite filtrar (calidad y seguridad)."""
    t = section_title.lower()
    if "negocio" in t:
        return "negocio"
    if "residencial" in t:
        return "residencial"
    return "general"


def ascii_slug(text: str) -> str:
    """Los IDs de vectores deben ser ASCII (Pinecone lo exige): sin acentos ni símbolos raros."""
    plain = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Za-z0-9_]+", "_", plain).strip("_") or "doc"


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip()) if p.strip()]


def _group_sentences(text: str, max_words: int, overlap: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text):
        if current and sum(len(s.split()) for s in current) + len(sentence.split()) > max_words:
            groups.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            while current and sum(len(s.split()) for s in current) + len(sentence.split()) > max_words:
                current.pop(0)          # el solape no puede hacer que el fragmento nuevo se pase del tamaño máximo
        current.append(sentence)
    if current and (not groups or " ".join(current) != groups[-1]):
        groups.append(" ".join(current))
    return groups


def chunk_document(doc: Document, strategy: str = "sentences", max_words: int = 70, overlap: int = 1) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy debe ser una de: {', '.join(STRATEGIES)}")
    if max_words < 10:
        raise ValueError("max_words debe ser al menos 10")

    stem = ascii_slug(Path(doc.source).stem)
    chunks: list[Chunk] = []
    for p, (heading, text) in enumerate(doc.sections):
        base = {"source": doc.source, "section": heading, "type": doc.type, "audience": infer_audience(heading)}
        full = " ".join(text.split())
        if strategy == "section":
            pieces, extra = [full], {}
        elif strategy == "sentences":
            pieces, extra = _group_sentences(text, max_words, overlap), {}
        else:
            pieces = _group_sentences(text, max_words, 0)
            extra = {"parent_id": f"{stem}-p{p}", "parent_text": full}
        for j, piece in enumerate(pieces):
            chunk_id = f"{stem}-p{p}.{j}" if strategy == "parent_child" else f"{stem}-{len(chunks)}"
            chunks.append(Chunk(id=chunk_id, text=piece, metadata={**base, **extra, "n": len(chunks)}))
    return chunks


def text_to_embed(chunk: Chunk) -> str:
    """Lo que se vectoriza (y se indexa en BM25) es 'sección + texto': el título da contexto a un fragmento suelto."""
    return f"{chunk.metadata['section']}. {chunk.text}"
