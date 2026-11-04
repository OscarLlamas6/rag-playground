"""Fragmentar (chunking) y etiquetar con metadatos.

Dos presiones opuestas: un fragmento PEQUEÑO da un vector preciso pero poco contexto para responder; uno GRANDE da
contexto pero un vector "diluido". El patrón padre-hijo evita elegir: se BUSCA con hijos pequeños y se ENTREGA al
modelo el padre (la sección completa).

Estrategias:
    sentences     junta oraciones hasta `max_words`, con solape de `overlap` oraciones
    section       un fragmento por sección
    parent_child  hijos de `max_words` palabras que recuerdan el texto de su padre (la sección)
    docling_hybrid  el HybridChunker de Docling: sigue la jerarquía del documento (encabezados, tablas, listas) y respeta un
                    límite de TOKENS (`max_tokens`): parte lo que lo excede y une lo muy pequeño. Solo para PDF leídos con
                    Docling; los demás formatos usan `sentences` con un tamaño equivalente. ¡No es la búsqueda híbrida!

Audio y video no se fragmentan aquí: ya llegan como ventanas de tiempo (`media.group_segments`) y cada ventana es un fragmento
con cualquier estrategia.

Cada fragmento lleva metadatos para citar y filtrar: `source`, `section`, `type` y `audience`, y su posición:
`page`/`page_end` (PDF) o `media`, `start_s`, `end_s` (audio y video).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .config import get_settings
from .domain import Chunk, Document

MEDIA_TYPES = ("audio", "video")
STRATEGIES = ("sentences", "section", "parent_child", "docling_hybrid")
WORDS_PER_TOKEN = 0.7        # aproximación para los documentos sin Docling: 256 tokens ≈ 180 palabras


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


def _media_chunks(doc: Document, stem: str) -> list[Chunk]:
    """Cada ventana de la transcripción es un fragmento (con su tiempo). La sección es el rango 'mm:ss–mm:ss'."""
    chunks = []
    for n, ((heading, text), meta) in enumerate(zip(doc.sections, doc.section_meta)):
        context = f"{doc.title} (lo que se ve)" if meta.get("content") == "frame" else doc.title
        metadata = {"source": doc.source, "section": heading, "type": doc.type, "audience": "general", "embed_context": context,
                    **meta, "n": n}
        chunks.append(Chunk(id=f"{stem}-m{n}", text=" ".join(text.split()), metadata=metadata))
    return chunks


def _hybrid_chunks(doc: Document, stem: str, max_tokens: int, tokenizer: str) -> list[Chunk]:
    from . import pdf

    chunks = []
    for n, piece in enumerate(pdf.hybrid_chunks(doc, tokenizer, max_tokens)):
        heading = " › ".join(piece.headings) or doc.title
        metadata = {"source": doc.source, "section": heading, "type": doc.type, "audience": infer_audience(heading),
                    "contextualized": True, "strategy_used": "docling_hybrid", "n": n}
        if piece.page:
            metadata["page"] = piece.page
        if piece.page_end and piece.page_end != piece.page:
            metadata["page_end"] = piece.page_end
        chunks.append(Chunk(id=f"{stem}-h{n}", text=piece.text, metadata=metadata))
    return chunks


def chunk_document(doc: Document, strategy: str = "sentences", max_words: int = 70, overlap: int = 1,
                   max_tokens: int = 256, tokenizer: str | None = None) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy debe ser una de: {', '.join(STRATEGIES)}")
    if max_words < 10:
        raise ValueError("max_words debe ser al menos 10")
    if max_tokens < 32:
        raise ValueError("max_tokens debe ser al menos 32")

    if doc.type in MEDIA_TYPES:
        return _media_chunks(doc, ascii_slug(doc.source))
    if strategy == "docling_hybrid":
        if doc.native is not None:
            return _hybrid_chunks(doc, ascii_slug(doc.source), max_tokens, tokenizer or get_settings().docling_tokenizer)
        # sin Docling (md, html, docx): la estrategia existente más parecida, con un tamaño equivalente a `max_tokens`
        max_words, strategy = max(10, int(max_tokens * WORDS_PER_TOKEN)), "sentences"
        fallback = "sentences (respaldo de docling_hybrid)"
    else:
        fallback = strategy

    stem = ascii_slug(Path(doc.source).stem) if doc.type in ("md", "html", "docx") else ascii_slug(doc.source)
    chunks: list[Chunk] = []
    for p, (heading, text) in enumerate(doc.sections):
        position = doc.section_meta[p] if p < len(doc.section_meta) else {}
        base = {"source": doc.source, "section": heading, "type": doc.type, "audience": infer_audience(heading), **position}
        if fallback != strategy or doc.type == "pdf":
            base["strategy_used"] = fallback
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
    """Lo que se vectoriza (y se indexa en BM25) es 'contexto + texto': el título da contexto a un fragmento suelto.

    - Texto normal: el contexto es la sección.
    - Audio y video: el contexto es el título del archivo (el rango de tiempo no dice nada sobre el contenido).
    - docling_hybrid: el texto YA viene contextualizado por el chunker (encabezados + texto): se vectoriza tal cual.
    """
    if chunk.metadata.get("contextualized"):
        return chunk.text
    return f"{chunk.metadata.get('embed_context', chunk.metadata['section'])}. {chunk.text}"
