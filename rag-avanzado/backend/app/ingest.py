"""Leer el catálogo (Markdown) y partirlo en fragmentos.

El catálogo es corto (16 secciones), así que el fragmentado es sencillo: por oraciones con un máximo de palabras (cada
fragmento recuerda su sección) o una sección por fragmento. Lo interesante de este proyecto no está aquí sino en lo que se
hace DESPUÉS con el texto: vectorizarlo con o sin frase de contexto y extraerle un grafo.
"""

from __future__ import annotations

import re
from pathlib import Path

from .domain import Chunk, Document

STRATEGIES = ("sentences", "section")


def read_markdown(path: Path) -> Document:
    """Lee un .md. El `# ` inicial es el título del documento y cada `## ` abre una sección."""
    title = path.stem.replace("_", " ").capitalize()
    sections: list[tuple[str, str]] = []
    heading, buffer = "Introducción", []

    def close() -> None:
        if "".join(buffer).strip():
            sections.append((heading, "\n".join(buffer).strip()))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            close()
            heading, buffer = line[3:].strip(), []
        elif line.startswith("# "):
            title = line[2:].strip()
        else:
            buffer.append(line)
    close()
    return Document(source=path.name, title=title, sections=sections)


def load_documents(folder: Path) -> list[Document]:
    files = sorted(folder.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No hay archivos .md en {folder}")
    return [read_markdown(f) for f in files]


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip()) if p.strip()]


def _group(text: str, max_words: int, overlap: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text):
        if current and sum(len(s.split()) for s in current) + len(sentence.split()) > max_words:
            groups.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            while current and sum(len(s.split()) for s in current) + len(sentence.split()) > max_words:
                current.pop(0)
        current.append(sentence)
    if current and (not groups or " ".join(current) != groups[-1]):
        groups.append(" ".join(current))
    return groups


def chunk_documents(docs: list[Document], strategy: str = "sentences", max_words: int = 25, overlap: int = 0) -> list[Chunk]:
    """Fragmentos con id estable (`servicios_ti-7`) y metadatos para citar: `source`, `section`, `n`."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy debe ser una de: {', '.join(STRATEGIES)}")
    if max_words < 10:
        raise ValueError("max_words debe ser al menos 10")
    chunks: list[Chunk] = []
    for doc in docs:
        stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(doc.source).stem)
        for heading, text in doc.sections:
            pieces = [" ".join(text.split())] if strategy == "section" else _group(text, max_words, overlap)
            for piece in pieces:
                chunks.append(Chunk(id=f"{stem}-{len(chunks)}", text=piece,
                                    metadata={"source": doc.source, "section": heading, "n": len(chunks)}))
    return chunks


def text_to_embed(chunk: Chunk, heading: bool = True, with_context: bool = False) -> str:
    """Lo que se vectoriza: [frase de contexto] + [sección.] + texto.

    El título de la sección ya es un "encabezado contextual" barato (se vio antes). La frase del LLM (Contextual Retrieval)
    va ADELANTE y se compara contra ese mismo punto de partida. Con `heading=False` se vectoriza el texto crudo: es donde
    más se nota lo que aporta el contexto, porque un fragmento como «Se ejecuta en el Centro de datos Norte.» no dice de qué servicio habla.
    """
    parts = []
    if with_context and chunk.context:
        parts.append(chunk.context)
    if heading:
        parts.append(f"{chunk.metadata['section']}.")
    parts.append(chunk.text)
    return " ".join(parts)
