"""Leer documentos Markdown y partirlos en fragmentos (chunking).

Un RAG no vectoriza el documento entero: vectoriza FRAGMENTOS. Un fragmento muy grande mezcla temas y su
vector se "diluye"; uno muy pequeño pierde contexto ("cuesta 39 dólares": ¿qué cuesta?). Por eso el
tamaño y la estrategia son decisiones de diseño que conviene MEDIR (ver evaluation.py).

Estrategias:
    section    un fragmento por sección (`## Encabezado`)
    sentences  junta oraciones hasta `max_words`, repitiendo las últimas `overlap` oraciones en el siguiente
               fragmento: el SOLAPE evita cortar una idea por la mitad
    fixed      ventanas de `max_words` palabras, sin respetar la gramática
"""

from __future__ import annotations

import re
from pathlib import Path

from .domain import Chunk, Document

STRATEGIES = ("sentences", "section", "fixed")


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
    """Todos los .md de una carpeta, ordenados por nombre."""
    files = sorted(folder.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No hay archivos .md en {folder}")
    return [read_markdown(f) for f in files]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc: Document, strategy: str = "sentences", max_words: int = 70, overlap: int = 1) -> list[Chunk]:
    """Parte un documento en fragmentos con metadatos (`source`, `section`, `n`)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy debe ser una de: {', '.join(STRATEGIES)}")
    if max_words < 10:
        raise ValueError("max_words debe ser al menos 10")

    pieces: list[tuple[str, str]] = []  # (sección, texto)
    for heading, text in doc.sections:
        if strategy == "section":
            pieces.append((heading, " ".join(text.split())))
        elif strategy == "fixed":
            words = text.split()
            step = max(1, max_words - overlap * 10)
            for i in range(0, len(words), step):
                pieces.append((heading, " ".join(words[i : i + max_words])))
                if i + max_words >= len(words):
                    break
        else:
            current: list[str] = []
            first_of_section = len(pieces)
            for sentence in _sentences(text):
                size = sum(len(s.split()) for s in current) + len(sentence.split())
                if current and size > max_words:
                    pieces.append((heading, " ".join(current)))
                    current = current[-overlap:] if overlap else []
                    # el solape no puede hacer que el fragmento nuevo se pase del tamaño máximo
                    while current and sum(len(s.split()) for s in current) + len(sentence.split()) > max_words:
                        current.pop(0)
                current.append(sentence)
            if current and (len(pieces) == first_of_section or " ".join(current) != pieces[-1][1]):
                pieces.append((heading, " ".join(current)))

    stem = Path(doc.source).stem
    return [
        Chunk(id=f"{stem}#{n}", text=text, metadata={"source": doc.source, "section": heading, "n": n})
        for n, (heading, text) in enumerate(pieces)
    ]


def text_to_embed(chunk: Chunk) -> str:
    """Lo que se vectoriza es 'sección + texto': el título da contexto a un fragmento que, solo, sería ambiguo."""
    return f"{chunk.metadata['section']}. {chunk.text}"
