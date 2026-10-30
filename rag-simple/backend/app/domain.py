"""Los tres objetos que viajan por todo el RAG: Document, Chunk y Hit."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Un archivo ya leído: su nombre y sus secciones (título, texto)."""

    source: str
    title: str
    sections: list[tuple[str, str]]


@dataclass
class Chunk:
    """Un fragmento: el trozo de texto que se vectoriza y se recupera, con metadatos para poder CITAR.

    `metadata` lleva al menos `source` (archivo) y `section` (encabezado).
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class Hit:
    """Un resultado de búsqueda: un fragmento y su puntaje (similitud coseno: 1 = idéntico)."""

    id: str
    text: str
    metadata: dict
    score: float
