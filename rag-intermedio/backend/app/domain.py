"""Los objetos que viajan por todo el RAG: Document, Chunk y Hit."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Un archivo ya leído, sin importar su formato: nombre, tipo (md, html, docx) y secciones (título, texto)."""

    source: str
    title: str
    type: str
    sections: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """Un fragmento con sus metadatos.

    `metadata` lleva: `source`, `section`, `type` (md/html/docx), `audience` (residencial/negocio/general) y, en el
    patrón padre-hijo, `parent_id` y `parent_text` (la sección completa a la que pertenece este fragmento pequeño).
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class Hit:
    """Un resultado de búsqueda: un fragmento y su puntaje (el significado del puntaje depende de quién lo calculó)."""

    id: str
    text: str
    metadata: dict
    score: float
