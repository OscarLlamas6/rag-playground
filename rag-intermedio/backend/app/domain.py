"""Los objetos que viajan por todo el RAG: Document, Chunk y Hit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Un archivo ya leído, sin importar su formato: nombre, tipo (md, html, docx, pdf, audio, video) y secciones (título, texto).

    Campos opcionales de la ingesta moderna:
      `section_meta`  un dict por sección con su posición (`page`/`page_end` en PDF; `start_s`/`end_s` en audio y video).
      `info`          el reporte del archivo: si usó OCR, páginas, segundos, idioma, si vino de caché…
      `native`        el objeto nativo de Docling (para el HybridChunker). No se guarda ni se serializa.
      `skipped`       True si no se pudo leer (p. ej. falta el grupo `media`); la ingesta lo omite y lo avisa.
    """

    source: str
    title: str
    type: str
    sections: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)
    section_meta: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    native: Any = None
    skipped: bool = False


@dataclass
class Chunk:
    """Un fragmento con sus metadatos.

    `metadata` lleva: `source`, `section`, `type` (md/html/docx/pdf/audio/video), `audience` (residencial/negocio/general) y,
    en el patrón padre-hijo, `parent_id` y `parent_text` (la sección completa a la que pertenece este fragmento pequeño).
    Según la fuente suma `page`/`page_end` (PDF) o `media`, `start_s`, `end_s` (audio y video).
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
