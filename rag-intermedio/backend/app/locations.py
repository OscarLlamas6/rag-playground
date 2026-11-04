"""Dónde está un fragmento dentro de su fuente: página (PDF) o tiempo (audio y video). Sin dependencias pesadas.

Las citas del RAG dicen "manual.pdf, p. 12" o "charla.mp3, 12:30–13:10": así quien lee puede ir a comprobarlo.
"""

from __future__ import annotations


def format_time(seconds: float) -> str:
    """12:30 para 750 s; con horas, 1:02:03. Se redondea hacia abajo: el instante citado nunca queda después del real."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_range(start: float, end: float) -> str:
    return f"{format_time(start)}–{format_time(end)}"


def location(meta: dict) -> str:
    """'p. 12', 'p. 12–13', '12:30–13:10' o '' si el fragmento no tiene posición (md, html, docx)."""
    if meta.get("start_s") is not None and meta.get("end_s") is not None:
        return format_range(float(meta["start_s"]), float(meta["end_s"]))
    page = meta.get("page")
    if page:
        end = meta.get("page_end")
        return f"p. {page}–{end}" if end and end != page else f"p. {page}"
    return ""
