"""Dependencias OPCIONALES: el grupo `media` (Docling, faster-whisper, av y torch CPU) pesa varios gigas.

La app base (Markdown, HTML, Word, BM25, vectores) NO las necesita. Los módulos que las usan las importan de forma
PEREZOSA (dentro de las funciones) y, si faltan, lanzan `MediaDepsMissing` con la instrucción exacta. La API la
convierte en un 503 legible; la ingesta, en cambio, omite ese archivo con una advertencia y sigue con los demás.
"""

from __future__ import annotations

import importlib.util
import shutil

INSTALL_HINT = "instala el grupo `media`: `task install:media` (o `cd backend && uv sync --group media`)"


class MediaError(RuntimeError):
    """Algo del flujo de PDF, audio o video no se puede hacer (falta una herramienta, un modelo no se descarga…)."""


class MediaDepsMissing(MediaError):
    """Falta una dependencia del grupo `media`."""


def module_available(name: str) -> bool:
    """¿Está instalado el paquete? Se pregunta sin importarlo (importar torch o Docling tarda segundos)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require(feature: str, *modules: str) -> None:
    """Falla con un mensaje claro si falta alguno de los paquetes que `feature` necesita."""
    missing = [m for m in modules if not module_available(m)]
    if missing:
        raise MediaDepsMissing(f"{feature} necesita {', '.join(missing)}: {INSTALL_HINT}.")


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def capabilities() -> dict:
    """Qué está disponible en esta instalación (lo muestra /api/status y la interfaz)."""
    return {"docling": module_available("docling"), "faster_whisper": module_available("faster_whisper"),
            "ffmpeg": ffmpeg_path() is not None}
