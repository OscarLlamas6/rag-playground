"""Prompts versionados: viven en archivos (`prompts/<nombre>/<version>.j2`), no dentro del código.

Así se revisan en un pull request, se comparan versiones con la evaluación y se vuelve atrás si una versión empeora.
Usa Jinja2 con StrictUndefined: si falta una variable, falla en voz alta en lugar de enviar un prompt a medias.
"""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import get_settings
from .locations import location


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(get_settings().prompts_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["location"] = lambda metadata: location(metadata)       # 'p. 12' o '12:30–13:10' ('' si no hay)
    return env


def versions(name: str) -> list[str]:
    return sorted(p.stem for p in (get_settings().prompts_dir / name).glob("v*.j2"))


def render(name: str, version: str, **variables) -> str:
    return _env().get_template(f"{name}/{version}.j2").render(**variables).strip()
