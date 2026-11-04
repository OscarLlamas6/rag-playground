"""Configuración leída de variables de entorno (y de archivos .env).

Orden de búsqueda del .env: `rag-avanzado/.env`, luego `rag-playground/.env` (raíz del repo) y por último el
que encuentre `python-dotenv` desde el directorio actual. Una variable ya definida en el entorno nunca se pisa.

El almacén se elige con `AVANZADO_STORE` (y no con `STORE`): ese nombre ya lo usa `rag-simple` para `neon | memory`
y el `.env` de la raíz se comparte entre proyectos.

Las variables se leen cada vez que se llama a `get_settings()` (no al importar el módulo): así las pruebas
pueden cambiarlas con `monkeypatch` sin reiniciar nada.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
REPO_DIR = PROJECT_DIR.parent

for _env in (PROJECT_DIR / ".env", REPO_DIR / ".env"):
    load_dotenv(_env)
load_dotenv(find_dotenv(usecwd=True))

LLM_PROVIDERS = ("anthropic", "openai", "local", "fake")
EMBEDDING_PROVIDERS = ("local", "openai", "fake")
STORES = ("postgres", "memory")

DEFAULT_LLM_MODEL = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna"}
DEFAULT_EMBEDDING_MODEL = {"local": "text-embedding-nomic-embed-text-v2-moe", "openai": "text-embedding-3-small"}
# Para correr el backend en tu máquina contra el `db` del compose (publicado solo en 127.0.0.1:5434).
DEFAULT_DATABASE_URL = "postgresql://rag:rag@127.0.0.1:5434/rag"


def _clean(value: str | None) -> str:
    return (value or "").split("#")[0].strip()


def _local_url() -> str:
    """URL del servidor local (LM Studio). Dentro de un contenedor `localhost` es el propio contenedor, no tu máquina:
    con RUNNING_IN_DOCKER=1 (lo fija docker-compose) se cambia por `host.docker.internal`, que apunta al anfitrión."""
    url = _clean(os.getenv("LOCAL_LLM_URL")) or "http://localhost:1234/v1"
    if os.getenv("RUNNING_IN_DOCKER") == "1":
        url = re.sub(r"//(localhost|127\.0\.0\.1)", "//host.docker.internal", url)
    return url


def _llm_model(provider: str) -> str:
    """Con `local` manda LOCAL_LLM_MODEL; con anthropic u openai, LLM_MODEL o el modelo económico por defecto."""
    if provider == "local":
        return _clean(os.getenv("LOCAL_LLM_MODEL")) or _clean(os.getenv("LLM_MODEL"))
    return _clean(os.getenv("LLM_MODEL")) or DEFAULT_LLM_MODEL.get(provider, "fake")


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _clean(os.getenv(name)) or default
    value = value.lower()
    if value not in allowed:
        raise ValueError(f"{name}='{value}' no es válido. Usa uno de: {', '.join(allowed)}.")
    return value


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    embeddings_provider: str
    embeddings_model: str
    embeddings_dimensions: int | None
    store: str
    database_url: str
    local_url: str
    docs_dir: Path
    graph_file: Path
    eval_files: tuple[Path, ...]
    prompts_dir: Path
    cache_dir: Path


def get_settings() -> Settings:
    llm = _choice("LLM_PROVIDER", "anthropic", LLM_PROVIDERS)
    emb = _choice("EMBEDDINGS_PROVIDER", "local", EMBEDDING_PROVIDERS)
    dims = _clean(os.getenv("EMBEDDINGS_DIMENSIONS"))
    data = BACKEND_DIR / "data"
    return Settings(
        llm_provider=llm,
        llm_model=_llm_model(llm),
        embeddings_provider=emb,
        embeddings_model=_clean(os.getenv("EMBEDDINGS_MODEL")) or DEFAULT_EMBEDDING_MODEL.get(emb, "fake-hash"),
        embeddings_dimensions=int(dims) if dims.isdigit() else None,
        store=_choice("AVANZADO_STORE", "postgres", STORES),
        database_url=_clean(os.getenv("DATABASE_URL")) or DEFAULT_DATABASE_URL,
        local_url=_local_url(),
        docs_dir=Path(os.getenv("DOCS_DIR", data / "docs")),
        graph_file=Path(os.getenv("GRAPH_FILE", data / "graph" / "servicios_ti_grafo.json")),
        eval_files=tuple(Path(p) for p in os.getenv("EVAL_FILES", "").split(",") if p.strip())
        or (data / "eval" / "preguntas_curso.json", data / "eval" / "preguntas_extra.json"),
        prompts_dir=BACKEND_DIR / "prompts",
        cache_dir=Path(os.getenv("CACHE_DIR", BACKEND_DIR / ".cache")),
    )
