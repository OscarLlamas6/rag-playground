"""Configuración leída de variables de entorno (y de archivos .env).

Orden de búsqueda del .env: `rag-simple/.env`, luego `rag-playground/.env` (raíz del repo) y por último el
que encuentre `python-dotenv` desde el directorio actual. Una variable ya definida en el entorno nunca se pisa.

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
STORES = ("pinecone", "memory")

DEFAULT_LLM_MODEL = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna"}
DEFAULT_EMBEDDING_MODEL = {"local": "text-embedding-nomic-embed-text-v2-moe", "openai": "text-embedding-3-small"}


def _local_url() -> str:
    """URL del servidor local (LM Studio). Dentro de un contenedor `localhost` es el propio contenedor, no tu máquina:
    con RUNNING_IN_DOCKER=1 (lo fija docker-compose) se cambia por `host.docker.internal`, que apunta al anfitrión."""
    url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1").split("#")[0].strip()
    if os.getenv("RUNNING_IN_DOCKER") == "1":
        url = re.sub(r"//(localhost|127\.0\.0\.1)", "//host.docker.internal", url)
    return url


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
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
    pinecone_api_key: str | None
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    cache_file: Path
    local_url: str
    docs_dir: Path
    eval_file: Path
    prompts_dir: Path


def get_settings() -> Settings:
    llm = _choice("LLM_PROVIDER", "anthropic", LLM_PROVIDERS)
    emb = _choice("EMBEDDINGS_PROVIDER", "local", EMBEDDING_PROVIDERS)
    llm_model = os.getenv("LLM_MODEL") or (
        os.getenv("LOCAL_LLM_MODEL", "").split("#")[0].strip() if llm == "local" else DEFAULT_LLM_MODEL.get(llm, "fake")
    )
    emb_model = os.getenv("EMBEDDINGS_MODEL", "").split("#")[0].strip() or DEFAULT_EMBEDDING_MODEL.get(emb, "fake-hash")
    dims = os.getenv("EMBEDDINGS_DIMENSIONS", "").split("#")[0].strip()
    return Settings(
        llm_provider=llm,
        llm_model=llm_model,
        embeddings_provider=emb,
        embeddings_model=emb_model,
        embeddings_dimensions=int(dims) if dims.isdigit() else None,
        store=_choice("STORE", "pinecone", STORES),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index=os.getenv("PINECONE_INDEX", "rag-playground").split("#")[0].strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "rag-intermedio").split("#")[0].strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").split("#")[0].strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").split("#")[0].strip(),
        cache_file=Path(os.getenv("CHUNKS_CACHE", BACKEND_DIR / ".cache" / "chunks.json")),
        local_url=_local_url(),
        docs_dir=Path(os.getenv("DOCS_DIR", BACKEND_DIR / "data" / "docs")),
        eval_file=Path(os.getenv("EVAL_FILE", BACKEND_DIR / "data" / "eval" / "preguntas.json")),
        prompts_dir=BACKEND_DIR / "prompts",
    )
