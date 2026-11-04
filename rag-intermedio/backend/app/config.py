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
DOCLING_OCR_MODES = ("auto", "off", "force")

DEFAULT_LLM_MODEL = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna"}
DEFAULT_VISION_MODEL = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna", "local": "google/gemma-4-12b-qat"}
DEFAULT_EMBEDDING_MODEL = {"local": "text-embedding-nomic-embed-text-v2-moe", "openai": "text-embedding-3-small"}


def _local_url() -> str:
    """URL del servidor local (LM Studio). Dentro de un contenedor `localhost` es el propio contenedor, no tu máquina:
    con RUNNING_IN_DOCKER=1 (lo fija docker-compose) se cambia por `host.docker.internal`, que apunta al anfitrión."""
    url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1").split("#")[0].strip()
    if os.getenv("RUNNING_IN_DOCKER") == "1":
        url = re.sub(r"//(localhost|127\.0\.0\.1)", "//host.docker.internal", url)
    return url


def _llm_model(provider: str) -> str:
    """Modelo que redacta las respuestas.

    - Con `local` manda LOCAL_LLM_MODEL (el identificador que muestra LM Studio). Así puedes dejar `LLM_MODEL=gpt-6-luna`
      en el .env y cambiar de proveedor tocando solo LLM_PROVIDER, sin pedirle a LM Studio un modelo de OpenAI.
    - Con anthropic u openai manda LLM_MODEL y, si está vacío, el modelo económico por defecto de ese proveedor.
    """
    clean = lambda value: (value or "").split("#")[0].strip()  # noqa: E731
    if provider == "local":
        return clean(os.getenv("LOCAL_LLM_MODEL")) or clean(os.getenv("LLM_MODEL"))
    return clean(os.getenv("LLM_MODEL")) or DEFAULT_LLM_MODEL.get(provider, "fake")


def _clean(name: str, default: str = "") -> str:
    return os.getenv(name, default).split("#")[0].strip()


def _number(name: str, default: float, minimum: float = 0.0) -> float:
    raw = _clean(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}='{raw}' no es un número.") from exc
    if value < minimum:
        raise ValueError(f"{name}={raw} debe ser al menos {minimum:g}.")
    return value


def _default_tokenizer(embeddings_model: str) -> str:
    """El tokenizer del chunker debería ser el del modelo que vectoriza. Con nomic v2 es ese; con cualquier otro modelo
    (OpenAI, mxbai…) no hay un tokenizer de Hugging Face equivalente a mano, y se usa uno pequeño y rápido como aproximación."""
    if "nomic-embed-text-v2" in embeddings_model.lower():
        return "nomic-ai/nomic-embed-text-v2-moe"
    return "sentence-transformers/all-MiniLM-L6-v2"


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
    # ── ingesta moderna (grupo `media`) ──
    media_dir: Path
    media_cache_dir: Path           # transcripciones y conversiones de Docling ya hechas
    docling_max_pages: int          # tope de páginas por PDF (0 = sin tope)
    docling_ocr: str                # auto | off | force
    docling_min_chars: int          # una página con menos caracteres que esto cuenta como "sin texto"
    docling_tokenizer: str
    whisper_model: str
    whisper_language: str | None    # None = detectar el idioma
    whisper_prompt: str | None      # vocabulario para orientar a Whisper (nombres propios, siglas)
    media_window_s: float           # duración de cada fragmento de transcripción
    media_overlap_s: float
    frame_captions: bool            # EXPERIMENTAL: describir fotogramas de video con un modelo con visión
    frame_interval_s: float
    vision_provider: str
    vision_model: str


def get_settings() -> Settings:
    llm = _choice("LLM_PROVIDER", "anthropic", LLM_PROVIDERS)
    emb = _choice("EMBEDDINGS_PROVIDER", "local", EMBEDDING_PROVIDERS)
    llm_model = _llm_model(llm)
    emb_model = os.getenv("EMBEDDINGS_MODEL", "").split("#")[0].strip() or DEFAULT_EMBEDDING_MODEL.get(emb, "fake-hash")
    vision = _choice("VISION_PROVIDER", "local", ("local", "openai", "anthropic"))
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
        media_dir=Path(os.getenv("MEDIA_DIR") or BACKEND_DIR / "data" / "media"),
        media_cache_dir=Path(os.getenv("MEDIA_CACHE_DIR") or BACKEND_DIR / ".cache" / "media"),
        docling_max_pages=int(_number("DOCLING_MAX_PAGES", 30)),
        docling_ocr=_choice("DOCLING_OCR", "auto", DOCLING_OCR_MODES),
        docling_min_chars=int(_number("DOCLING_MIN_CHARS", 40)),
        docling_tokenizer=_clean("DOCLING_TOKENIZER") or _default_tokenizer(emb_model),
        whisper_model=_clean("WHISPER_MODEL", "small") or "small",
        whisper_language=_clean("WHISPER_LANGUAGE") or None,
        whisper_prompt=_clean("WHISPER_PROMPT") or None,
        media_window_s=_number("MEDIA_WINDOW_SECONDS", 50, 10),
        media_overlap_s=_number("MEDIA_OVERLAP_SECONDS", 8),
        frame_captions=_clean("FRAME_CAPTIONS", "false").lower() in ("1", "true", "yes", "si", "sí"),
        frame_interval_s=_number("FRAME_INTERVAL_SECONDS", 60, 5),
        vision_provider=vision,
        vision_model=_clean("VISION_MODEL") or DEFAULT_VISION_MODEL.get(vision, "") or _clean("LOCAL_LLM_MODEL"),
    )
