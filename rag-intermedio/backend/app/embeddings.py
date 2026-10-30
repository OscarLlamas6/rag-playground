"""Vectorizar texto: convertir cada texto en una lista de números cuya cercanía refleja el parecido de significado.

Proveedores (variable EMBEDDINGS_PROVIDER):
    local   un servidor compatible con OpenAI en tu equipo (LM Studio). Gratis. Es el modo recomendado.
    openai  la API de OpenAI (text-embedding-3-small): fracciones de centavo.
    fake    vectores calculados con un hash de palabras. NO entienden significado (solo coinciden las palabras
            repetidas): sirven para probar el flujo sin servidor ni cuenta, y para las pruebas automáticas.

Algunos modelos exigen un PREFIJO distinto para documentos y consultas (nomic: `search_document: ` y
`search_query: `). Si lo olvidas, la calidad baja sin avisar, así que se aplica aquí, en un solo lugar.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata

from .config import Settings, get_settings

PREFIXES = {
    "nomic": ("search_document: ", "search_query: "),
    "mxbai": ("", "Represent this sentence for searching relevant passages: "),
}
FAKE_DIMENSIONS = 768  # igual que nomic: así el mismo índice sirve para el modo real y el modo fake


class EmbeddingError(RuntimeError):
    """El proveedor de embeddings no está disponible (servidor apagado, llave faltante…)."""


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _prefixes(model: str) -> tuple[str, str]:
    for key, pair in PREFIXES.items():
        if key in model.lower():
            return pair
    return ("", "")


def _fake_vector(text: str) -> list[float]:
    vec = [0.0] * FAKE_DIMENSIONS
    for token in re.findall(r"[a-z0-9]+", strip_accents(text).lower()):
        if len(token) < 3:
            continue
        digest = hashlib.md5(token[:6].encode()).digest()          # raíz de 6 letras: 'cancelar' ≈ 'cancelación'
        index = int.from_bytes(digest[:4], "big") % FAKE_DIMENSIONS
        vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _client(settings: Settings):
    import openai

    if settings.embeddings_provider == "local":
        return openai.OpenAI(base_url=settings.local_url, api_key="no-se-usa", max_retries=1, timeout=120)
    if not os.getenv("OPENAI_API_KEY"):
        raise EmbeddingError("Falta OPENAI_API_KEY (EMBEDDINGS_PROVIDER=openai).")
    return openai.OpenAI(max_retries=2)


def embed(texts: list[str], kind: str = "document", batch: int = 32) -> list[list[float]]:
    """Un vector por texto. `kind` es "document" (lo que guardas) o "query" (lo que buscas)."""
    if kind not in ("document", "query"):
        raise ValueError("kind debe ser 'document' o 'query'")
    settings = get_settings()
    if settings.embeddings_provider == "fake":
        return [_fake_vector(t) for t in texts]

    doc_prefix, query_prefix = _prefixes(settings.embeddings_model)
    prefix = doc_prefix if kind == "document" else query_prefix
    extra = {"dimensions": settings.embeddings_dimensions} if settings.embeddings_provider == "openai" and settings.embeddings_dimensions else {}
    client = _client(settings)
    out: list[list[float]] = []
    try:
        for i in range(0, len(texts), batch):
            chunk = [prefix + t for t in texts[i : i + batch]]
            response = client.embeddings.create(model=settings.embeddings_model, input=chunk, **extra)
            out.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001 - convertimos cualquier fallo del SDK en un mensaje claro
        hint = (f"Revisa tu llave y tu conexión, y que EMBEDDINGS_MODEL ('{settings.embeddings_model}') sea un modelo de OpenAI, "
                "por ejemplo text-embedding-3-small (si dejaste el de LM Studio, cámbialo).")
        if settings.embeddings_provider == "local":
            hint = f"Abre LM Studio, activa el servidor local y carga el modelo '{settings.embeddings_model}'. URL usada: {settings.local_url}."
            if os.getenv("RUNNING_IN_DOCKER") == "1":
                hint += (" Desde un contenedor, LM Studio solo es alcanzable si acepta conexiones de la red local "
                         "(LM Studio → Developer → «Serve on Local Network») o si usas docker-compose.host.yml (ver el README).")
        raise EmbeddingError(f"No se pudo vectorizar con '{settings.embeddings_provider}' ({type(exc).__name__}). {hint}") from exc
    return out


def embed_query(text: str) -> list[float]:
    return embed([text], kind="query")[0]
