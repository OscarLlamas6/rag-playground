"""Caché local de fragmentos (sin vectores).

BM25 vive en memoria: se construye a partir del texto de los fragmentos. Pinecone guarda los vectores, no hay un
"índice BM25" en la nube; así que al ingestar se guarda también una copia de los fragmentos en un JSON local, y al
arrancar el servidor se reconstruye BM25 desde ahí. (Pinecone y otros ofrecen búsqueda híbrida nativa con vectores
dispersos; aquí se construye a mano para que se vea cómo funciona.)
"""

from __future__ import annotations

import json
from pathlib import Path

from .domain import Chunk


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks], ensure_ascii=False),
                    encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [Chunk(id=r["id"], text=r["text"], metadata=r["metadata"]) for r in json.loads(path.read_text(encoding="utf-8"))]
