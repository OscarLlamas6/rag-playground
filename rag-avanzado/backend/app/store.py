"""Almacén de FRAGMENTOS con sus vectores.

    PgVectorStore  PostgreSQL con la extensión pgvector (imagen oficial `pgvector/pgvector`, en el docker-compose). Una tabla normal
                   con columnas `vector(N)`; se consulta con SQL (`ORDER BY embedding <=> consulta`). Los vectores viven junto a
                   las demás tablas del proyecto (entidades, relaciones, comunidades): una sola base para todo.
    MemoryStore    Una matriz de NumPy en la RAM. Sin Docker, sin persistencia. Para pruebas y para `task demo`.

Cada fragmento guarda DOS vectores: `embedding` (el texto normal) y `embedding_ctx` (con la frase de contexto delante; solo
si la ingesta fue contextual). Así la evaluación puede comparar "con" y "sin" contexto sobre la MISMA ingesta.

Este proyecto es dueño de sus tablas: `reset(dimension)` las borra y las recrea.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Settings, get_settings
from .domain import Chunk, Item
from .pg import PgConnection, StoreError, safe_identifier

__all__ = ["StoreError", "ChunkStore", "MemoryStore", "PgVectorStore", "create_store", "chunk_item"]


def chunk_item(chunk: Chunk, score: float = 0.0) -> Item:
    return Item(id=chunk.id, kind="chunk", text=chunk.text, label=chunk.metadata.get("section", "?"), score=score,
                meta={"source": chunk.metadata.get("source", "?"), "section": chunk.metadata.get("section", "?"), "context": chunk.context})


class ChunkStore(Protocol):
    name: str

    def reset(self, dimension: int) -> None: ...
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def search(self, vector: list[float], k: int = 4, contextual: bool = False) -> list[Item]: ...
    def by_section(self, section: str) -> list[Chunk]: ...
    def count(self) -> int: ...
    def count_contextual(self) -> int: ...
    def drop(self) -> None: ...


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms        # con vectores de longitud 1, coseno = producto punto


class MemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._plain = np.empty((0, 0), dtype=np.float32)
        self._ctx = np.empty((0, 0), dtype=np.float32)

    def reset(self, dimension: int) -> None:
        self._chunks = []
        self._plain = self._ctx = np.empty((0, dimension), dtype=np.float32)

    def upsert(self, chunks: list[Chunk]) -> None:
        by_id = {c.id: c for c in self._chunks}
        for c in chunks:
            by_id[c.id] = c
        self._chunks = list(by_id.values())
        self._plain = _unit(np.array([c.vector for c in self._chunks], dtype=np.float32))
        self._ctx = _unit(np.array([c.vector_ctx if c.vector_ctx is not None else np.zeros(len(c.vector)) for c in self._chunks], dtype=np.float32))

    def search(self, vector: list[float], k: int = 4, contextual: bool = False) -> list[Item]:
        if not self._chunks:
            return []
        if contextual and not self.count_contextual():
            raise StoreError("La ingesta no tiene frases de contexto: reingesta con «contextual» activado.")
        q = np.array(vector, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scores = (self._ctx if contextual else self._plain) @ q
        return [chunk_item(self._chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k]]

    def by_section(self, section: str) -> list[Chunk]:
        return [c for c in self._chunks if c.metadata.get("section") == section]

    def count(self) -> int:
        return len(self._chunks)

    def count_contextual(self) -> int:
        return sum(c.vector_ctx is not None for c in self._chunks)

    def drop(self) -> None:
        self.reset(0)


class PgVectorStore:
    name = "postgres"

    def __init__(self, url: str, table: str = "av_chunks") -> None:
        self.db = PgConnection(url)
        self.table = safe_identifier(table)

    def reset(self, dimension: int) -> None:
        def go(c):
            c.execute(f"DROP TABLE IF EXISTS {self.table}")
            c.execute(f"CREATE TABLE {self.table} (id text PRIMARY KEY, text text NOT NULL, context text NOT NULL DEFAULT '', "
                      "metadata jsonb NOT NULL DEFAULT '{}', "
                      f"embedding vector({int(dimension)}) NOT NULL, embedding_ctx vector({int(dimension)}))")
        self.db.run(go)

    def upsert(self, chunks: list[Chunk]) -> None:
        from psycopg.types.json import Jsonb

        rows = [(c.id, c.text, c.context, Jsonb(c.metadata), np.array(c.vector, dtype=np.float32),
                 None if c.vector_ctx is None else np.array(c.vector_ctx, dtype=np.float32)) for c in chunks]

        def go(c):
            with c.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} (id, text, context, metadata, embedding, embedding_ctx) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, context = EXCLUDED.context, metadata = EXCLUDED.metadata, "
                    "embedding = EXCLUDED.embedding, embedding_ctx = EXCLUDED.embedding_ctx", rows)
        self.db.run(go)

    def search(self, vector: list[float], k: int = 4, contextual: bool = False) -> list[Item]:
        if contextual and not self.count_contextual():
            raise StoreError("La ingesta no tiene frases de contexto: reingesta con «contextual» activado.")
        q = np.array(vector, dtype=np.float32)
        column = "embedding_ctx" if contextual else "embedding"
        # `<=>` es la DISTANCIA coseno de pgvector (0 = idénticos); la similitud es 1 - distancia.
        rows = self.db.run(lambda c: c.execute(
            f"SELECT id, text, context, metadata, 1 - ({column} <=> %s) FROM {self.table} "
            f"WHERE {column} IS NOT NULL ORDER BY {column} <=> %s LIMIT %s", (q, q, k)).fetchall())
        return [chunk_item(Chunk(i, t, m, ctx), float(s)) for i, t, ctx, m, s in rows]

    def by_section(self, section: str) -> list[Chunk]:
        rows = self.db.run(lambda c: c.execute(
            f"SELECT id, text, context, metadata FROM {self.table} WHERE metadata->>'section' = %s ORDER BY (metadata->>'n')::int",
            (section,)).fetchall())
        return [Chunk(i, t, m, ctx) for i, t, ctx, m in rows]

    def _count(self, where: str = "") -> int:
        def go(c):
            if not c.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0]:
                return 0
            return c.execute(f"SELECT count(*) FROM {self.table} {where}").fetchone()[0]
        return self.db.run(go)

    def count(self) -> int:
        return self._count()

    def count_contextual(self) -> int:
        return self._count("WHERE embedding_ctx IS NOT NULL")

    def drop(self) -> None:
        self.db.run(lambda c: c.execute(f"DROP TABLE IF EXISTS {self.table}"))


def create_store(settings: Settings | None = None) -> ChunkStore:
    settings = settings or get_settings()
    return MemoryStore() if settings.store == "memory" else PgVectorStore(settings.database_url)
