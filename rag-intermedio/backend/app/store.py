"""Almacén vectorial con filtros por metadatos.

    PineconeStore  base vectorial ADMINISTRADA y dedicada: se usa por API (índice → namespace → registros), sin SQL ni
                   servidores. Escala sola; cobra por uso. Es de consistencia EVENTUAL: lo recién escrito puede tardar
                   unos segundos en aparecer en las búsquedas.
    MemoryStore    una matriz de NumPy en la RAM, para pruebas y para aprender.

El FILTRO por metadatos se aplica DENTRO de la búsqueda (no después, y menos en el prompt): `audiences=["negocio",
"general"]` significa "solo fragmentos para clientes Negocio o de uso general". Un filtro en el prompt no es un control
de acceso: el dato ya llegó al modelo. Filtrar en el almacén evita que llegue.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .config import Settings, get_settings
from .domain import Chunk, Hit


class StoreError(RuntimeError):
    """El almacén no está disponible o no está configurado."""


class VectorStore(Protocol):
    name: str

    def reset(self, dimension: int) -> None: ...
    def upsert(self, chunks: list[Chunk]) -> None: ...
    def search(self, vector: list[float], k: int = 5, audiences: list[str] | None = None) -> list[Hit]: ...
    def count(self) -> int: ...
    def drop(self) -> None: ...


class MemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def reset(self, dimension: int) -> None:
        self._chunks, self._matrix = [], np.empty((0, dimension), dtype=np.float32)

    def upsert(self, chunks: list[Chunk]) -> None:
        by_id = {c.id: c for c in self._chunks}
        by_id.update({c.id: c for c in chunks})
        self._chunks = list(by_id.values())
        matrix = np.array([c.vector for c in self._chunks], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def search(self, vector: list[float], k: int = 5, audiences: list[str] | None = None) -> list[Hit]:
        if not self._chunks:
            return []
        q = np.array(vector, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scores = self._matrix @ q
        out: list[Hit] = []
        for i in np.argsort(-scores):
            c = self._chunks[i]
            if audiences is None or c.metadata.get("audience") in audiences:
                out.append(Hit(c.id, c.text, c.metadata, float(scores[i])))
            if len(out) == k:
                break
        return out

    def count(self) -> int:
        return len(self._chunks)

    def drop(self) -> None:
        self._chunks, self._matrix = [], np.empty((0, 0), dtype=np.float32)


class PineconeStore:
    name = "pinecone"

    def __init__(self, settings: Settings) -> None:
        if not settings.pinecone_api_key:
            raise StoreError("Falta PINECONE_API_KEY. Crea tu cuenta en Pinecone y copia tu API key en el .env "
                             "(o usa STORE=memory para probar sin cuenta).")
        self.s = settings
        self._pc = None
        self._index = None

    def _client(self):
        if self._pc is None:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=self.s.pinecone_api_key)
        return self._pc

    def _idx(self):
        if self._index is None:
            pc = self._client()
            if not pc.has_index(self.s.pinecone_index):
                raise StoreError(f"El índice '{self.s.pinecone_index}' no existe todavía: ejecuta la ingesta.")
            self._index = pc.Index(host=pc.describe_index(self.s.pinecone_index).host)
        return self._index

    @staticmethod
    def _is_conflict(exc: Exception) -> bool:
        """409: el recurso ya existe. Se detecta por clase o por código para no depender de la versión del SDK."""
        return type(exc).__name__ == "ConflictError" or getattr(exc, "status_code", None) == 409 or getattr(exc, "status", None) == 409

    def _ensure_index(self, dimension: int) -> None:
        """Crea el índice SOLO si no existe. Es idempotente: llamarla muchas veces, o desde dos procesos a la vez, da igual.

        Comprobar y luego crear (`has_index` + `create_index`) es una carrera: si otra ingesta lo crea entre las dos
        llamadas, Pinecone responde 409 (conflicto). Ese 409 significa justo lo que queríamos (el índice existe), así que
        se trata como un no-op.
        """
        from pinecone import ServerlessSpec

        pc = self._client()
        if not pc.has_index(self.s.pinecone_index):
            try:
                pc.create_index(name=self.s.pinecone_index, dimension=dimension, metric="cosine",
                                spec=ServerlessSpec(cloud=self.s.pinecone_cloud, region=self.s.pinecone_region))
            except Exception as exc:  # noqa: BLE001
                if not self._is_conflict(exc):
                    raise
        for _ in range(60):                                      # un índice recién creado tarda unos segundos en estar listo
            if pc.describe_index(self.s.pinecone_index).status.ready:
                break
            time.sleep(2)

    def reset(self, dimension: int) -> None:
        """Asegura que el índice exista (sin fallar si ya existe) y VACÍA el namespace de este proyecto.

        La dimensión de un índice NO se puede cambiar después de crearlo: si usas otro modelo de embeddings, usa otro
        índice (PINECONE_INDEX) o bórralo desde la consola de Pinecone.
        """
        pc = self._client()
        try:
            self._ensure_index(dimension)
            self._index = pc.Index(host=pc.describe_index(self.s.pinecone_index).host)
            existing = self._index.describe_index_stats().dimension
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"No se pudo preparar Pinecone ({type(exc).__name__}): {exc}") from exc
        if existing and existing != dimension:
            raise StoreError(f"El índice '{self.s.pinecone_index}' tiene {existing} dimensiones, pero tu modelo de embeddings "
                             f"produce {dimension}. Usa otro índice (PINECONE_INDEX) o bórralo en la consola de Pinecone.")
        self._clear_namespace()

    def _clear_namespace(self) -> None:
        try:
            self._idx().delete(delete_all=True, namespace=self.s.pinecone_namespace)
        except Exception as exc:  # noqa: BLE001 - un namespace vacío o inexistente no es un error
            if "not found" not in str(exc).lower():
                raise

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        """Pinecone solo admite texto, números, booleanos y listas de texto como metadatos. El texto va dentro, como un dato más."""
        clean = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in chunk.metadata.items()}
        return {**clean, "text": chunk.text}

    def upsert(self, chunks: list[Chunk]) -> None:
        records = [{"id": c.id, "values": [float(x) for x in c.vector], "metadata": self._metadata(c)} for c in chunks]
        for i in range(0, len(records), 100):                    # un upsert admite hasta 1000 registros y 2 MB
            self._idx().upsert(vectors=records[i : i + 100], namespace=self.s.pinecone_namespace)
        deadline = time.time() + 30                              # consistencia eventual: esperar a que se vean
        while time.time() < deadline and self.count() < len(chunks):
            time.sleep(1)

    def search(self, vector: list[float], k: int = 5, audiences: list[str] | None = None) -> list[Hit]:
        flt = {"audience": {"$in": audiences}} if audiences else None
        resp = self._idx().query(vector=[float(x) for x in vector], top_k=k, include_metadata=True,
                                 namespace=self.s.pinecone_namespace, filter=flt)
        hits = []
        for m in resp.matches:
            meta = dict(m.metadata or {})
            text = meta.pop("text", "")
            hits.append(Hit(m.id, text, meta, float(m.score)))
        return hits

    def count(self) -> int:
        try:
            stats = self._idx().describe_index_stats()
        except StoreError:
            return 0
        summary = (stats.namespaces or {}).get(self.s.pinecone_namespace)
        return int(getattr(summary, "vector_count", 0) or 0) if summary is not None else 0

    def drop(self) -> None:
        self._clear_namespace()


def create_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return MemoryStore() if settings.store == "memory" else PineconeStore(settings)
