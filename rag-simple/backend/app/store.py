"""Almacén vectorial: dónde se guardan los fragmentos con sus vectores y de dónde se buscan los más parecidos.

Dos implementaciones con la MISMA interfaz:

    NeonStore    PostgreSQL de Neon con la extensión pgvector. Una tabla normal con una columna `vector(N)`;
                 se consulta con SQL (`ORDER BY embedding <=> consulta`). Los vectores viven junto a tus datos.
    MemoryStore  Una matriz de NumPy en la RAM. Sin cuenta, sin persistencia. Para pruebas y para aprender.

Este playground es DUEÑO de su tabla: `reset(dimension)` la borra y la recrea. Así cambiar de modelo de embeddings
(otra dimensión) o de estrategia de fragmentado es simplemente volver a ingestar.
"""

from __future__ import annotations

import re
import threading
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
    def search(self, vector: list[float], k: int = 4) -> list[Hit]: ...
    def count(self) -> int: ...
    def drop(self) -> None: ...


class MemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def reset(self, dimension: int) -> None:
        self._chunks = []
        self._matrix = np.empty((0, dimension), dtype=np.float32)

    def upsert(self, chunks: list[Chunk]) -> None:
        by_id = {c.id: c for c in self._chunks}
        for c in chunks:
            by_id[c.id] = c
        self._chunks = list(by_id.values())
        matrix = np.array([c.vector for c in self._chunks], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms        # con vectores de longitud 1, coseno = producto punto

    def search(self, vector: list[float], k: int = 4) -> list[Hit]:
        if not self._chunks:
            return []
        q = np.array(vector, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scores = self._matrix @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self._chunks[i].id, self._chunks[i].text, self._chunks[i].metadata, float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._chunks)

    def drop(self) -> None:
        self._chunks, self._matrix = [], np.empty((0, 0), dtype=np.float32)


def _safe_identifier(name: str) -> str:
    """Un nombre de tabla no se puede pasar como parámetro SQL: se valida para evitar inyección SQL."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", name):
        raise ValueError(f"Nombre de tabla no válido: {name!r} (usa letras, números y guion bajo)")
    return name


class NeonStore:
    name = "neon"

    def __init__(self, url: str | None, table: str) -> None:
        if not url:
            raise StoreError("Falta NEON_DATABASE_URL. Crea un proyecto en Neon y copia su cadena de conexión en el .env "
                             "(o usa STORE=memory para probar sin base de datos).")
        self.url = url
        self.table = _safe_identifier(table)
        self._conn = None
        self._lock = threading.Lock()   # FastAPI atiende en hilos distintos; una conexión se usa de a una petición

    def _connection(self):
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            try:
                # Neon suspende el cómputo tras unos minutos sin uso: la primera conexión puede tardar unos segundos.
                self._conn = psycopg.connect(self.url, autocommit=True, connect_timeout=30)
            except Exception as exc:  # noqa: BLE001
                raise StoreError(f"No se pudo conectar a Neon ({type(exc).__name__}). Revisa NEON_DATABASE_URL.") from exc
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")   # antes de registrar el tipo `vector`
            register_vector(self._conn)
        return self._conn

    def _drop_connection(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001 - ya estaba rota
            pass
        self._conn = None

    def _run(self, operation):
        """Ejecuta `operation(conexión)` y se recupera de una conexión caída.

        Neon suspende el cómputo (o el servidor cierra la sesión) tras un rato sin actividad: la conexión larga muere y la
        siguiente consulta falla con OperationalError. Se descarta, se abre otra y se reintenta UNA vez; todas las
        operaciones son idempotentes. Cualquier otro error de la base se convierte en un StoreError con un mensaje claro
        (y no en un "almacén vacío" ni en un error 500 sin explicación).
        """
        import psycopg

        with self._lock:
            for attempt in (1, 2):
                try:
                    return operation(self._connection())
                except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
                    self._drop_connection()
                    if attempt == 2:
                        raise StoreError(f"Se perdió la conexión con Neon ({type(exc).__name__}). Reintenta en unos segundos.") from exc
                except StoreError:
                    raise
                except psycopg.Error as exc:
                    if "different vector dimensions" in str(exc):
                        raise StoreError(
                            f"La tabla '{self.table}' se creó con otro modelo de embeddings (otra dimensión de vector). "
                            "Ve a la pestaña Ingesta y vuelve a ingestar con el modelo actual."
                        ) from exc
                    raise StoreError(f"Neon respondió con un error ({type(exc).__name__}): {str(exc).splitlines()[0]}") from exc

    def reset(self, dimension: int) -> None:
        def go(c):
            c.execute(f"DROP TABLE IF EXISTS {self.table}")
            c.execute(
                f"CREATE TABLE {self.table} ("
                "  id text PRIMARY KEY,"
                "  text text NOT NULL,"
                "  metadata jsonb NOT NULL DEFAULT '{}',"
                f"  embedding vector({int(dimension)}) NOT NULL)"
            )
        self._run(go)

    def upsert(self, chunks: list[Chunk]) -> None:
        from psycopg.types.json import Jsonb

        rows = [(c.id, c.text, Jsonb(c.metadata), np.array(c.vector, dtype=np.float32)) for c in chunks]

        def go(c):
            with c.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                    "embedding = EXCLUDED.embedding",
                    rows,
                )
        self._run(go)

    def search(self, vector: list[float], k: int = 4) -> list[Hit]:
        q = np.array(vector, dtype=np.float32)
        # `<=>` es la DISTANCIA coseno de pgvector (0 = idénticos); la similitud es 1 - distancia.
        rows = self._run(lambda c: c.execute(
            f"SELECT id, text, metadata, 1 - (embedding <=> %s) FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
            (q, q, k),
        ).fetchall())
        return [Hit(i, t, m, float(s)) for i, t, m, s in rows]

    def count(self) -> int:
        def go(c):
            exists = c.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0]
            return c.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0] if exists else 0
        return self._run(go)

    def drop(self) -> None:
        self._run(lambda c: c.execute(f"DROP TABLE IF EXISTS {self.table}"))


def create_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    if settings.store == "memory":
        return MemoryStore()
    return NeonStore(settings.neon_url, settings.neon_table)
