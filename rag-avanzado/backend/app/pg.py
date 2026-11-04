"""Conexión a PostgreSQL con resiliencia: una conexión larga, un candado y un reintento si se cayó.

La comparten el almacén vectorial (`store.py`) y el del grafo (`graphstore.py`). Es el patrón de `NeonStore` de rag-simple:
la conexión larga puede morir (se reinició el contenedor, el servidor cerró la sesión); eso no debe verse como "almacén vacío",
así que se descarta, se abre otra y se reintenta UNA vez (todas las operaciones son idempotentes). Cualquier otro error de la
base se convierte en un StoreError con un mensaje claro.
"""

from __future__ import annotations

import re
import threading
import time


class StoreError(RuntimeError):
    """El almacén no está disponible o no está configurado."""


def safe_identifier(name: str) -> str:
    """Un nombre de tabla no se puede pasar como parámetro SQL: se valida para evitar inyección SQL."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", name):
        raise ValueError(f"Nombre de tabla no válido: {name!r} (usa letras, números y guion bajo)")
    return name


class PgConnection:
    def __init__(self, url: str) -> None:
        if not url:
            raise StoreError("Falta DATABASE_URL (o usa AVANZADO_STORE=memory para probar sin base de datos).")
        self.url = url
        self._conn = None
        self._lock = threading.RLock()   # FastAPI atiende en hilos; una conexión se usa de a una petición

    def _connection(self):
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            last: Exception | None = None
            for _ in range(5):               # el contenedor `db` puede estar terminando de arrancar
                try:
                    self._conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    time.sleep(1)
            else:
                raise StoreError(f"No se pudo conectar a PostgreSQL ({type(last).__name__}). Revisa DATABASE_URL y que el servicio "
                                 "`db` esté sano (docker compose ps; en tu máquina: task db:up).") from last
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")   # antes de registrar el tipo `vector`
            register_vector(self._conn)
        return self._conn

    def _drop(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001 - ya estaba rota
            pass
        self._conn = None

    def run(self, operation):
        """Ejecuta `operation(conexión)`; si la conexión estaba caída, abre otra y reintenta una vez."""
        import psycopg

        with self._lock:
            for attempt in (1, 2):
                try:
                    return operation(self._connection())
                except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
                    self._drop()
                    if attempt == 2:
                        raise StoreError(f"Se perdió la conexión con PostgreSQL ({type(exc).__name__}). Reintenta en unos segundos.") from exc
                except StoreError:
                    raise
                except psycopg.Error as exc:
                    if "different vector dimensions" in str(exc):
                        raise StoreError("Las tablas se crearon con otro modelo de embeddings (otra dimensión de vector). "
                                         "Ve a la pestaña Ingesta y vuelve a ingestar con el modelo actual.") from exc
                    raise StoreError(f"PostgreSQL respondió con un error ({type(exc).__name__}): {str(exc).splitlines()[0]}") from exc
