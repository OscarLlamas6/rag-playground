"""PostgreSQL (el servicio `db`) puede reiniciarse: la conexión larga muere. Eso no debe verse como "almacén vacío": se reconecta
una vez y, si no se puede, se explica. Mismo patrón que NeonStore en rag-simple, probado sin base de datos real."""

import psycopg
import pytest

from app import main
from app.pg import PgConnection, StoreError
from app.store import MemoryStore, PgVectorStore


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return self.rows


class FakeConnection:
    closed = False

    def __init__(self, *script):
        self.script = list(script)          # cada paso: una excepción (se lanza) o las filas que devuelve

    def execute(self, sql, params=None):
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return Result(step)

    def close(self):
        self.closed = True


def store_with(*connections):
    store = PgVectorStore("postgresql://rag:rag@host/rag")
    pending = iter(connections)
    store.db._connection = lambda: next(pending)      # cada intento obtiene "una conexión nueva", como tras descartar la rota
    return store


def test_a_dead_connection_is_replaced_and_the_operation_retried():
    dead = FakeConnection(psycopg.OperationalError("server closed the connection unexpectedly"))
    alive = FakeConnection([("av_chunks",)], [(33,)])
    assert store_with(dead, alive).count() == 33


def test_if_reconnecting_also_fails_the_error_says_so():
    store = store_with(FakeConnection(psycopg.OperationalError("caída")), FakeConnection(psycopg.OperationalError("caída")))
    with pytest.raises(StoreError, match="Se perdió la conexión con PostgreSQL"):
        store.count()


def test_a_vector_dimension_mismatch_tells_you_to_reingest():
    store = store_with(FakeConnection(psycopg.errors.DataException("different vector dimensions 1536 and 768")))
    with pytest.raises(StoreError, match="pestaña Ingesta"):
        store.search([0.1] * 768)


def test_other_database_errors_become_a_readable_store_error():
    store = store_with(FakeConnection(psycopg.errors.UndefinedTable('relation "av_chunks" does not exist')))
    with pytest.raises(StoreError, match="PostgreSQL respondió con un error"):
        store.search([0.1] * 3)


def test_unreachable_database_has_an_actionable_message(monkeypatch):
    monkeypatch.setattr("app.pg.time.sleep", lambda _s: None)
    with pytest.raises(StoreError, match="docker compose ps"):
        PgConnection("postgresql://rag:rag@127.0.0.1:1/rag").run(lambda c: c)


def test_table_names_are_validated_against_sql_injection():
    with pytest.raises(ValueError):
        PgVectorStore("postgresql://x", "tabla; DROP TABLE x")


class Unreachable(MemoryStore):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def count(self):
        raise self.error


@pytest.mark.parametrize("error", [StoreError("Se perdió la conexión con PostgreSQL"), RuntimeError("algo inesperado")])
def test_an_unreachable_store_is_a_503_not_an_empty_store(client, error):
    """El bug original de rag-simple: no poder contar se interpretaba como 0 fragmentos y se mandaba a ingestar de nuevo (409)."""
    main.service().store = Unreachable(error)
    for path, body in (("/api/ask", {"question": "¿Qué es la cola de mensajes?", "mode": "vector"}), ("/api/evaluate", {})):
        response = client.post(path, json=body)
        assert response.status_code == 503, path
        assert "vacío" not in response.json()["detail"]
