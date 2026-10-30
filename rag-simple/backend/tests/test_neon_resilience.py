"""Neon suspende el cómputo (o cierra la sesión) tras un rato sin uso: la conexión larga muere. Eso no debe verse como
"almacén vacío": se reconecta una vez y, si no se puede, se explica."""

import psycopg
import pytest

from app import main
from app.store import MemoryStore, NeonStore, StoreError


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
    store = NeonStore("postgresql://usuario:clave@host/base", "rag_simple_chunks")
    pending = iter(connections)
    store._connection = lambda: next(pending)      # cada intento obtiene "una conexión nueva", como tras descartar la rota
    return store


def test_a_dead_connection_is_replaced_and_the_operation_retried():
    dead = FakeConnection(psycopg.OperationalError("server closed the connection unexpectedly"))
    alive = FakeConnection([("rag_simple_chunks",)], [(28,)])
    assert store_with(dead, alive).count() == 28


def test_if_reconnecting_also_fails_the_error_says_so():
    store = store_with(FakeConnection(psycopg.OperationalError("caída")), FakeConnection(psycopg.OperationalError("caída")))
    with pytest.raises(StoreError, match="Se perdió la conexión con Neon"):
        store.count()


def test_a_vector_dimension_mismatch_tells_you_to_reingest():
    store = store_with(FakeConnection(psycopg.errors.DataException("different vector dimensions 1536 and 768")))
    with pytest.raises(StoreError, match="pestaña Ingesta"):
        store.search([0.1] * 768)


def test_other_database_errors_become_a_readable_store_error():
    store = store_with(FakeConnection(psycopg.errors.UndefinedTable('relation "rag_simple_chunks" does not exist')))
    with pytest.raises(StoreError, match="Neon respondió con un error"):
        store.search([0.1] * 3)


class Unreachable(MemoryStore):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def count(self):
        raise self.error


@pytest.mark.parametrize("error", [StoreError("Se perdió la conexión con Neon"), RuntimeError("algo inesperado")])
def test_an_unreachable_store_is_a_503_not_an_empty_store(client, error):
    """El bug original: no poder contar se interpretaba como 0 fragmentos y se mandaba a ingestar de nuevo (409)."""
    main.service().store = Unreachable(error)
    for path, body in (("/api/ask", {"question": "¿Cuánto cuesta el plan de 300 Mbps?"}), ("/api/evaluate", {})):
        response = client.post(path, json=body)
        assert response.status_code == 503, path
        assert "vacío" not in response.json()["detail"]
