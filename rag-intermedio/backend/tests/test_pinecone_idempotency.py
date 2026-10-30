"""Crear el índice de Pinecone debe ser idempotente, también cuando dos ingestas se pisan (la carrera has_index → create_index)."""

from types import SimpleNamespace

import pytest
from pinecone.errors.exceptions import ConflictError, PineconeApiException

from app.config import get_settings
from app.store import PineconeStore, StoreError


class FakeIndex:
    def __init__(self, dimension):
        self.dimension, self.deleted = dimension, []

    def describe_index_stats(self):
        return SimpleNamespace(dimension=self.dimension, namespaces={})

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


class FakePinecone:
    def __init__(self, exists=False, dimension=768, create_error=None, visible_after_conflict=True):
        self.exists, self.dimension, self.create_error = exists, dimension, create_error
        self.created = 0
        self.index = FakeIndex(dimension)

    def has_index(self, name):
        return self.exists

    def create_index(self, **kwargs):
        self.created += 1
        self.dimension = kwargs["dimension"]
        self.index.dimension = kwargs["dimension"]
        if self.create_error:
            self.exists = True                  # otro proceso ya lo había creado
            raise self.create_error
        self.exists = True

    def describe_index(self, name):
        return SimpleNamespace(status=SimpleNamespace(ready=True), host="host-falso", dimension=self.dimension)

    def Index(self, host):                      # noqa: N802 - así se llama en el SDK
        return self.index


@pytest.fixture
def make_store(monkeypatch):
    monkeypatch.setenv("STORE", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-prueba")

    def build(fake):
        store = PineconeStore(get_settings())
        store._client = lambda: fake
        return store
    return build


def test_creates_the_index_when_missing(make_store):
    fake = FakePinecone(exists=False)
    make_store(fake).reset(1536)
    assert fake.created == 1 and fake.dimension == 1536


def test_does_not_create_when_it_already_exists(make_store):
    fake = FakePinecone(exists=True, dimension=768)
    make_store(fake).reset(768)
    assert fake.created == 0
    assert fake.index.deleted == [{"delete_all": True, "namespace": "rag-intermedio"}]       # solo se vacía el namespace


def test_reset_twice_is_a_no_op_the_second_time(make_store):
    fake = FakePinecone(exists=False)
    store = make_store(fake)
    store.reset(1536)
    store.reset(1536)
    assert fake.created == 1


def test_a_conflict_on_create_is_treated_as_already_exists(make_store):
    """Otra ingesta creó el índice entre has_index y create_index: el 409 no debe romper nada."""
    fake = FakePinecone(exists=False, dimension=1536, create_error=ConflictError("ya existe"))
    make_store(fake).reset(1536)                                  # no lanza
    assert fake.created == 1


def test_a_409_by_status_code_is_also_a_conflict(make_store):
    fake = FakePinecone(exists=False, dimension=1536, create_error=PineconeApiException("conflicto", 409))
    make_store(fake).reset(1536)


def test_other_errors_on_create_are_not_swallowed(make_store):
    fake = FakePinecone(exists=False, create_error=PineconeApiException("cuota de índices agotada", 403))
    with pytest.raises(StoreError, match="No se pudo preparar Pinecone"):
        make_store(fake).reset(1536)


def test_an_existing_index_with_another_dimension_gives_an_actionable_error(make_store):
    fake = FakePinecone(exists=True, dimension=768)
    with pytest.raises(StoreError, match="768 dimensiones.*1536"):
        make_store(fake).reset(1536)
