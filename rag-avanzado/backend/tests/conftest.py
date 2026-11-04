"""Las pruebas no necesitan servidores ni cuentas: embeddings y modelo "fake", almacén y grafo en memoria, caché en una carpeta temporal."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "fake")
    monkeypatch.setenv("AVANZADO_STORE", "memory")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    main._service = None          # un servicio (y un almacén) nuevo en cada prueba


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def ready(client):
    """Ingesta contextual + grafo curado y extraído + resúmenes: todo lo que pueden pedir las pruebas."""
    assert client.post("/api/ingest", json={"contextual": True}).status_code == 200
    for source in ("curated", "extracted"):
        assert client.post("/api/graph/build", json={"source": source}).status_code == 200
        assert client.post("/api/graph/summarize", json={"source": source}).status_code == 200
    return client
