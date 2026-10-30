"""Las pruebas no necesitan servidores ni cuentas: embeddings y modelo "fake", almacén en memoria."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "fake")
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("CHUNKS_CACHE", str(tmp_path / "chunks.json"))   # que las pruebas no ensucien el caché real
    main._service = None


@pytest.fixture
def client():
    return TestClient(main.app)
