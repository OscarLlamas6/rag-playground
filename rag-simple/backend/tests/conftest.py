"""Las pruebas no necesitan servidores ni cuentas: embeddings y modelo "fake", almacén en memoria."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "fake")
    monkeypatch.setenv("STORE", "memory")
    main._service = None          # un servicio (y un almacén) nuevo en cada prueba


@pytest.fixture
def client():
    return TestClient(main.app)
