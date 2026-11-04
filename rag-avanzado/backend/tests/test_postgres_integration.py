"""Integración con el PostgreSQL REAL del docker-compose (imagen pgvector). Se omite si no hay base de datos.

    task test:integration      levanta SOLO el servicio `db`, corre estas pruebas y lo elimina (con su volumen)

Son destructivas con las tablas av_* de esa base: no las apuntes a una base con datos que quieras conservar.
"""

import os

import pytest

from app import graph as G
from app.config import get_settings
from app.domain import Chunk
from app.graphstore import MemoryGraphStore, PgGraphStore
from app.ingest import load_documents
from app.store import PgVectorStore

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="define TEST_DATABASE_URL (task test:integration levanta el servicio db)")


@pytest.fixture
def curated():
    s = get_settings()
    return G.build_curated(s.graph_file, load_documents(s.docs_dir))


@pytest.fixture
def pg(curated):
    store = PgGraphStore(URL)
    store.drop()
    store.save(curated, {n: [1.0 if i == j else 0.0 for j in range(4)] for i, n in enumerate(curated.entities)})
    yield store
    store.drop()


def test_the_graph_roundtrips_through_postgres(pg, curated):
    loaded = pg.load("curated")
    assert loaded.entities == curated.entities and sorted(t.key for t in loaded.triples) == sorted(t.key for t in curated.triples)
    assert all(t.evidence for t in loaded.triples) and pg.sources() == {"curated": {"entities": 24, "triples": 40}}
    assert pg.load("extracted") is None


def test_recursive_cte_gives_the_same_edges_as_the_in_memory_walk(pg, curated):
    mem = MemoryGraphStore()
    mem.save(curated)
    key = lambda t: [(e["subject"], e["relation"], e["object"], e["depth"], e["seed"]) for e in t.edges]   # noqa: E731
    checked = 0
    for seeds in (["Base de datos de clientes"], ["Chat de soporte"], ["Servicio de autenticación", "Cola de mensajes"], ["Centro de datos Sur"]):
        for direction in ("reverse", "forward", "both"):
            for hops in (1, 2, 4):
                a, b = pg.traverse("curated", seeds, direction, hops, G.IMPACT_RELATIONS), mem.traverse("curated", seeds, direction, hops, G.IMPACT_RELATIONS)
                assert key(a) == key(b), (seeds, direction, hops)
                checked += 1
    assert checked == 36
    assert pg.traverse("curated", ["Base de datos de clientes"], "reverse", 4, G.IMPACT_RELATIONS).queries[0]["equivalent"] is False
    assert pg.team_facts("curated", ["Portal Mi Andes"])[0]["object"] == "Equipo Web"


def test_the_recursive_cte_survives_a_cycle(pg):
    from app.domain import GraphData, Triple
    pg.save(GraphData({n: {"tipo": "servicio", "alias": []} for n in "ABC"},
                      [Triple("A", "depende_de", "B"), Triple("B", "depende_de", "C"), Triple("C", "depende_de", "A")], "cyclic"))
    t = pg.traverse("cyclic", ["A"], "forward", 8, G.IMPACT_RELATIONS)
    assert [(e["subject"], e["object"]) for e in t.edges] == [("A", "B"), ("B", "C")]


def test_entity_linking_uses_pgvector_cosine_distance(pg, curated):
    first = list(curated.entities)[0]
    assert pg.search_entities("curated", [0.9, 0.1, 0.0, 0.0], k=1)[0][0] == first


def test_communities_and_summaries_are_stored(pg, curated):
    comms = G.detect_communities(curated)
    comms[0].summary = "Resumen de prueba"
    pg.save_communities("curated", comms)
    loaded = pg.load_communities("curated")
    assert [c.members for c in loaded] == [c.members for c in comms] and loaded[0].summary == "Resumen de prueba"


def test_chunks_with_and_without_context_vectors():
    store = PgVectorStore(URL)
    store.drop()
    store.reset(3)
    store.upsert([Chunk("a-0", "Se ejecuta en el Centro de datos Norte.", {"section": "Red troncal", "source": "s.md", "n": 0}, "Habla de la Red troncal.",
                        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                  Chunk("a-1", "Lo administra el Equipo Redes.", {"section": "Red troncal", "source": "s.md", "n": 1}, "", [0.0, 0.0, 1.0], None)])
    assert (store.count(), store.count_contextual()) == (2, 1)
    assert store.search([1.0, 0.0, 0.0], 1)[0].id == "a-0" and store.search([0.0, 1.0, 0.0], 5, contextual=True)[0].meta["context"] == "Habla de la Red troncal."
    assert [c.id for c in store.by_section("Red troncal")] == ["a-0", "a-1"]
    store.drop()
    assert store.count() == 0


def test_the_connection_recovers_when_the_server_kills_it():
    store = PgVectorStore(URL)
    store.reset(3)
    assert store.count() == 0
    killer = PgVectorStore(URL)
    killer.db.run(lambda c: c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = current_database() "
                                      "AND application_name = ''").fetchall())
    assert store.count() == 0           # reconectó solo
    store.drop()


def test_the_whole_application_runs_on_postgres(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setenv("AVANZADO_STORE", "postgres")
    monkeypatch.setenv("DATABASE_URL", URL)
    main._service = None
    client = TestClient(main.app)
    svc = main.service()
    svc.store.drop()
    svc.graphs.drop()
    assert client.post("/api/ingest", json={"contextual": True}).json()["store"] == "postgres"
    for source in ("curated", "extracted"):
        assert client.post("/api/graph/build", json={"source": source}).status_code == 200
    assert client.post("/api/graph/summarize", json={"source": "curated"}).json()["communities"] == 3
    r = client.post("/api/ask", json={"question": "Si falla la base de datos de clientes, ¿qué servicios dejan de funcionar?", "mode": "hybrid_graph"}).json()
    assert r["sql"][0]["equivalent"] is False and "WITH RECURSIVE" in r["sql"][0]["sql"]
    assert len([i for i in r["items"] if i["kind"] == "fact"]) == 13 and r["invalid_citations"] == []
    assert client.post("/api/ask", json={"question": "¿Qué grupos de servicios hay?", "mode": "graph_global"}).status_code == 200
    ev = client.post("/api/evaluate", json={}).json()["summary"]
    assert ev["configs"]["graph_local"]["context_recall"] > ev["configs"]["vector_k4"]["context_recall"]
    svc.store.drop()
    svc.graphs.drop()
