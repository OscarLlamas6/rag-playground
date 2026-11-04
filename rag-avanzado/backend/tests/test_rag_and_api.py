"""El pipeline completo y la API, sin red: embeddings y modelo fake, todo en memoria."""

import pytest

from app import main
from app.fusion import rrf
from app.domain import Item
from app.rag import cited_numbers, decide_direction, needs_teams

IMPACT = "Si falla la base de datos de clientes, ¿qué servicios dejan de funcionar?"


def ask(client, question=IMPACT, **body):
    return client.post("/api/ask", json={"question": question, **body})


# ───────────── estado y errores claros ─────────────
def test_health_and_docs(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200


def test_status_before_and_after_setup(client, ready):
    s = client.get("/api/status").json()
    assert s["store"] == "memory" and s["chunks"] == 33 and s["contextual_chunks"] == 33
    assert s["graphs"]["curated"] == {"entities": 24, "triples": 40} and s["summaries"] == {"curated": 3, "extracted": 3}


def test_asking_before_ingesting_is_a_409_that_says_what_to_do(client):
    r = ask(client, mode="vector")
    assert r.status_code == 409 and "ingesta" in r.json()["detail"]


def test_graph_modes_need_the_graph_first(client):
    client.post("/api/ingest", json={})
    for mode in ("graph_local", "hybrid_graph"):
        r = ask(client, mode=mode)
        assert r.status_code == 409 and "Construir grafo" in r.json()["detail"], mode
    client.post("/api/graph/build", json={"source": "curated"})
    r = ask(client, mode="graph_global")
    assert r.status_code == 409 and "Resumir comunidades" in r.json()["detail"]
    assert ask(client, mode="graph_local", graph="extracted").status_code == 409


def test_contextual_search_needs_a_contextual_ingest(client):
    client.post("/api/ingest", json={"contextual": False})
    r = ask(client, mode="vector", contextual=True)
    assert r.status_code == 409 and "contextual" in r.json()["detail"]


@pytest.mark.parametrize("body", [{"mode": "graph_local", "decompose": True}, {"mode": "graph_global", "decompose": True}])
def test_decompose_only_applies_to_vector_and_hybrid(ready, body):
    r = ask(ready, **body)
    assert r.status_code == 422 and "descomposición" in r.json()["detail"]


@pytest.mark.parametrize("body", [{"mode": "magia"}, {"k": 0}, {"hops": 9}, {"question": "no"}, {"direction": "diagonal"}])
def test_invalid_requests_are_422(ready, body):
    assert ask(ready, **body).status_code == 422


def test_ingest_validation(client):
    assert client.post("/api/ingest", json={"strategy": "magia"}).status_code == 422
    assert client.post("/api/ingest", json={"max_words": 3}).status_code == 422


def test_unreachable_model_is_a_503_with_advice(ready, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "x")
    r = ask(ready, mode="vector")
    assert r.status_code == 503 and "LM Studio" in r.json()["detail"]


def test_unreachable_embeddings_are_a_503(ready, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_URL", "http://127.0.0.1:1/v1")
    r = ask(ready, mode="vector")
    assert r.status_code == 503 and "LM Studio" in r.json()["detail"]


# ───────────── las piezas ─────────────
def test_rrf_rewards_what_is_well_placed_in_several_lists():
    mk = lambda i: Item(i, "chunk", i)   # noqa: E731
    fused = rrf({"sub1": [mk("A"), mk("C"), mk("B")], "sub2": [mk("B"), mk("A"), mk("D")]}, k=4)
    assert [i.id for i, _ in fused] == ["A", "B", "C", "D"]
    assert fused[0][1] == {"sub1": 1, "sub2": 2} and round(fused[0][0].score, 4) == round(1 / 61 + 1 / 62, 4)


def test_citation_parsing():
    assert cited_numbers("A [1] y B [2, 3]; ver [10].") == {1, 2, 3, 10}
    assert cited_numbers("sin citas") == set()


@pytest.mark.parametrize("question, direction", [
    (IMPACT, "reverse"), ("Si se apaga el Centro de datos Sur, ¿qué servicios se afectan?", "reverse"),
    ("¿De qué servicios depende el chat de soporte?", "forward"), ("¿Qué equipo administra el Portal Mi Andes?", "forward"),
    ("Háblame de la red troncal", "both")])
def test_direction_rules_are_fixed_and_visible(question, direction):
    chosen, reason = decide_direction(question)
    assert chosen == direction and reason
    assert decide_direction(question, "forward") == ("forward", "elegido por ti")


def test_teams_are_added_only_when_the_question_asks_for_them():
    assert needs_teams("¿Qué equipos tendrían que intervenir?") and not needs_teams(IMPACT)


# ───────────── los cuatro modos ─────────────
def test_vector_mode_returns_k_fragments(ready):
    r = ask(ready, "¿Qué es la Cola de mensajes?", mode="vector", k=3).json()
    assert [s["key"] for s in r["stages"]] == ["vector"] and len(r["items"]) == 3
    assert {i["kind"] for i in r["items"]} == {"chunk"} and r["subgraph"] is None


def test_graph_local_follows_the_dependencies_backwards(ready):
    r = ask(ready, mode="graph_local").json()
    assert [s["key"] for s in r["stages"]] == ["link", "traverse", "facts", "origin"]
    facts = [i for i in r["items"] if i["kind"] == "fact"]
    assert {f["meta"]["subject"] for f in facts} == {"API de facturación", "App móvil", "Chat de soporte", "Pasarela de pagos", "Portal Mi Andes",
                                                      "Servicio de autenticación", "Sistema de tickets"}
    assert all(f["meta"]["evidence"] and f["meta"]["chunk_id"] for f in facts)            # procedencia: oración y fragmento
    assert r["direction"] == "reverse" and "WITH RECURSIVE" in r["sql"][0]["sql"]
    assert {n["id"]: n["role"] for n in r["subgraph"]["nodes"]}["Base de datos de clientes"] == "seed"
    assert r["items"][0]["kind"] == "fact" and r["items"][-1]["kind"] == "chunk"          # primero los hechos, luego su texto de origen


def test_graph_local_with_teams_and_forward_direction(ready):
    r = ask(ready, "Si falla el directorio de usuarios, ¿qué equipos tendrían que intervenir?", mode="graph_local").json()
    teams = {i["meta"]["object"] for i in r["items"] if i["kind"] == "fact" and i["meta"]["relation"] == "administrado_por"}
    assert teams == {"Equipo Seguridad", "Equipo Pagos", "Equipo Soporte TI", "Equipo Web"}
    r = ask(ready, "¿De qué servicios depende el chat de soporte?", mode="graph_local", hops=1).json()
    assert {i["meta"]["object"] for i in r["items"] if i["kind"] == "fact"} == {"Sistema de tickets", "Servicio de autenticación"}


def test_graph_local_without_entities_has_no_context_and_does_not_call_the_model(ready):
    r = ask(ready, "¿Cuál es el servicio del que dependen directamente más servicios?", mode="graph_local", link_threshold=1.0).json()
    assert r["items"] == [] and r["usage"] is None and r["abstained"] and r["answer"].startswith("No encuentro")


def test_graph_local_on_the_extracted_graph(ready):
    keys = lambda r: {i["id"] for i in r["items"] if i["kind"] == "fact"}   # noqa: E731
    extracted, curated = ask(ready, mode="graph_local", graph="extracted").json(), ask(ready, mode="graph_local").json()
    assert keys(extracted) == keys(curated) and len(keys(curated)) == 13


def test_hybrid_has_one_numbered_context_with_typed_items(ready):
    r = ask(ready, mode="hybrid_graph", k=3).json()
    kinds = [i["kind"] for i in r["items"]]
    assert kinds[0] == "fact" and kinds.count("chunk") == 3 and kinds == sorted(kinds, key=lambda k: k != "fact")
    assert "(hecho · salto 1" in r["prompt"] and "(fragmento ·" in r["prompt"] and "[1]" in r["prompt"]
    assert [i["number"] for i in r["items"]] == list(range(1, len(kinds) + 1))


def test_decompose_runs_each_subquestion_and_fuses_with_rrf(ready):
    q = "Si falla la pasarela de pagos, ¿qué servicio se ve afectado y qué equipo la administra?"
    v = ask(ready, q, mode="vector", decompose=True).json()
    assert [s["key"] for s in v["stages"]] == ["decompose", "vector", "rrf"]
    assert len(v["stages"][0]["items"]) == 2 and v["stages"][0]["llm"]["provider"] == "fake"
    h = ask(ready, q, mode="hybrid_graph", decompose=True).json()
    assert "decompose" in [s["key"] for s in h["stages"]] and "rrf" in [s["key"] for s in h["stages"]]
    assert any(i["kind"] == "fact" for i in h["items"])


def test_graph_global_answers_from_community_summaries(ready):
    r = ask(ready, "¿Qué grandes grupos de servicios hay y qué hace cada uno?", mode="graph_global").json()
    assert [s["key"] for s in r["stages"]] == ["map"] and len(r["stages"][0]["items"]) == 3
    assert r["items"] and {i["kind"] for i in r["items"]} == {"community"}


def test_contextual_chunks_carry_their_context_phrase(ready):
    r = ask(ready, "¿Dónde se ejecuta la Cola de mensajes?", mode="vector", contextual=True).json()
    assert all(i["meta"]["context"].startswith("Este fragmento pertenece a la sección") for i in r["items"])
    plain = ask(ready, "¿Dónde se ejecuta la Cola de mensajes?", mode="vector").json()
    assert [i["score"] for i in r["items"]] != [i["score"] for i in plain["items"]]


# ───────────── citas ─────────────
def test_citations_are_verified_by_code(ready, monkeypatch):
    ok = ask(ready, mode="graph_local").json()
    assert ok["invalid_citations"] == [] and not ok["uncited"] and "fact" in ok["cited_kinds"]
    assert any(i["cited"] for i in ok["items"])

    from app import llm
    real = llm.generate
    def fake(prompt, max_tokens=500, purpose="answer"):
        r = real(prompt, max_tokens, purpose)
        if purpose == "answer":
            r.text = "Dejan de funcionar el portal y la app [99]."
        return r
    monkeypatch.setattr("app.rag.generate", fake)
    bad = ask(ready, mode="graph_local").json()
    assert bad["invalid_citations"] == [99]
    def nocite(prompt, max_tokens=500, purpose="answer"):
        r = real(prompt, max_tokens, purpose)
        r.text = "Dejan de funcionar el portal y la app." if purpose == "answer" else r.text
        return r
    monkeypatch.setattr("app.rag.generate", nocite)
    assert ask(ready, mode="graph_local").json()["uncited"] is True


# ───────────── el grafo y la ingesta por la API ─────────────
def test_graph_endpoint_before_building_is_not_an_error(client):
    assert client.get("/api/graph").json() == {"source": "curated", "built": False, "nodes": [], "edges": [], "communities": [], "stats": None}


def test_graph_endpoint_returns_nodes_edges_stats_and_communities(ready):
    g = ready.get("/api/graph").json()
    assert len(g["nodes"]) == 24 and len(g["edges"]) == 40 and g["stats"]["edges"] == 40
    assert len(g["communities"]) == 3 and all(c["summary"] for c in g["communities"])
    assert {n["community"] for n in g["nodes"] if n["type"] != "equipo"} == {1, 2, 3}
    assert ready.get("/api/graph?source=otro").status_code == 422


def test_extracted_graph_report_includes_precision_recall_and_rejections(ready):
    e = ready.post("/api/graph/build", json={"source": "extracted"}).json()["extraction"]
    assert e["calls"] == 0 and e["cache_hits"] == 16 and e["sections"] == 16      # la segunda construcción sale de la caché
    assert e["metrics"]["f1"] == 1.0 and len(e["rejected"]) == 2
    forced = ready.post("/api/graph/build", json={"source": "extracted", "force": True}).json()["extraction"]
    assert forced["calls"] == 16 and forced["cache_hits"] == 0
    assert ready.post("/api/graph/build", json={"source": "extracted", "prompt_version": "v2"}).json()["extraction"]["calls"] == 16
    assert ready.post("/api/graph/build", json={"source": "extracted", "prompt_version": "v99"}).status_code == 422


def test_contextual_ingest_caches_the_context_phrases(client):
    first = client.post("/api/ingest", json={"contextual": True}).json()
    again = client.post("/api/ingest", json={"contextual": True}).json()
    assert (first["context_calls"], first["context_cache_hits"]) == (33, 0) and (again["context_calls"], again["context_cache_hits"]) == (0, 33)
    assert first["preview"][0]["context"]
    assert client.post("/api/ingest", json={"contextual": False}).json()["preview"][0]["context"] == ""


def test_documents_and_examples(client):
    docs = client.get("/api/documents").json()
    assert docs[0]["source"] == "servicios_ti.md" and len(docs[0]["sections"]) == 17
    ex = client.get("/api/examples").json()
    assert len(ex) == 12 and {e["origin"] for e in ex} == {"curso", "extra"}


def test_a_second_heavy_operation_at_the_same_time_is_a_409(client):
    svc = main.service()
    assert svc._lock.acquire(blocking=False)
    try:
        assert client.post("/api/ingest", json={}).status_code == 409
        assert client.post("/api/graph/build", json={}).status_code == 409
    finally:
        svc._lock.release()
