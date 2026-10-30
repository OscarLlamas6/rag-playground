import pytest

from app.rag import RagService


def ingested(strategy="parent_child", **kw):
    svc = RagService()
    svc.ingest(strategy=strategy, **kw)
    return svc


def test_hybrid_returns_parents_and_a_trace_with_every_stage():
    svc = ingested()
    r = svc.search("¿Qué indica la alarma OPT-1107?", mode="hybrid", k=3)
    assert [s.key for s in r.stages] == ["vector", "bm25", "rrf", "parent"]
    assert "OPT-1107" in r.hits[0].text
    assert len(r.hits[0].text.split()) > 35                       # se entregó el padre, no el hijo
    assert len({h.id for h in r.hits}) == len(r.hits)             # sin padres repetidos
    assert set(r.stages[2].hits[0]["ranks"]) <= {"vector", "bm25"}


def test_each_mode_runs_only_its_own_stages():
    svc = ingested("sentences", max_words=60)
    assert [s.key for s in svc.search("plan de 300 Mbps", mode="vector").stages] == ["vector"]
    assert [s.key for s in svc.search("plan de 300 Mbps", mode="bm25").stages] == ["bm25"]


def test_audience_filter_keeps_only_that_audience_plus_general():
    svc = ingested()
    q = "¿En cuánto tiempo deben reparar una falla?"
    for mode in ("vector", "bm25", "hybrid"):
        hits = svc.search(q, mode=mode, k=8, audience="negocio", expand_parent=False).hits
        assert hits and {h.metadata["audience"] for h in hits} <= {"negocio", "general"}, mode
    top = svc.search(q, mode="hybrid", audience="negocio").hits[0]
    assert "4 horas" in top.text
    assert "48 horas" in svc.search(q, mode="hybrid", audience="residencial").hits[0].text


def test_rerank_adds_a_stage_with_llm_usage():
    svc = ingested()
    r = svc.search("¿Cuánto cuesta el plan de 300 Mbps?", rerank=True, candidates=6)
    assert r.stages[-1].key == "rerank" and r.stages[-1].llm["provider"] == "fake"
    assert r.rerank_llm is not None


def test_ask_cites_and_sums_cost():
    svc = ingested()
    a = svc.ask("¿Cuánto cuesta el plan de 300 Mbps?", rerank=True)
    assert "39" in a.text and a.cited and not a.invalid and not a.abstained
    assert set(a.timings_ms) >= {"vector", "bm25", "rrf", "rerank", "generate", "total"}


def test_no_lexical_match_means_no_llm_call():
    svc = ingested()
    a = svc.ask("xyzzy qwerty plugh", mode="bm25")
    assert a.abstained and a.llm is None and a.hits == []


def test_lexical_index_is_rebuilt_from_the_cache_after_a_restart():
    ingested()
    fresh = RagService()                                          # "servidor reiniciado": sin BM25 en memoria
    assert fresh.search("alarma OPT-1107", mode="bm25").hits


def test_api_flow(client):
    assert client.post("/api/ask", json={"question": "¿Cuánto cuesta el plan de 300 Mbps?"}).status_code == 409
    ingest = client.post("/api/ingest", json={}).json()
    assert ingest["documents"] == 3 and ingest["chunks"] > 30 and set(
        d["type"] for d in ingest["per_document"].values()) == {"md", "html", "docx"}
    body = client.post("/api/ask", json={"question": "¿Cuánto cuesta el plan de 300 Mbps?", "mode": "hybrid", "k": 3}).json()
    assert "39" in body["answer"] and [s["key"] for s in body["stages"]][:3] == ["vector", "bm25", "rrf"]
    assert body["hits"][0]["number"] == 1 and body["usage"]["provider"] == "fake"
    assert client.post("/api/ask", json={"question": "hola, ¿qué tal?", "mode": "magic"}).status_code == 422
    assert client.post("/api/ask", json={"question": "hola, ¿qué tal?", "audience": "otro"}).status_code == 422
    assert client.post("/api/ask", json={"question": "hola, ¿qué tal?", "prompt_version": "v9"}).status_code == 422
    assert client.post("/api/ingest", json={"strategy": "nope"}).status_code == 422


def test_documents_and_examples(client):
    docs = client.get("/api/documents").json()
    assert sorted(d["type"] for d in docs) == ["docx", "html", "md"]
    ex = client.get("/api/examples").json()
    assert len(ex) == 26 and {e["type"] for e in ex} == {"semantica", "lexica", "tabla", "filtro"}


def test_evaluate_compares_modes_by_question_type(client):
    client.post("/api/ingest", json={})
    s = client.post("/api/evaluate", json={"k": 3}).json()["summary"]
    assert s["n"] == 26 and set(s["modes"]) == {"vector", "bm25", "hybrid"}
    bm25 = s["modes"]["bm25"]["by_type"]["lexica"]
    assert bm25["n"] == 11 and bm25["hit1"] >= 9                    # BM25 resuelve los códigos exactos
    assert s["modes"]["hybrid"]["hitk"] >= s["modes"]["vector"]["hitk"] - 2
    with_rerank = client.post("/api/evaluate", json={"k": 3, "with_rerank": True}).json()["summary"]
    assert "hybrid+rerank" in with_rerank["modes"]


def test_two_ingestions_cannot_overlap(client):
    from app import main
    from app.rag import IngestBusy, RagService

    service = main.service()
    assert service._ingest_lock.acquire(blocking=False)         # "hay otra ingesta en curso"
    try:
        response = client.post("/api/ingest", json={})
        assert response.status_code == 409 and "en curso" in response.json()["detail"]
        with pytest.raises(IngestBusy):
            RagService.ingest(service)
    finally:
        service._ingest_lock.release()
    assert client.post("/api/ingest", json={}).status_code == 200      # liberado: vuelve a funcionar
