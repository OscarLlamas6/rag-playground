import pytest

from app.rag import RagService


def test_ingest_then_ask_cites_the_right_source():
    svc = RagService()
    report = svc.ingest(strategy="sentences", max_words=60, overlap=1)
    assert report.documents == 3 and report.chunks > 10 and report.store == "memory"
    answer = svc.ask("¿Cuánto cuesta el plan de 300 Mbps?", k=3)
    assert "39" in answer.text
    assert answer.cited and not answer.invalid and not answer.abstained
    assert answer.hits[0].metadata["section"] == "Planes de internet"
    assert answer.prompt and "<contexto>" in answer.prompt
    assert set(answer.timings_ms) >= {"embed_query", "search", "generate", "total"}


def test_out_of_scope_question_abstains():
    svc = RagService()
    svc.ingest()
    assert svc.ask("¿Cuál es la capital de Australia?").abstained


def test_threshold_skips_the_llm():
    svc = RagService()
    svc.ingest()
    answer = svc.ask("¿Cuál es la capital de Australia?", threshold=0.99)
    assert answer.llm is None and answer.skipped_reason and answer.abstained


def test_api_flow(client):
    assert client.get("/api/status").json()["chunks"] == 0
    assert client.post("/api/ask", json={"question": "¿Cuánto cuesta el plan de 300 Mbps?"}).status_code == 409

    ingest = client.post("/api/ingest", json={"strategy": "sentences", "max_words": 60, "overlap": 1}).json()
    assert ingest["chunks"] > 10
    assert client.get("/api/status").json()["chunks"] == ingest["chunks"]

    body = client.post("/api/ask", json={"question": "¿Cuánto cuesta el plan de 300 Mbps?", "k": 3}).json()
    assert "39" in body["answer"] and body["hits"][0]["number"] == 1
    assert any(h["cited"] for h in body["hits"]) and body["usage"]["provider"] == "fake"

    assert client.post("/api/ingest", json={"strategy": "nope"}).status_code == 422
    assert client.post("/api/ask", json={"question": "hola, ¿qué tal?", "prompt_version": "v99"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422       # demasiado corta


def test_documents_and_examples(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 3 and docs[0]["sections"][0]["text"]
    examples = client.get("/api/examples").json()
    assert any(e["out_of_scope"] for e in examples) and len(examples) == 14


def test_evaluate_retrieval_and_generation(client):
    client.post("/api/ingest", json={})
    retrieval = client.post("/api/evaluate", json={"k": 4}).json()
    assert retrieval["summary"]["in_scope"] == 12 and retrieval["summary"]["out_of_scope"] == 2
    assert retrieval["summary"]["retrieval"]["hitk"] >= 8          # con vectores "fake" ya funciona la mayoría
    assert "generation" not in retrieval["summary"]

    full = client.post("/api/evaluate", json={"k": 4, "with_generation": True}).json()
    assert full["summary"]["generation"]["abstain_ok"] >= 1
    assert all("answer" in r for r in full["rows"])


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
