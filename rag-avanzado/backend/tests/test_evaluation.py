"""La evaluación: cobertura por código, configuraciones, tipos de pregunta y configuraciones que no se pueden correr."""

import pytest

from app import evaluation as ev
from app import graph as G
from app.config import get_settings

ENT = G.load_inventory(get_settings().graph_file)[0]


def test_coverage_ignores_entities_named_in_the_question():
    q = "Si falla la base de datos de clientes, ¿qué servicios dejan de funcionar?"
    c = ev.coverage(ENT, "Fallan la Base de datos de clientes, el Portal Mi Andes y la App móvil.", ["Portal Mi Andes", "App móvil", "Chat de soporte"], q)
    assert c["recall"] == pytest.approx(2 / 3) and c["precision"] == 1.0 and c["missing"] == ["Chat de soporte"] and c["extra"] == []
    c = ev.coverage(ENT, "El Portal Mi Andes y la Cola de mensajes.", ["Portal Mi Andes"], q)
    assert c["recall"] == 1.0 and c["precision"] == 0.5 and c["extra"] == ["Cola de mensajes"]


def test_questions_have_eight_from_the_course_and_extras_for_contrast():
    qs = ev.load_questions()
    assert [q["id"] for q in qs if q["origen"] == "curso"] == [f"MH{i}" for i in range(1, 9)]
    assert {q["tipo"] for q in qs} <= set(ev.TYPES) and "directa" in {q["tipo"] for q in qs}


def test_free_evaluation_skips_what_needs_the_llm_and_what_is_missing(client):
    client.post("/api/ingest", json={})
    client.post("/api/graph/build", json={"source": "curated"})
    r = client.post("/api/evaluate", json={}).json()
    s = r["summary"]
    assert set(s["configs"]) == {"vector_k4", "vector_k8", "graph_local", "graph_local_h2", "hybrid"}
    assert "contextual" in s["skipped"]["vector_ctx"] and "extracted" in s["skipped"]["graph_local_ext"]
    assert "answer_recall" not in s["configs"]["vector_k4"] and s["cost_usd"] == 0 and s["n"] == 12
    assert "ruido" in s["warning"] and len(r["rows"]) == 12
    assert s["configs"]["graph_local"]["context_recall"] > s["configs"]["vector_k4"]["context_recall"]


def test_full_evaluation_with_answers(ready):
    r = ready.post("/api/evaluate", json={"generate": True}).json()
    s = r["summary"]
    assert s["skipped"] == {} and len(s["configs"]) == len(ev.CONFIGS)
    graph = s["configs"]["graph_local"]
    assert {"answer_recall", "answer_precision", "complete", "valid_citations", "avg_ms", "cost_usd", "by_type"} <= set(graph)
    assert graph["by_type"]["impacto"]["context_recall"] == 1.0 and graph["by_type"]["impacto"]["n"] == 5
    row = r["rows"][0]["results"]["graph_local"]
    assert row["answer"] and "answer_missing" in row


def test_choose_configs_and_questions(ready):
    r = ready.post("/api/evaluate", json={"configs": ["vector_dec", "graph_global"], "questions": ["MH1", "MH2"]}).json()
    assert set(r["summary"]["configs"]) == {"vector_dec", "graph_global"} and r["summary"]["n"] == 2
    assert ready.post("/api/evaluate", json={"configs": ["nada"]}).status_code == 422


def test_evaluating_with_nothing_set_up_is_a_409(client):
    assert client.post("/api/evaluate", json={}).status_code == 409


def test_config_list_endpoint(client):
    names = [c["name"] for c in client.get("/api/evaluate/configs").json()]
    assert names == [c.name for c in ev.CONFIGS]
