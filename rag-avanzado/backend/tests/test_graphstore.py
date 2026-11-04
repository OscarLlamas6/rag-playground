"""Los recorridos del grafo (la versión en memoria; la de PostgreSQL se compara con ella en test_postgres_integration.py)."""

import networkx as nx
import pytest

from app import graph as G
from app.config import get_settings
from app.domain import GraphData, Triple
from app.graphstore import MemoryGraphStore, display_sql
from app.ingest import load_documents
from app.pg import StoreError

DEPS = G.IMPACT_RELATIONS


@pytest.fixture
def store():
    s = get_settings()
    st = MemoryGraphStore()
    st.save(G.build_curated(s.graph_file, load_documents(s.docs_dir)), {})
    return st


def names(t, key="subject"):
    return sorted({e[key] for e in t.edges})


def test_reverse_walk_answers_the_impact_question(store):
    """«Si cae la base de datos de clientes, ¿qué servicios dejan de funcionar?»: las dependencias se recorren AL REVÉS."""
    t = store.traverse("curated", ["Base de datos de clientes"], "reverse", 4, DEPS)
    assert names(t) == ["API de facturación", "App móvil", "Chat de soporte", "Pasarela de pagos", "Portal Mi Andes",
                        "Servicio de autenticación", "Sistema de tickets"]
    depth = {(e["subject"], e["object"]): e["depth"] for e in t.edges}
    assert depth[("Servicio de autenticación", "Base de datos de clientes")] == 1
    assert depth[("Portal Mi Andes", "Servicio de autenticación")] == 2
    assert depth[("Chat de soporte", "Sistema de tickets")] == 2


def test_forward_walk_answers_the_dependency_question(store):
    t = store.traverse("curated", ["Chat de soporte"], "forward", 4, DEPS)
    assert sorted({e["object"] for e in t.edges}) == sorted(nx.descendants(G.to_digraph(store.load("curated"), DEPS), "Chat de soporte"))
    assert {e["direction"] for e in t.edges} == {"forward"}


def test_the_wrong_direction_gives_the_wrong_answer(store):
    wrong = store.traverse("curated", ["Base de datos de clientes"], "forward", 4, DEPS)
    assert names(wrong, "object") == ["Centro de datos Norte", "Red troncal"]       # de qué depende, no a quién afecta


def test_hops_limit_the_walk(store):
    one = store.traverse("curated", ["Base de datos de clientes"], "reverse", 1, DEPS)
    assert names(one) == ["API de facturación", "Servicio de autenticación", "Sistema de tickets"]
    assert max(e["depth"] for e in one.edges) == 1


def test_both_directions_merge_without_duplicates(store):
    t = store.traverse("curated", ["Servicio de autenticación"], "both", 2, DEPS)
    keys = [(e["subject"], e["relation"], e["object"]) for e in t.edges]
    assert len(keys) == len(set(keys)) and {e["direction"] for e in t.edges} == {"reverse", "forward"}
    assert len(t.queries) == 2


def test_several_seeds_and_their_origin(store):
    t = store.traverse("curated", ["Cola de mensajes", "Directorio de usuarios"], "reverse", 1, DEPS)
    assert {(e["subject"], e["seed"]) for e in t.edges} == {("Servicio de notificaciones", "Cola de mensajes"), ("Servicio de autenticación", "Directorio de usuarios")}


def test_a_cycle_does_not_loop_forever():
    data = GraphData({n: {"tipo": "servicio"} for n in "ABC"}, [Triple("A", "depende_de", "B"), Triple("B", "depende_de", "C"), Triple("C", "depende_de", "A")])
    st = MemoryGraphStore()
    st.save(data)
    t = st.traverse("curated", ["A"], "forward", 8, DEPS)
    # la arista que cierra el ciclo (C → A) no se sigue: el camino ya pasó por A. Igual que en la CTE (NOT ... = ANY(path)).
    assert [(e["subject"], e["object"]) for e in t.edges] == [("A", "B"), ("B", "C")]


def test_team_lookup(store):
    teams = store.team_facts("curated", ["Directorio de usuarios", "Portal Mi Andes"])
    assert {(t["subject"], t["object"]) for t in teams} == {("Directorio de usuarios", "Equipo Seguridad"), ("Portal Mi Andes", "Equipo Web")}


def test_the_sql_is_a_recursive_cte_with_displayable_parameters(store):
    t = store.traverse("curated", ["Red troncal"], "reverse", 3, DEPS)
    q = t.queries[0]
    assert "WITH RECURSIVE" in q["sql"] and ":seeds" in q["sql"] and "%(" not in q["sql"]
    assert q["params"]["hops"] == 3 and q["equivalent"] is True
    assert display_sql("a = %(x)s") == "a = :x"


def test_entity_search_by_embedding(store):
    st = MemoryGraphStore()
    st.save(store.load("curated"), {"Red troncal": [1.0, 0.0], "Cola de mensajes": [0.0, 1.0]})
    assert st.search_entities("curated", [0.9, 0.1], k=1)[0][0] == "Red troncal"


def test_errors_are_clear(store):
    with pytest.raises(ValueError, match="direction"):
        store.traverse("curated", ["Red troncal"], "diagonal", 2, DEPS)
    with pytest.raises(ValueError, match="hops"):
        store.traverse("curated", ["Red troncal"], "reverse", 0, DEPS)
    with pytest.raises(StoreError, match="no existe"):
        store.traverse("extracted", ["Red troncal"], "reverse", 2, DEPS)
