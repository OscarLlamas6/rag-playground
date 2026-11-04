"""El grafo: nombres, procedencia, extracción, normalización, validación, medición y comunidades."""

import networkx as nx
import pytest

from app import evaluation, graph as G
from app.config import get_settings
from app.domain import GraphData, Triple
from app.graph import ExtractionRun, RawTriple
from app.ingest import load_documents


@pytest.fixture
def curated():
    s = get_settings()
    return G.build_curated(s.graph_file, load_documents(s.docs_dir))


def raw(section, s, r, o):
    return (section, RawTriple(sujeto=s, relacion=r, objeto=o))


def test_the_curated_graph_has_the_course_numbers(curated):
    assert len(curated.entities) == 24 and len(curated.triples) == 40


def test_every_curated_fact_carries_its_provenance(curated):
    for t in curated.triples:
        assert t.section == t.subject and t.evidence, t
        assert G.mentions(curated.entities, t.evidence, t.object), t   # la oración citada nombra el objeto


@pytest.mark.parametrize("written, official", [
    ("la Red troncal", "Red troncal"), ("el Directorio de usuarios", "Directorio de usuarios"), ("LDAP", "Directorio de usuarios"),
    ("SERVICIO DE AUTENTICACION", "Servicio de autenticación"), ("«Chat de soporte»", "Chat de soporte"), ("el SSO", "Servicio de autenticación"),
    ("Servicio de correo corporativo", None)])
def test_canonical_names(curated, written, official):
    assert G.canonical_name(curated.entities, written) == official


def test_linking_by_name_prefers_the_longest_name(curated):
    found = dict(G.link_by_name(curated.entities, "Si cae la base de datos de clientes y el SSO, ¿qué pasa?"))
    assert set(found) == {"Base de datos de clientes", "Servicio de autenticación"}      # «base de datos» no se cuenta aparte
    assert G.link_by_name(curated.entities, "¿Qué hora es?") == []


def run_of(*items):
    return ExtractionRun(raw=list(items))


def normalized(curated, *items):
    sections = G.sections_of(load_documents(get_settings().docs_dir))
    return G.normalize_and_validate(run_of(*items), curated.entities, sections)


def test_normalization_fixes_names_relations_and_inverse_relations(curated):
    n = normalized(curated,
                   raw("Base de datos de clientes", "Base de datos de clientes", "depende de", "la Red troncal"),
                   raw("Base de datos de clientes", "Base de datos de clientes", "se ejecuta en", "el Centro de datos Norte"),
                   raw("Base de datos de clientes", "Equipo Datos", "administra", "Base de datos de clientes"))
    assert {t.key for t in n.graph.triples} == {
        ("Base de datos de clientes", "depende_de", "Red troncal"), ("Base de datos de clientes", "corre_en", "Centro de datos Norte"),
        ("Base de datos de clientes", "administrado_por", "Equipo Datos")}
    assert not n.rejected


@pytest.mark.parametrize("triple, reason", [
    (("Portal Mi Andes", "usa", "Red troncal"), "relación no permitida"),
    (("Portal Mi Andes", "depende_de", "Facebook"), "entidad desconocida"),
    (("Portal Mi Andes", "depende_de", "Portal Mi Andes"), "sí misma"),
    (("Portal Mi Andes", "administrado_por", "App móvil"), "debe ser un equipo"),
    (("Portal Mi Andes", "depende_de", "Equipo Web"), "equipo"),
    (("Red troncal", "corre_en", "Portal Mi Andes"), "infraestructura"),
    (("Sistema de tickets", "depende_de", "Chat de soporte"), "sujeto no es la sección"),        # la arista falsa que crea un ciclo
    (("Portal Mi Andes", "depende_de", "Cola de mensajes"), "sin respaldo"),                      # cierto en el mundo, pero no lo dice la sección
])
def test_validation_rejects_what_a_model_gets_wrong(curated, triple, reason):
    section = "Chat de soporte" if triple[0] == "Sistema de tickets" else "Portal Mi Andes" if triple[0] != "Red troncal" else "Red troncal"
    n = normalized(curated, raw(section, *triple))
    assert not n.graph.triples
    assert reason in n.rejected[0].reason


def test_duplicates_are_merged_and_counted(curated):
    n = normalized(curated, raw("Red troncal", "Red troncal", "corre_en", "Centro de datos Norte"),
                   raw("Red troncal", "la Red troncal", "se ejecuta en", "Centro de datos Norte"))
    assert len(n.graph.triples) == 1 and n.duplicates == 1


def test_extracted_triples_are_measured_against_the_curated_ones(curated):
    truth = {t.key for t in curated.triples}
    perfect = G.compare_triples(truth, set(truth))
    assert perfect["precision"] == perfect["recall"] == perfect["f1"] == 1.0
    partial = G.compare_triples(truth, set(sorted(truth)[:20]) | {("A", "depende_de", "B")})
    assert partial["precision"] == round(20 / 21, 3) and partial["recall"] == 0.5
    assert partial["extra"] == [("A", "depende_de", "B")] and len(partial["missing"]) == 20
    assert G.compare_triples(truth, set())["precision"] == 0.0


def test_validation_improves_precision_over_normalization_alone(client):
    client.post("/api/ingest", json={})
    report = client.post("/api/graph/build", json={"source": "extracted"}).json()["extraction"]
    assert report["metrics"]["precision"] == 1.0 and report["metrics"]["recall"] == 1.0           # el extractor fake es bueno…
    assert report["metrics_without_validation"]["precision"] < 1.0                                # …pero sin validar deja una arista falsa
    reasons = sorted(r["reason"] for r in report["rejected"])
    assert reasons == ["el sujeto no es la sección «Chat de soporte» (posible arista falsa)", "entidad desconocida: «Servicio de correo corporativo»"]


def test_communities_are_a_partition_of_the_services(curated):
    comms = G.detect_communities(curated)
    members = [m for c in comms for m in c.members]
    services = {n for n, a in curated.entities.items() if a["tipo"] != "equipo"}
    assert sorted(members) == sorted(services) and len(comms) == 3
    assert [c.id for c in comms] == [1, 2, 3] and G.detect_communities(curated) == comms           # repetible
    where = {m: c.id for c in comms for m in c.members}
    assert where["Pasarela de pagos"] == where["Portal Mi Andes"] and where["Red troncal"] == where["Base de datos de clientes"]


def test_community_prompt_inputs_include_how_many_depend_on_each_member(curated):
    inputs = G.community_prompt_inputs(curated, G.detect_communities(curated)[1])
    assert {m["name"]: m["dependents"] for m in inputs["members"]}["Servicio de autenticación"] == 5
    assert inputs["teams"] and inputs["external"]


def test_stats(curated):
    stats = G.graph_stats(curated)
    assert (stats["nodes"], stats["edges"], stats["components"], stats["has_cycles"]) == (24, 40, 1, False)
    assert stats["most_depended_on"][0] == {"name": "Servicio de autenticación", "degree": 5}


def test_a_cycle_is_detected():
    data = GraphData({"A": {"tipo": "x"}, "B": {"tipo": "x"}}, [Triple("A", "depende_de", "B"), Triple("B", "depende_de", "A")])
    assert G.graph_stats(data)["has_cycles"]


def test_expected_answers_match_the_curated_graph(curated):
    """Los datos de la evaluación se validan contra el grafo: si alguien edita una pregunta o el JSON, esta prueba avisa."""
    impact = G.to_digraph(curated, G.IMPACT_RELATIONS)
    team = {t.subject: t.object for t in curated.triples if t.relation == "administrado_por"}
    ent = curated.entities
    for q in evaluation.load_questions():
        seeds = [e for e, _ in G.link_by_name(ent, q["pregunta"])]
        if q["id"] in ("MH1", "MH2", "MH8", "EX4"):
            assert sorted(nx.ancestors(impact, seeds[0])) == sorted(q["esperadas"]), q["id"]
        elif q["id"] == "MH3":
            assert sorted(nx.descendants(impact, seeds[0])) == sorted(q["esperadas"])
        elif q["id"] == "MH4":
            assert q["esperadas"] == [team["Base de datos de clientes"]]
        elif q["id"] == "MH5":
            affected = nx.ancestors(impact, seeds[0]) | {seeds[0]}
            assert sorted({team[n] for n in affected if n in team}) == sorted(q["esperadas"])
        elif q["id"] == "MH6":
            assert sorted(q["esperadas"]) == sorted([*nx.ancestors(impact, seeds[0]), team[seeds[0]]])
        elif q["id"] == "MH7":
            assert max(impact.in_degree(n) for n in impact if ent[n]["tipo"] != "equipo") == impact.in_degree(q["esperadas"][0]) == 5
        elif q["id"] == "EX1":
            assert q["esperadas"] == [team[seeds[0]]]
        elif q["id"] == "EX2":
            assert q["esperadas"] == [o for s, o in impact.out_edges(seeds[0]) if ent[o]["tipo"] == "infraestructura"]
        elif q["id"] == "EX3":
            assert sorted(q["esperadas"]) == sorted(impact.successors(seeds[0]))
