"""El grafo de conocimiento: construirlo, extraerlo con un LLM, normalizarlo, validarlo, medirlo y agruparlo en comunidades.

    texto del catálogo ──(LLM extrae)──► tripletas crudas ──► normalizar ──► validar ──► grafo ──► comunidades
                                                                  │             │
                                           "la Red troncal" → "Red troncal"   rechaza lo inventado o sin respaldo en el texto

Dos orígenes (`source`):
    curated    un archivo JSON escrito a mano (`data/graph/servicios_ti_grafo.json`): gratis y fiable. Es la referencia.
    extracted  un LLM lee cada sección del catálogo y propone tripletas. El resultado se MIDE contra el curado:
               precisión (¿qué fracción de lo extraído es correcto?) y recall (¿cuánto de lo curado se encontró?).

Cada tripleta guarda su PROCEDENCIA: la sección y la oración exacta del catálogo que la respalda. Eso permite citarla y
volver al texto original desde un hecho del grafo.

Los recorridos por el grafo (que en la base de datos son CTE recursivas) están en `graphstore.py`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from pydantic import AliasChoices, BaseModel, Field, ValidationError

from . import prompts
from .cache import cached_generate
from .domain import Document, GraphData, Triple
from .embeddings import strip_accents
from .llm import LLMResult

RELATIONS = ("depende_de", "corre_en", "administrado_por")
IMPACT_RELATIONS = ("depende_de", "corre_en")        # las que propagan una falla; "administrado_por" no
RELATION_PHRASE = {"depende_de": "depende de", "corre_en": "se ejecuta en", "administrado_por": "es administrado por"}
# Cómo suelen escribir la relación los modelos; "inverso" = el modelo escribió la relación al revés (A administra B).
RELATION_ALIASES = {
    "depende_de": "depende_de", "depende": "depende_de", "dependede": "depende_de", "depends_on": "depende_de", "necesita": "depende_de",
    "requiere": "depende_de", "corre_en": "corre_en", "se_ejecuta_en": "corre_en", "ejecuta_en": "corre_en", "runs_on": "corre_en",
    "aloja_en": "corre_en", "administrado_por": "administrado_por", "es_administrado_por": "administrado_por",
    "lo_administra": "administrado_por", "managed_by": "administrado_por",
}
INVERSE_RELATIONS = {"administra": "administrado_por", "es_administrador_de": "administrado_por"}


def normalize(text: str) -> str:
    """Minúsculas y sin acentos, para comparar 'Autenticación' con 'autenticacion'."""
    return strip_accents(text).lower()


def fact_phrase(subject: str, relation: str, obj: str) -> str:
    return f"{subject} {RELATION_PHRASE.get(relation, relation)} {obj}."


# ───────────────────────── entidades: nombres, alias y vinculación ─────────────────────────

_ARTICLE = re.compile(r"^(el|la|los|las|del|al)\s+", re.IGNORECASE)


def canonical_name(entities: dict[str, dict], name: str) -> str | None:
    """Nombre oficial de una entidad escrita de cualquier forma (con alias, sin artículo, sin acentos), o None."""
    clean = normalize(_ARTICLE.sub("", name.strip().strip("«»\"'")))
    for node, attrs in entities.items():
        if clean in {normalize(node), *(normalize(a) for a in attrs.get("alias", []))}:
            return node
    return None


def link_by_name(entities: dict[str, dict], text: str) -> list[tuple[str, str]]:
    """Entidades mencionadas en un texto, por nombre o alias (la vinculación de entidades más simple posible).

    Devuelve (entidad, texto con el que coincidió) en el orden en que aparecen. Busca primero los nombres más largos y
    tapa lo ya encontrado, para que el alias "base de datos" no se cuente dentro de "Base de datos de clientes".
    """
    t = normalize(text)
    candidates = sorted(((normalize(alias), node) for node, attrs in entities.items() for alias in [node, *attrs.get("alias", [])]),
                        key=lambda c: -len(c[0]))
    found: list[tuple[int, str, str]] = []
    for alias, node in candidates:
        m = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", t)
        if m:
            found.append((m.start(), node, alias))
            t = t[: m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    seen: dict[str, str] = {}
    for _, node, alias in sorted(found):
        seen.setdefault(node, alias)
    return list(seen.items())


def mentions(entities: dict[str, dict], text: str, name: str) -> bool:
    """¿El texto menciona a esa entidad (por su nombre o por un alias)?"""
    aliases = [normalize(name), *(normalize(a) for a in entities.get(name, {}).get("alias", []))]
    t = normalize(text)
    return any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", t) for a in aliases)


# ───────────────────────── procedencia ─────────────────────────


def sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip()) if p.strip()]


def find_evidence(sections: dict[str, str], entities: dict[str, dict], section: str, obj: str) -> str | None:
    """La primera oración de la sección que menciona al objeto de la tripleta: es la que respalda el hecho."""
    for sentence in sentences(sections.get(section, "")):
        if mentions(entities, sentence, obj):
            return sentence
    return None


def sections_of(documents: list[Document]) -> dict[str, str]:
    return {title: " ".join(text.split()) for d in documents for title, text in d.sections}


# ───────────────────────── 1. el grafo curado ─────────────────────────


def load_inventory(path: Path) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entities = {name: {"tipo": a.get("tipo", "otro"), "alias": list(a.get("alias", []))} for name, a in data["entidades"].items()}
    return entities, [tuple(t) for t in data["tripletas"]]


def build_curated(path: Path, documents: list[Document]) -> GraphData:
    """El grafo escrito a mano, con la procedencia de cada tripleta buscada en el catálogo."""
    entities, raw = load_inventory(path)
    sections = sections_of(documents)
    triples = [Triple(s, r, o, s if s in sections else "", find_evidence(sections, entities, s, o) or "") for s, r, o in raw]
    return GraphData(entities, triples, "curated")


# ───────────────────────── 2. extraer tripletas con un LLM ─────────────────────────


class RawTriple(BaseModel):
    sujeto: str = Field(validation_alias=AliasChoices("sujeto", "subject"))
    relacion: str = Field(validation_alias=AliasChoices("relacion", "relation", "relación"))
    objeto: str = Field(validation_alias=AliasChoices("objeto", "object"))


class RawExtraction(BaseModel):
    tripletas: list[RawTriple] = Field(validation_alias=AliasChoices("tripletas", "triplets", "triples"))


JSON_EXAMPLE = '{"tripletas": [{"sujeto": "nombre", "relacion": "depende_de", "objeto": "nombre"}]}'


def extract_json(text: str) -> str:
    """El primer objeto JSON del texto (los modelos pequeños lo envuelven en cercas de código o en explicaciones)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start: end + 1] if 0 <= start < end else text


@dataclass
class ExtractionRun:
    raw: list[tuple[str, RawTriple]] = field(default_factory=list)       # (sección, tripleta cruda)
    calls: list[LLMResult] = field(default_factory=list)
    cache_hits: int = 0
    failed_sections: list[str] = field(default_factory=list)


def extract_triples(sections: dict[str, str], entities: dict[str, dict], version: str = "v1", force: bool = False) -> ExtractionRun:
    """Una llamada al LLM por sección (las secciones sin relaciones también cuestan una llamada). Valida el JSON con pydantic
    y reintenta UNA vez explicando el error. Las respuestas se cachean en disco por (modelo, prompt): repetir es gratis."""
    run = ExtractionRun()
    for title, text in sections.items():
        prompt = prompts.render("extraer_tripletas", version, titulo=title, texto=text, entidades=list(entities)) + \
            f"\n\nResponde ÚNICAMENTE con un objeto JSON con esta forma exacta, sin texto adicional ni cercas de código:\n{JSON_EXAMPLE}"
        error = ""
        for attempt in (1, 2):
            call_prompt = prompt if attempt == 1 else f"{prompt}\n\nTu respuesta anterior fue inválida ({error}). Corrígela."
            result, hit = cached_generate("extraer", call_prompt, max_tokens=900, force=force and attempt == 1)
            run.calls.append(result)
            run.cache_hits += hit
            try:
                parsed = RawExtraction.model_validate_json(extract_json(result.text))
            except ValidationError as exc:
                error = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())[:200]
                continue
            run.raw += [(title, t) for t in parsed.tripletas]
            break
        else:
            run.failed_sections.append(title)
    return run


@dataclass
class Rejection:
    section: str
    triple: tuple[str, str, str]
    reason: str


def _canonicalize(entities: dict[str, dict], raw: RawTriple) -> tuple[tuple[str, str, str] | None, str]:
    """Pasos de NORMALIZACIÓN: llevar la relación y las dos entidades a su nombre oficial."""
    rel_key = normalize(raw.relacion).strip().replace(" ", "_").replace("-", "_")
    inverse = rel_key in INVERSE_RELATIONS
    relation = INVERSE_RELATIONS.get(rel_key) or RELATION_ALIASES.get(rel_key)
    if relation is None:
        return None, f"relación no permitida: «{raw.relacion}»"
    s, o = canonical_name(entities, raw.sujeto), canonical_name(entities, raw.objeto)
    for original, canonical in ((raw.sujeto, s), (raw.objeto, o)):
        if canonical is None:
            return None, f"entidad desconocida: «{original}»"
    if inverse:
        s, o = o, s
    return (s, relation, o), ""


def _validate(entities: dict[str, dict], section: str, sections: dict[str, str], triple: tuple[str, str, str]) -> str:
    """Pasos de VALIDACIÓN: reglas que un modelo rompe aunque los nombres sean correctos. Devuelve el motivo del rechazo o ''."""
    s, r, o = triple
    kind = lambda n: entities[n].get("tipo")      # noqa: E731
    if s == o:
        return "una entidad no puede depender de sí misma"
    if r == "administrado_por" and kind(o) != "equipo":
        return "el objeto de «administrado_por» debe ser un equipo"
    if r != "administrado_por" and (kind(s) == "equipo" or kind(o) == "equipo"):
        return "un equipo no participa en dependencias ni en ejecución"
    if r == "corre_en" and kind(o) != "infraestructura":
        return "«corre_en» apunta a infraestructura (centros de datos, redes)"
    if s != section:
        return f"el sujeto no es la sección «{section}» (posible arista falsa)"
    if find_evidence(sections, entities, section, o) is None:
        return "el objeto no aparece en el texto de la sección (sin respaldo)"
    return ""


@dataclass
class Normalization:
    graph: GraphData
    rejected: list[Rejection]
    duplicates: int
    canonical_only: set[tuple[str, str, str]]       # lo que quedaría SIN validar (solo normalizado)


def normalize_and_validate(run: ExtractionRun, entities: dict[str, dict], sections: dict[str, str]) -> Normalization:
    kept: dict[tuple[str, str, str], Triple] = {}
    rejected: list[Rejection] = []
    canonical_only: set[tuple[str, str, str]] = set()
    duplicates = 0
    for section, raw in run.raw:
        triple, reason = _canonicalize(entities, raw)
        original = (raw.sujeto, raw.relacion, raw.objeto)
        if triple is None:
            rejected.append(Rejection(section, original, reason))
            continue
        canonical_only.add(triple)
        reason = _validate(entities, section, sections, triple)
        if reason:
            rejected.append(Rejection(section, original, reason))
        elif triple in kept:
            duplicates += 1
        else:
            kept[triple] = Triple(*triple, section=section, evidence=find_evidence(sections, entities, section, triple[2]) or "")
    return Normalization(GraphData(entities, list(kept.values()), "extracted"), rejected, duplicates, canonical_only)


def compare_triples(curated: set[tuple[str, str, str]], extracted: set[tuple[str, str, str]]) -> dict:
    """Mide un conjunto de tripletas contra las curadas: precisión (¿qué fracción de lo extraído es correcto?) y recall
    (¿cuánto de lo curado se encontró?). Una arista de más o de menos cambia las respuestas de TODOS los recorridos que pasan por ella."""
    correct = curated & extracted
    precision = len(correct) / len(extracted) if extracted else 0.0
    recall = len(correct) / len(curated) if curated else 0.0
    return {"extracted": len(extracted), "correct": len(correct), "curated": len(curated),
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
            "missing": sorted(curated - extracted), "extra": sorted(extracted - curated)}


# ───────────────────────── 3. el grafo como red: estadísticas, comunidades ─────────────────────────


def to_digraph(data: GraphData, relations: tuple[str, ...] | None = None) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from((n, a) for n, a in data.entities.items())
    for t in data.triples:
        if relations is None or t.relation in relations:
            G.add_edge(t.subject, t.object, relation=t.relation)
    return G


def graph_stats(data: GraphData) -> dict:
    G = to_digraph(data)
    deps = to_digraph(data, IMPACT_RELATIONS)
    by_relation: dict[str, int] = defaultdict(int)
    for t in data.triples:
        by_relation[t.relation] += 1
    top_in = sorted(((n, deps.in_degree(n)) for n in deps if data.entities[n].get("tipo") != "equipo"), key=lambda x: (-x[1], x[0]))[:5]
    top_out = sorted(((n, deps.out_degree(n)) for n in deps if data.entities[n].get("tipo") != "equipo"), key=lambda x: (-x[1], x[0]))[:5]
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "by_relation": dict(by_relation),
            "components": nx.number_weakly_connected_components(G),
            "has_cycles": not nx.is_directed_acyclic_graph(deps),
            "avg_degree": round(2 * G.number_of_edges() / max(1, G.number_of_nodes()), 2),
            "most_depended_on": [{"name": n, "degree": d} for n, d in top_in],
            "most_dependencies": [{"name": n, "degree": d} for n, d in top_out]}


@dataclass
class Community:
    id: int
    title: str
    members: list[str]
    summary: str = ""


def detect_communities(data: GraphData, seed: int = 42) -> list[Community]:
    """Comunidades = grupos de servicios muy conectados entre sí (Louvain, de networkx: sin dependencias pesadas).

    Se agrupan solo los servicios (no los equipos) y solo por las relaciones que propagan fallas. Louvain optimiza la
    modularidad; con `seed` fija el resultado es repetible. Microsoft GraphRAG usa Leiden y niveles jerárquicos.
    """
    services = [n for n, a in data.entities.items() if a.get("tipo") != "equipo"]
    U = to_digraph(data, IMPACT_RELATIONS).subgraph(services).to_undirected()
    groups = nx.community.louvain_communities(U, seed=seed)
    ordered = sorted((sorted(g) for g in groups), key=lambda g: (-len(g), g[0]))
    out: list[Community] = []
    deps = to_digraph(data, IMPACT_RELATIONS)
    for i, members in enumerate(ordered, start=1):
        hub = max(members, key=lambda n: (deps.in_degree(n), n))
        out.append(Community(i, f"Comunidad {i} · alrededor de «{hub}»", members))
    return out


def community_prompt_inputs(data: GraphData, community: Community) -> dict:
    """Lo que se le entrega al LLM para resumir una comunidad: miembros, hechos internos y a cuántos sirve cada uno."""
    members = set(community.members)
    deps = to_digraph(data, IMPACT_RELATIONS)
    internal = [t for t in data.triples if t.subject in members and t.object in members and t.relation in IMPACT_RELATIONS]
    outside = [t for t in data.triples if t.relation in IMPACT_RELATIONS and (t.subject in members) != (t.object in members)]
    teams = sorted({t.object for t in data.triples if t.relation == "administrado_por" and t.subject in members})
    return {"members": [{"name": m, "type": data.entities[m].get("tipo", "otro"), "dependents": deps.in_degree(m)} for m in community.members],
            "internal": [fact_phrase(*t.key) for t in internal],
            "external": [fact_phrase(*t.key) for t in outside], "teams": teams}
