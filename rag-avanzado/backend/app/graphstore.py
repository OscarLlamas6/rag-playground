"""Almacén del GRAFO: entidades, relaciones (con su procedencia) y comunidades (con su resumen).

    PgGraphStore      PostgreSQL. El grafo vive en tablas normales y los recorridos son CTE RECURSIVAS de SQL:
                      no hace falta una base de grafos (Neo4j) para un grafo pequeño o mediano, y los vectores, el texto
                      y el grafo quedan en la misma base, con las mismas copias de seguridad y transacciones.
    MemoryGraphStore  Los mismos datos en listas de Python y networkx. Para pruebas y para `task demo` sin Docker.

Los dos devuelven EXACTAMENTE lo mismo para el mismo recorrido (hay una prueba que lo compara) y los dos devuelven el SQL
de la consulta, que la interfaz muestra: en memoria es el SQL "equivalente" de lo que se calculó con Python.

Tablas:
    av_entities     (graph, name, type, aliases, embedding)        una fila por entidad; el vector sirve para vincular entidades
    av_relations    (graph, subject, relation, object, section, evidence)    una fila por tripleta, con su procedencia
    av_communities  (graph, id, title, members, summary)           una fila por comunidad, con su resumen
`graph` es 'curated' o 'extracted': los dos grafos conviven para poder compararlos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import Settings, get_settings
from .domain import GraphData, Triple
from .graph import Community
from .pg import PgConnection, StoreError

DIRECTIONS = ("reverse", "forward", "both")

# Recorrido AL REVÉS (impacto): parte de la entidad que falla y sube por quienes dependen de ella.
# `frontier` es el nodo desde el que se sigue caminando; `path` evita dar vueltas en un ciclo.
SQL_REVERSE = """WITH RECURSIVE walk AS (
    SELECT r.subject, r.relation, r.object, r.section, r.evidence,
           1 AS depth, r.object AS seed, ARRAY[r.object, r.subject] AS path, r.subject AS frontier
    FROM av_relations r
    WHERE r.graph = %(graph)s AND r.object = ANY(%(seeds)s) AND r.relation = ANY(%(relations)s)
  UNION ALL
    SELECT r.subject, r.relation, r.object, r.section, r.evidence,
           w.depth + 1, w.seed, w.path || r.subject, r.subject
    FROM av_relations r
    JOIN walk w ON r.object = w.frontier
    WHERE r.graph = %(graph)s AND r.relation = ANY(%(relations)s)
      AND w.depth < %(hops)s AND NOT r.subject = ANY(w.path)
)
SELECT subject, relation, object, section, evidence,
       min(depth) AS depth, (array_agg(seed ORDER BY depth, seed))[1] AS seed
FROM walk
GROUP BY subject, relation, object, section, evidence
ORDER BY min(depth), subject, object;"""

# Recorrido HACIA ADELANTE (dependencias): parte de la entidad y baja por aquello de lo que depende.
SQL_FORWARD = """WITH RECURSIVE walk AS (
    SELECT r.subject, r.relation, r.object, r.section, r.evidence,
           1 AS depth, r.subject AS seed, ARRAY[r.subject, r.object] AS path, r.object AS frontier
    FROM av_relations r
    WHERE r.graph = %(graph)s AND r.subject = ANY(%(seeds)s) AND r.relation = ANY(%(relations)s)
  UNION ALL
    SELECT r.subject, r.relation, r.object, r.section, r.evidence,
           w.depth + 1, w.seed, w.path || r.object, r.object
    FROM av_relations r
    JOIN walk w ON r.subject = w.frontier
    WHERE r.graph = %(graph)s AND r.relation = ANY(%(relations)s)
      AND w.depth < %(hops)s AND NOT r.object = ANY(w.path)
)
SELECT subject, relation, object, section, evidence,
       min(depth) AS depth, (array_agg(seed ORDER BY depth, seed))[1] AS seed
FROM walk
GROUP BY subject, relation, object, section, evidence
ORDER BY min(depth), subject, object;"""

SQL_TEAMS = """SELECT subject, relation, object, section, evidence
FROM av_relations
WHERE graph = %(graph)s AND relation = 'administrado_por' AND subject = ANY(%(nodes)s)
ORDER BY subject;"""

SQL_LINK = """SELECT name, 1 - (embedding <=> %(query)s) AS similitud
FROM av_entities
WHERE graph = %(graph)s AND embedding IS NOT NULL
ORDER BY embedding <=> %(query)s
LIMIT %(k)s;"""


def display_sql(sql: str) -> str:
    """Para mostrarlo: `%(seeds)s` → `:seeds` (los parámetros se muestran aparte)."""
    return re.sub(r"%\((\w+)\)s", r":\1", sql)


@dataclass
class Traversal:
    """Resultado de un recorrido: las aristas halladas (con su distancia a la entidad de partida) y las consultas ejecutadas."""

    edges: list[dict] = field(default_factory=list)       # {subject, relation, object, section, evidence, depth, seed, direction}
    queries: list[dict] = field(default_factory=list)     # {direction, sql, params}


class GraphStore(Protocol):
    name: str

    def save(self, data: GraphData, vectors: dict[str, list[float]] | None = None) -> None: ...
    def load(self, source: str) -> GraphData | None: ...
    def sources(self) -> dict[str, dict]: ...
    def search_entities(self, source: str, vector: list[float], k: int = 3) -> list[tuple[str, float]]: ...
    def traverse(self, source: str, seeds: list[str], direction: str, hops: int, relations: tuple[str, ...]) -> Traversal: ...
    def team_facts(self, source: str, nodes: list[str]) -> list[dict]: ...
    def save_communities(self, source: str, communities: list[Community]) -> None: ...
    def load_communities(self, source: str) -> list[Community]: ...
    def drop(self) -> None: ...


def _check(direction: str, hops: int) -> None:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction debe ser una de: {', '.join(DIRECTIONS)}")
    if not 1 <= hops <= 8:
        raise ValueError("hops debe estar entre 1 y 8")


def _merge(edges: list[dict]) -> list[dict]:
    """Une los recorridos de las dos direcciones: cada arista queda una vez, con su menor distancia."""
    best: dict[tuple, dict] = {}
    for e in edges:
        key = (e["subject"], e["relation"], e["object"])
        if key not in best or e["depth"] < best[key]["depth"]:
            best[key] = e
    return sorted(best.values(), key=lambda e: (e["depth"], e["subject"], e["object"]))


def _directions(direction: str) -> list[str]:
    return ["reverse", "forward"] if direction == "both" else [direction]


class MemoryGraphStore:
    name = "memory"

    def __init__(self) -> None:
        self._graphs: dict[str, GraphData] = {}
        self._vectors: dict[str, dict[str, np.ndarray]] = {}
        self._communities: dict[str, list[Community]] = {}

    def save(self, data: GraphData, vectors: dict[str, list[float]] | None = None) -> None:
        self._graphs[data.source] = data
        unit = lambda v: np.array(v, dtype=np.float32) / (np.linalg.norm(v) or 1.0)   # noqa: E731
        self._vectors[data.source] = {n: unit(v) for n, v in (vectors or {}).items()}
        self._communities.pop(data.source, None)

    def load(self, source: str) -> GraphData | None:
        return self._graphs.get(source)

    def sources(self) -> dict[str, dict]:
        return {s: {"entities": len(g.entities), "triples": len(g.triples)} for s, g in self._graphs.items()}

    def search_entities(self, source: str, vector: list[float], k: int = 3) -> list[tuple[str, float]]:
        vectors = self._vectors.get(source) or {}
        if not vectors:
            return []
        q = np.array(vector, dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        scored = sorted(((n, float(v @ q)) for n, v in vectors.items()), key=lambda x: -x[1])
        return scored[:k]

    def _walk(self, data: GraphData, seeds: list[str], direction: str, hops: int, relations: tuple[str, ...]) -> list[dict]:
        """Lo mismo que la CTE recursiva: recorre caminos simples (sin repetir nodos) de hasta `hops` aristas."""
        reverse = direction == "reverse"
        index: dict[str, list[Triple]] = {}
        for t in data.triples:
            if t.relation in relations:
                index.setdefault(t.object if reverse else t.subject, []).append(t)
        best: dict[tuple, dict] = {}

        def go(frontier: str, depth: int, path: tuple[str, ...], seed: str) -> None:
            for t in index.get(frontier, []):
                new = t.subject if reverse else t.object
                if new in path:
                    continue
                if t.key not in best or depth < best[t.key]["depth"]:
                    best[t.key] = {"subject": t.subject, "relation": t.relation, "object": t.object, "section": t.section,
                                   "evidence": t.evidence, "depth": depth, "seed": seed, "direction": direction}
                if depth < hops:
                    go(new, depth + 1, (*path, new), seed)

        for seed in sorted(seeds):
            go(seed, 1, (seed,), seed)
        return list(best.values())

    def traverse(self, source: str, seeds: list[str], direction: str, hops: int, relations: tuple[str, ...]) -> Traversal:
        _check(direction, hops)
        data = self._graphs.get(source)
        if data is None:
            raise StoreError(f"El grafo «{source}» no existe: constrúyelo primero en la pestaña Grafo.")
        out = Traversal()
        for d in _directions(direction):
            out.edges += self._walk(data, seeds, d, hops, relations)
            sql = SQL_REVERSE if d == "reverse" else SQL_FORWARD
            out.queries.append({"direction": d, "sql": display_sql(sql), "equivalent": True,
                                "params": {"graph": source, "seeds": seeds, "relations": list(relations), "hops": hops}})
        out.edges = _merge(out.edges)
        return out

    def team_facts(self, source: str, nodes: list[str]) -> list[dict]:
        data = self._graphs.get(source)
        if data is None:
            return []
        return [{"subject": t.subject, "relation": t.relation, "object": t.object, "section": t.section, "evidence": t.evidence}
                for t in sorted(data.triples, key=lambda t: t.subject) if t.relation == "administrado_por" and t.subject in nodes]

    def save_communities(self, source: str, communities: list[Community]) -> None:
        self._communities[source] = communities

    def load_communities(self, source: str) -> list[Community]:
        return list(self._communities.get(source, []))

    def drop(self) -> None:
        self._graphs, self._vectors, self._communities = {}, {}, {}


class PgGraphStore:
    name = "postgres"

    def __init__(self, url: str) -> None:
        self.db = PgConnection(url)

    # ───────────── esquema ─────────────
    def _ensure(self, c, dimension: int | None = None) -> None:
        """Crea las tablas si no existen. Si cambió la dimensión de los embeddings, las recrea (hay que reconstruir los grafos)."""
        if dimension:
            row = c.execute("SELECT a.atttypmod FROM pg_attribute a WHERE a.attrelid = to_regclass('av_entities') AND a.attname = 'embedding'").fetchone()
            if row and row[0] != dimension:
                c.execute("DROP TABLE IF EXISTS av_entities, av_relations, av_communities")
        vec = f"vector({int(dimension)})" if dimension else "vector"
        c.execute(f"CREATE TABLE IF NOT EXISTS av_entities (graph text NOT NULL, name text NOT NULL, type text NOT NULL, "
                  f"aliases jsonb NOT NULL DEFAULT '[]', embedding {vec}, PRIMARY KEY (graph, name))")
        c.execute("CREATE TABLE IF NOT EXISTS av_relations (graph text NOT NULL, subject text NOT NULL, relation text NOT NULL, "
                  "object text NOT NULL, section text NOT NULL DEFAULT '', evidence text NOT NULL DEFAULT '', "
                  "PRIMARY KEY (graph, subject, relation, object))")
        c.execute("CREATE INDEX IF NOT EXISTS av_relations_object ON av_relations (graph, object)")
        c.execute("CREATE INDEX IF NOT EXISTS av_relations_subject ON av_relations (graph, subject)")
        c.execute("CREATE TABLE IF NOT EXISTS av_communities (graph text NOT NULL, id int NOT NULL, title text NOT NULL, "
                  "members jsonb NOT NULL, summary text NOT NULL DEFAULT '', PRIMARY KEY (graph, id))")

    def _exists(self, c) -> bool:
        return bool(c.execute("SELECT to_regclass('av_relations')").fetchone()[0])

    # ───────────── escritura ─────────────
    def save(self, data: GraphData, vectors: dict[str, list[float]] | None = None) -> None:
        from psycopg.types.json import Jsonb

        dimension = len(next(iter(vectors.values()))) if vectors else None
        if dimension is None:       # sin vectores: se reutiliza la dimensión que ya tenga la tabla (o se deja libre)
            dimension = self.db.run(lambda c: (c.execute("SELECT a.atttypmod FROM pg_attribute a WHERE a.attrelid = to_regclass('av_entities') "
                                                         "AND a.attname = 'embedding'").fetchone() or [None])[0])
            dimension = dimension if dimension and dimension > 0 else None

        def go(c):
            self._ensure(c, dimension)
            for table in ("av_entities", "av_relations", "av_communities"):
                c.execute(f"DELETE FROM {table} WHERE graph = %s", (data.source,))
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO av_entities (graph, name, type, aliases, embedding) VALUES (%s, %s, %s, %s, %s)",
                    [(data.source, n, a.get("tipo", "otro"), Jsonb(a.get("alias", [])),
                      np.array(vectors[n], dtype=np.float32) if vectors and n in vectors else None) for n, a in data.entities.items()])
                cur.executemany(
                    "INSERT INTO av_relations (graph, subject, relation, object, section, evidence) VALUES (%s, %s, %s, %s, %s, %s)",
                    [(data.source, t.subject, t.relation, t.object, t.section, t.evidence) for t in data.triples])
        self.db.run(go)

    def save_communities(self, source: str, communities: list[Community]) -> None:
        from psycopg.types.json import Jsonb

        def go(c):
            self._ensure(c)
            c.execute("DELETE FROM av_communities WHERE graph = %s", (source,))
            with c.cursor() as cur:
                cur.executemany("INSERT INTO av_communities (graph, id, title, members, summary) VALUES (%s, %s, %s, %s, %s)",
                                [(source, k.id, k.title, Jsonb(k.members), k.summary) for k in communities])
        self.db.run(go)

    # ───────────── lectura ─────────────
    def load(self, source: str) -> GraphData | None:
        def go(c):
            if not self._exists(c):
                return None
            ents = c.execute("SELECT name, type, aliases FROM av_entities WHERE graph = %s ORDER BY name", (source,)).fetchall()
            if not ents:
                return None
            rels = c.execute("SELECT subject, relation, object, section, evidence FROM av_relations WHERE graph = %s "
                             "ORDER BY subject, relation, object", (source,)).fetchall()
            return GraphData({n: {"tipo": t, "alias": a} for n, t, a in ents}, [Triple(*r) for r in rels], source)
        return self.db.run(go)

    def sources(self) -> dict[str, dict]:
        def go(c):
            if not self._exists(c):
                return {}
            ents = dict(c.execute("SELECT graph, count(*) FROM av_entities GROUP BY graph").fetchall())
            rels = dict(c.execute("SELECT graph, count(*) FROM av_relations GROUP BY graph").fetchall())
            return {g: {"entities": n, "triples": rels.get(g, 0)} for g, n in ents.items()}
        return self.db.run(go)

    def search_entities(self, source: str, vector: list[float], k: int = 3) -> list[tuple[str, float]]:
        q = np.array(vector, dtype=np.float32)

        def go(c):
            if not self._exists(c):
                return []
            return c.execute(SQL_LINK, {"graph": source, "query": q, "k": k}).fetchall()
        return [(n, float(s)) for n, s in self.db.run(go)]

    def traverse(self, source: str, seeds: list[str], direction: str, hops: int, relations: tuple[str, ...]) -> Traversal:
        _check(direction, hops)
        out = Traversal()
        params = {"graph": source, "seeds": list(seeds), "relations": list(relations), "hops": hops}
        for d in _directions(direction):
            sql = SQL_REVERSE if d == "reverse" else SQL_FORWARD
            rows = self.db.run(lambda c, sql=sql: c.execute(sql, params).fetchall())
            out.edges += [{"subject": s, "relation": r, "object": o, "section": sec, "evidence": ev, "depth": depth,
                           "seed": seed, "direction": d} for s, r, o, sec, ev, depth, seed in rows]
            out.queries.append({"direction": d, "sql": display_sql(sql), "equivalent": False, "params": params})
        out.edges = _merge(out.edges)
        return out

    def team_facts(self, source: str, nodes: list[str]) -> list[dict]:
        rows = self.db.run(lambda c: c.execute(SQL_TEAMS, {"graph": source, "nodes": list(nodes)}).fetchall())
        return [{"subject": s, "relation": r, "object": o, "section": sec, "evidence": ev} for s, r, o, sec, ev in rows]

    def load_communities(self, source: str) -> list[Community]:
        def go(c):
            if not self._exists(c):
                return []
            return c.execute("SELECT id, title, members, summary FROM av_communities WHERE graph = %s ORDER BY id", (source,)).fetchall()
        return [Community(i, t, m, s) for i, t, m, s in self.db.run(go)]

    def drop(self) -> None:
        self.db.run(lambda c: c.execute("DROP TABLE IF EXISTS av_entities, av_relations, av_communities"))


def create_graph_store(settings: Settings | None = None) -> GraphStore:
    settings = settings or get_settings()
    return MemoryGraphStore() if settings.store == "memory" else PgGraphStore(settings.database_url)
