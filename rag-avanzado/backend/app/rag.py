"""El flujo del RAG avanzado, con una TRAZA de cada etapa.

    pregunta ─► [descomponer en subpreguntas] ─┐
                                               ▼
    modo vector        búsqueda vectorial (con o sin frase de contexto) ─► top-k
    modo graph_local   enlazar entidades ─► recorrer el grafo N saltos (CTE recursiva) ─► hechos + fragmentos de origen
    modo graph_global  resúmenes de comunidades ─► MAPA (qué dice cada una) ─► REDUCCIÓN (una respuesta)
    modo hybrid_graph  vectores + hechos del grafo en UN solo contexto numerado
                                               │
                  contexto numerado [1] [2]… ─► prompt con citas ─► LLM ─► verificar citas

El pipeline es FIJO y lo elige quien pregunta: no hay un agente que decida qué hacer (eso es el RAG agéntico, que se verá
más adelante). Sigue siendo one-shot: cada pregunta es independiente, sin historial.

Cada etapa devuelve lo que vio, cuánto tardó y cuánto costó, para que la interfaz pueda MOSTRAR qué hizo cada pieza.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

from . import graph as G
from . import prompts
from .cache import cached_generate
from .domain import Chunk, GraphData, Item
from .embeddings import EmbeddingError, embed, embed_query
from .fusion import rrf
from .graphstore import GraphStore, Traversal, create_graph_store
from .ingest import chunk_documents, load_documents, text_to_embed
from .config import get_settings
from .llm import NOT_FOUND, LLMResult, generate
from .store import ChunkStore, StoreError, chunk_item, create_store

MODES = ("vector", "graph_local", "graph_global", "hybrid_graph")
DIRECTIONS = ("auto", "reverse", "forward", "both")
KIND_LABEL = {"chunk": "fragmento", "fact": "hecho", "community": "comunidad"}

_FAILURE_WORDS = ("falla", "falle", "fallan", "cae", "caiga", "caen", "se cae", "apaga", "apague", "afecta", "afectan", "afectado",
                  "afectados", "deja de funcionar", "dejan de funcionar", "dejaria", "dejarian", "intervenir", "interrumpe", "sin servicio")
_FORWARD_WORDS = ("depende", "dependen de", "dependencias", "necesita", "requiere", "donde guarda", "se ejecuta", "corre en", "administra")
_TEAM_WORDS = ("equipo", "equipos", "administra", "administrado", "responsable", "intervenir")


class NeedsSetup(RuntimeError):
    """Falta un paso previo (ingestar, construir el grafo, resumir comunidades): la API lo traduce a 409."""


class IngestBusy(RuntimeError):
    """Ya hay una ingesta o una construcción en curso: reconstruyen datos compartidos y no se pueden hacer dos veces a la vez."""


def cited_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        numbers.update(int(n) for n in re.split(r"\s*,\s*", group))
    return numbers


def decide_direction(question: str, direction: str = "auto") -> tuple[str, str]:
    """Qué sentido recorrer. Con `auto` se usan REGLAS FIJAS por palabras (no un modelo que decida): una pregunta de impacto
    («si falla X…») recorre las dependencias AL REVÉS; una de dependencias («¿de qué depende X?») hacia adelante."""
    if direction != "auto":
        return direction, "elegido por ti"
    q = G.normalize(question)
    if any(w in q for w in _FAILURE_WORDS):
        return "reverse", "la pregunta habla de una falla o impacto: se recorre AL REVÉS (quién depende de la entidad)"
    if any(w in q for w in _FORWARD_WORDS):
        return "forward", "la pregunta habla de dependencias o de quién administra: se recorre HACIA ADELANTE (de qué depende la entidad)"
    return "both", "sin palabras de impacto ni de dependencia: se recorre en ambos sentidos"


def needs_teams(question: str) -> bool:
    q = G.normalize(question)
    return any(w in q for w in _TEAM_WORDS)


def item_view(item: Item, **extra) -> dict:
    return {"id": item.id, "kind": item.kind, "kind_label": KIND_LABEL[item.kind], "label": item.label, "text": item.text,
            "score": round(item.score, 4), "meta": item.meta, **extra}


@dataclass
class Stage:
    key: str
    title: str
    note: str
    items: list[dict]
    ms: float
    llm: dict | None = None
    sql: list[dict] | None = None


@dataclass
class Retrieval:
    items: list[Item]                                  # el contexto que recibe el modelo, en orden
    stages: list[Stage]
    llms: list[LLMResult] = field(default_factory=list)   # llamadas al LLM hechas ANTES de responder (descomponer, mapa)
    subgraph: dict | None = None
    sql: list[dict] = field(default_factory=list)
    direction: str | None = None

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.llms)


@dataclass
class Answer:
    question: str
    text: str
    items: list[Item]
    cited: set[int]
    invalid: set[int]
    abstained: bool
    prompt: str | None
    llm: LLMResult | None
    retrieval: Retrieval
    timings_ms: dict[str, float]

    @property
    def uncited(self) -> bool:
        return not self.abstained and not self.cited

    @property
    def cost_usd(self) -> float:
        return self.retrieval.cost_usd + (self.llm.cost_usd if self.llm else 0.0)


@dataclass
class IngestReport:
    documents: int
    chunks: int
    dimension: int
    store: str
    contextual: bool
    context_calls: int
    context_cache_hits: int
    cost_usd: float
    seconds: float
    heading: bool
    preview: list[dict] = field(default_factory=list)
    per_section: dict[str, int] = field(default_factory=dict)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class RagService:
    def __init__(self, store: ChunkStore | None = None, graphs: GraphStore | None = None) -> None:
        self.store: ChunkStore = store or create_store()
        self.graphs: GraphStore = graphs or create_graph_store()
        self._lock = threading.Lock()

    # ═════════════════════════ ingesta ═════════════════════════
    def ingest(self, strategy: str = "sentences", max_words: int = 25, overlap: int = 0, contextual: bool = False,
               heading: bool = True, force_context: bool = False) -> IngestReport:
        if not self._lock.acquire(blocking=False):
            raise IngestBusy("Ya hay una operación pesada en curso (ingesta o construcción del grafo): espera a que termine.")
        try:
            return self._ingest(strategy, max_words, overlap, contextual, heading, force_context)
        finally:
            self._lock.release()

    def _ingest(self, strategy: str, max_words: int, overlap: int, contextual: bool, heading: bool, force: bool) -> IngestReport:
        t0 = time.perf_counter()
        documents = load_documents(get_settings().docs_dir)
        chunks = chunk_documents(documents, strategy, max_words, overlap)
        calls = hits = 0
        cost = 0.0
        if contextual:
            whole = {d.source: "\n\n".join(f"## {t}\n{x}" for t, x in d.sections) for d in documents}
            for c in chunks:
                prompt = prompts.render("contexto_fragmento", "v1", documento=whole[c.metadata["source"]],
                                        seccion=c.metadata["section"], fragmento=c.text)
                result, hit = cached_generate("contexto", prompt, max_tokens=120, force=force)
                c.context = " ".join(result.text.split()).strip(" \"'«»")
                calls += not hit
                hits += hit
                cost += result.cost_usd
        vectors = embed([text_to_embed(c, heading) for c in chunks], kind="document")
        for c, v in zip(chunks, vectors):
            c.vector = v
        if contextual:
            for c, v in zip(chunks, embed([text_to_embed(c, heading, with_context=True) for c in chunks], kind="document")):
                c.vector_ctx = v
        self.store.reset(len(vectors[0]))
        self.store.upsert(chunks)
        per_section: dict[str, int] = {}
        for c in chunks:
            per_section[c.metadata["section"]] = per_section.get(c.metadata["section"], 0) + 1
        preview = [{"id": c.id, "section": c.metadata["section"], "text": c.text, "context": c.context} for c in chunks[:6]]
        return IngestReport(len(documents), len(chunks), len(vectors[0]), self.store.name, contextual, calls, hits, cost,
                            time.perf_counter() - t0, heading, preview, per_section)

    # ═════════════════════════ construcción del grafo ═════════════════════════
    def build_graph(self, source: str = "curated", prompt_version: str = "v1", force: bool = False) -> dict:
        if source not in ("curated", "extracted"):
            raise ValueError("source debe ser 'curated' o 'extracted'")
        if not self._lock.acquire(blocking=False):
            raise IngestBusy("Ya hay una operación pesada en curso (ingesta o construcción del grafo): espera a que termine.")
        try:
            return self._build_graph(source, prompt_version, force)
        finally:
            self._lock.release()

    def _build_graph(self, source: str, prompt_version: str, force: bool) -> dict:
        t0 = time.perf_counter()
        settings = get_settings()
        documents = load_documents(settings.docs_dir)
        curated = G.build_curated(settings.graph_file, documents)
        report: dict = {"source": source, "warnings": []}
        if source == "curated":
            data = curated
        else:
            sections = G.sections_of(documents)
            run = G.extract_triples({t: x for t, x in sections.items() if t != "Introducción"}, curated.entities, prompt_version, force)
            norm = G.normalize_and_validate(run, curated.entities, sections)
            data = norm.graph
            truth = {t.key for t in curated.triples}
            report["extraction"] = {
                "prompt_version": prompt_version, "sections": len(sections) - ("Introducción" in sections), "calls": len(run.calls) - run.cache_hits,
                "cache_hits": run.cache_hits,
                "failed_sections": run.failed_sections, "raw_triples": len(run.raw),
                "cost_usd": round(sum(c.cost_usd for c in run.calls), 6),
                "tokens_in": sum(c.tokens_in for c in run.calls), "tokens_out": sum(c.tokens_out for c in run.calls),
                "llm_seconds": round(sum(c.seconds for c in run.calls), 2),
                "duplicates": norm.duplicates,
                "rejected": [{"section": r.section, "triple": list(r.triple), "reason": r.reason} for r in norm.rejected],
                "metrics_without_validation": G.compare_triples(truth, norm.canonical_only),
                "metrics": G.compare_triples(truth, {t.key for t in data.triples}),
            }
        vectors = None
        try:
            names = list(data.entities)
            texts = ["; ".join([n, *data.entities[n].get("alias", [])]) for n in names]
            vectors = dict(zip(names, embed(texts, kind="document")))
        except EmbeddingError as exc:
            report["warnings"].append(f"Sin vectores de entidades (el enlace por similitud queda desactivado): {exc}")
        self.graphs.save(data, vectors)
        communities = G.detect_communities(data)
        self.graphs.save_communities(source, communities)
        report.update({"entities": len(data.entities), "triples": len(data.triples), "stats": G.graph_stats(data),
                       "communities": len(communities), "seconds": round(time.perf_counter() - t0, 2)})
        return report

    def summarize_communities(self, source: str = "curated", force: bool = False) -> dict:
        """Un resumen por comunidad (1 llamada al LLM cada una; cacheado en disco y guardado en la base)."""
        data = self._graph(source)
        communities = self.graphs.load_communities(source) or G.detect_communities(data)
        calls = hits = 0
        cost = seconds = 0.0
        for c in communities:
            prompt = prompts.render("resumen_comunidad", "v1", comunidad=G.community_prompt_inputs(data, c))
            result, hit = cached_generate("resumen", prompt, max_tokens=300, force=force)
            c.summary = " ".join(result.text.split())
            calls += not hit
            hits += hit
            cost += result.cost_usd
            seconds += result.seconds
        self.graphs.save_communities(source, communities)
        return {"source": source, "communities": len(communities), "calls": calls, "cache_hits": hits,
                "cost_usd": round(cost, 6), "llm_seconds": round(seconds, 2)}

    def _graph(self, source: str) -> GraphData:
        data = self.graphs.load(source)
        if data is None:
            raise NeedsSetup(f"El grafo «{source}» no está construido: ve a la pestaña Grafo y pulsa «Construir grafo»"
                             f"{' (el extraído usa el LLM)' if source == 'extracted' else ''}.")
        return data

    def graph_view(self, source: str) -> dict:
        """El grafo para dibujarlo. Si aún no se construyó devuelve `built: false` (no es un error: es el estado inicial)."""
        data = self.graphs.load(source)
        if data is None:
            return {"source": source, "built": False, "nodes": [], "edges": [], "communities": [], "stats": None}
        communities = self.graphs.load_communities(source)
        where = {m: c.id for c in communities for m in c.members}
        return {"source": source, "built": True, "stats": G.graph_stats(data),
                "nodes": [{"id": n, "type": a.get("tipo", "otro"), "aliases": a.get("alias", []), "community": where.get(n)}
                          for n, a in data.entities.items()],
                "edges": [{"subject": t.subject, "relation": t.relation, "object": t.object, "section": t.section, "evidence": t.evidence}
                          for t in data.triples],
                "communities": [{"id": c.id, "title": c.title, "members": c.members, "summary": c.summary} for c in communities]}

    # ═════════════════════════ recuperación: piezas ═════════════════════════
    def chunk_count(self) -> int:
        """Cuántos fragmentos hay. Si NO se puede contar, es un error del almacén (503), nunca "está vacío" (409)."""
        try:
            return self.store.count()
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"No se pudo consultar el almacén ({type(exc).__name__}): {exc}") from exc

    def _require_chunks(self) -> None:
        if self.chunk_count() == 0:
            raise NeedsSetup("El almacén de fragmentos está vacío: ejecuta primero la ingesta.")

    def _vector(self, question: str, k: int, contextual: bool) -> list[Item]:
        return self.store.search(embed_query(question), k, contextual)

    def _decompose(self, question: str, maximum: int = 3) -> tuple[list[str], LLMResult | None]:
        """Pide al LLM separar la pregunta en subpreguntas. Si no devuelve JSON válido, usa la pregunta original."""
        prompt = prompts.render("descomponer", "v1", pregunta=question, maximo=maximum)
        result = generate(prompt, max_tokens=250, purpose="descomponer")
        try:
            subs = [s.strip() for s in json.loads(G.extract_json(result.text))["subpreguntas"] if str(s).strip()][:maximum]
        except (ValueError, KeyError, TypeError):
            subs = []
        return subs or [question], result

    def _link(self, question: str, source: str, data: GraphData, threshold: float, max_seeds: int) -> list[dict]:
        """Vinculación de entidades: por coincidencia de nombre o alias, y por similitud de embedding con las entidades del grafo."""
        links = [{"entity": e, "method": "nombre", "score": 1.0, "matched": m} for e, m in G.link_by_name(data.entities, question)]
        known = {x["entity"] for x in links}
        for name, score in self.graphs.search_entities(source, embed_query(question), k=3):
            if name not in known and score >= threshold and len(links) < max(max_seeds, len(known)):
                links.append({"entity": name, "method": "embedding", "score": round(score, 3), "matched": ""})
        return links

    def _chunk_for(self, section: str, evidence: str, memo: dict | None = None) -> Chunk | None:
        """El fragmento donde está la oración que respalda un hecho (por sección y texto)."""
        memo = memo if memo is not None else {}
        if section not in memo:
            memo[section] = self.store.by_section(section) if section else []
        chunks = memo[section]
        norm = G.normalize(" ".join(evidence.split()))
        return next((c for c in chunks if norm and norm in G.normalize(c.text)), chunks[0] if chunks else None)

    def _facts(self, question: str, source: str, hops: int, direction: str, teams: bool | None, max_facts: int,
               threshold: float, with_chunks: bool, k: int) -> Retrieval:
        """El corazón de la búsqueda local: enlazar entidades → recorrer N saltos → hechos con procedencia (+ fragmentos de origen)."""
        data = self._graph(source)
        stages: list[Stage] = []

        t0 = time.perf_counter()
        links = self._link(question, source, data, threshold, max_seeds=3)
        stages.append(Stage("link", "Enlazar entidades",
                            "Qué entidades del grafo menciona la pregunta: por nombre o alias (exacto) y por similitud de embedding (aproximado).",
                            [{"title": x["entity"], "detail": f"{x['method']}" + (f" · «{x['matched']}»" if x["matched"] else f" · similitud {x['score']}"),
                              "meta": x} for x in links], _ms(t0)))
        seeds = [x["entity"] for x in links]
        if not seeds:
            return Retrieval([], stages, direction=None)

        chosen, why = decide_direction(question, direction)
        t0 = time.perf_counter()
        walk: Traversal = self.graphs.traverse(source, seeds, chosen, hops, G.IMPACT_RELATIONS)
        edges = list(walk.edges)
        want_teams = needs_teams(question) if teams is None else teams
        team_edges: list[dict] = []
        if want_teams:
            nodes = sorted({*seeds, *(e["subject"] for e in edges), *(e["object"] for e in edges)})
            team_edges = self.graphs.team_facts(source, nodes)
        stages.append(Stage(
            "traverse", f"Recorrer el grafo ({hops} saltos, {'al revés' if chosen == 'reverse' else 'hacia adelante' if chosen == 'forward' else 'ambos sentidos'})",
            f"{why}. {len(edges)} relaciones halladas" + (f" y {len(team_edges)} de equipos" if want_teams else "") + ".",
            [{"title": f"{e['subject']} → {e['object']}", "detail": f"{e['relation']} · salto {e['depth']} desde «{e['seed']}»"} for e in edges[:40]],
            _ms(t0), sql=walk.queries))

        facts: list[Item] = []
        memo: dict = {}
        for e in edges[:max_facts]:
            chunk = self._chunk_for(e["section"], e["evidence"], memo)
            facts.append(Item(id=f"{e['subject']}|{e['relation']}|{e['object']}", kind="fact",
                              text=G.fact_phrase(e["subject"], e["relation"], e["object"]), label=f"salto {e['depth']} · fuente: «{e['section']}»",
                              score=1 / e["depth"],
                              meta={**e, "chunk_id": chunk.id if chunk else None}))
        for e in team_edges:
            chunk = self._chunk_for(e["section"], e["evidence"], memo)
            facts.append(Item(id=f"{e['subject']}|{e['relation']}|{e['object']}", kind="fact",
                              text=G.fact_phrase(e["subject"], e["relation"], e["object"]), label=f"equipo · fuente: «{e['section']}»",
                              score=0.5, meta={**e, "depth": 0, "seed": "", "chunk_id": chunk.id if chunk else None}))
        stages.append(Stage("facts", "Hechos con su procedencia",
                            "Cada hecho es una tripleta del grafo y apunta a la oración del catálogo que la respalda."
                            + (f" Se cortó en {max_facts} hechos." if len(edges) > max_facts else ""),
                            [{"title": f.text, "detail": f.label + (f" · fragmento {f.meta['chunk_id']}" if f.meta.get("chunk_id") else "")} for f in facts], 0.0))

        items = list(facts)
        if with_chunks:
            wanted: dict[str, tuple[float, Item]] = {}
            for f in facts:
                cid = f.meta.get("chunk_id")
                if not cid or cid in wanted:
                    continue
                chunk = self._chunk_for(f.meta["section"], f.meta["evidence"], memo)
                if chunk:
                    wanted[cid] = (f.meta["depth"], chunk_item(chunk, f.score))
            origin = [it for _, it in sorted(wanted.values(), key=lambda x: x[0])][:k]
            items += origin
            stages.append(Stage("origin", "Fragmentos de donde salieron los hechos",
                                f"Los {k} fragmentos más cercanos a la entidad de partida (el texto original, para poder verificarlo).",
                                [{"title": it.label, "detail": it.text} for it in origin], 0.0))

        involved = {*seeds, *(e["subject"] for e in edges), *(e["object"] for e in edges), *(e["subject"] for e in team_edges),
                    *(e["object"] for e in team_edges)}
        subgraph = {"nodes": [{"id": n, "type": data.entities.get(n, {}).get("tipo", "otro"),
                               "role": "seed" if n in seeds else "team" if data.entities.get(n, {}).get("tipo") == "equipo" else "reached"}
                              for n in sorted(involved)],
                    "edges": [{"subject": e["subject"], "relation": e["relation"], "object": e["object"], "depth": e.get("depth", 0)} for e in [*edges, *team_edges]]}
        return Retrieval(items, stages, subgraph=subgraph, sql=walk.queries, direction=chosen)

    # ═════════════════════════ recuperación: los cuatro modos ═════════════════════════
    def retrieve(self, question: str, mode: str = "vector", k: int = 4, hops: int = 4, direction: str = "auto",
                 decompose: bool = False, contextual: bool = False, graph: str = "curated", max_facts: int = 30,
                 link_threshold: float = 0.65, teams: bool | None = None) -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"mode debe ser uno de: {', '.join(MODES)}")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction debe ser una de: {', '.join(DIRECTIONS)}")
        if decompose and mode not in ("vector", "hybrid_graph"):
            raise ValueError("La descomposición se aplica solo a los modos «vector» e «hybrid_graph».")
        if mode in ("vector", "hybrid_graph"):
            self._require_chunks()
            if contextual and not self.store.count_contextual():
                raise NeedsSetup("La ingesta no tiene frases de contexto: ve a la pestaña Ingesta y reingesta con «contextual» activado.")
        if mode == "graph_local":
            return self._facts(question, graph, hops, direction, teams, max_facts, link_threshold, with_chunks=True, k=k)
        if mode == "graph_global":
            return self._global(question, graph)
        if mode == "vector":
            return self._vector_mode(question, k, decompose, contextual)
        return self._hybrid(question, k, hops, direction, decompose, contextual, graph, max_facts, link_threshold, teams)

    def _vector_mode(self, question: str, k: int, decompose: bool, contextual: bool) -> Retrieval:
        stages: list[Stage] = []
        llms: list[LLMResult] = []
        title = "Búsqueda vectorial" + (" (con frase de contexto)" if contextual else "")
        note = "Por significado: similitud coseno entre la pregunta y cada fragmento (operador <=> de pgvector)."
        if not decompose:
            t0 = time.perf_counter()
            items = self._vector(question, k, contextual)
            stages.append(Stage("vector", title, note, [_chunk_stage_item(i) for i in items], _ms(t0)))
            return Retrieval(items, stages)
        t0 = time.perf_counter()
        subs, llm = self._decompose(question)
        llms.append(llm)
        stages.append(Stage("decompose", "Descomposición en subpreguntas",
                            "Un LLM separa la pregunta en partes sencillas; cada una se busca por separado.",
                            [{"title": s, "detail": ""} for s in subs], _ms(t0), llm.view()))
        lists: dict[str, list[Item]] = {}
        t0 = time.perf_counter()
        for name, q in [("original", question), *[(f"sub{i}", s) for i, s in enumerate(subs, 1)]]:
            lists[name] = self._vector(q, k, contextual)
        stages.append(Stage("vector", title + " por subpregunta", note,
                            [{"title": f"{n}: {i.label}", "detail": i.text} for n, ls in lists.items() for i in ls], _ms(t0)))
        t0 = time.perf_counter()
        fused = rrf(lists, k=k)
        stages.append(Stage("rrf", "Fusión RRF", "Suma 1/(60+posición) de cada lista: sube lo que está bien ubicado en varias subpreguntas.",
                            [{"title": i.label, "detail": i.text, "meta": {"ranks": r}} for i, r in fused], _ms(t0)))
        return Retrieval([i for i, _ in fused], stages, llms)

    def _hybrid(self, question: str, k: int, hops: int, direction: str, decompose: bool, contextual: bool, graph: str,
                max_facts: int, threshold: float, teams: bool | None) -> Retrieval:
        stages: list[Stage] = []
        llms: list[LLMResult] = []
        queries = [("original", question)]
        if decompose:
            t0 = time.perf_counter()
            subs, llm = self._decompose(question)
            llms.append(llm)
            stages.append(Stage("decompose", "Descomposición en subpreguntas",
                                "Un LLM separa la pregunta en partes; cada una recupera sus hechos y sus fragmentos.",
                                [{"title": s, "detail": ""} for s in subs], _ms(t0), llm.view()))
            queries += [(f"sub{i}", s) for i, s in enumerate(subs, 1)]
        fact_lists: dict[str, list[Item]] = {}
        chunk_lists: dict[str, list[Item]] = {}
        sub = Retrieval([], [])
        for name, q in queries:
            r = self._facts(q, graph, hops, direction, teams, max_facts, threshold, with_chunks=False, k=k)
            fact_lists[name] = r.items
            if name == "original":
                sub = r
                stages += r.stages
            else:
                sub.subgraph = _merge_subgraphs(sub.subgraph, r.subgraph)
                sub.sql += r.sql
            t0 = time.perf_counter()
            chunk_lists[name] = self._vector(q, k, contextual)
            if name == "original":
                stages.append(Stage("vector", "Búsqueda vectorial" + (" (con frase de contexto)" if contextual else ""),
                                    "Los fragmentos más parecidos a la pregunta, para complementar los hechos.",
                                    [_chunk_stage_item(i) for i in chunk_lists[name]], _ms(t0)))
        if decompose:
            t0 = time.perf_counter()
            fused_facts = [i for i, _ in rrf(fact_lists, k=max_facts)]
            fused_chunks = rrf(chunk_lists, k=k)
            stages.append(Stage("rrf", "Fusión RRF (hechos y fragmentos de todas las subpreguntas)",
                                "Cada subpregunta aporta su lista de hechos y su lista de fragmentos; se fusionan por posición.",
                                [{"title": i.text, "detail": f"hecho · {i.label}"} for i in fused_facts[:12]]
                                + [{"title": i.label, "detail": i.text, "meta": {"ranks": r}} for i, r in fused_chunks], _ms(t0)))
            facts, chunks = fused_facts, [i for i, _ in fused_chunks]
        else:
            facts, chunks = fact_lists["original"], chunk_lists["original"]
        # un solo contexto numerado: primero los hechos (las relaciones), después los fragmentos (el texto)
        items = [*facts, *chunks]
        stages.append(Stage("merge", "Contexto combinado",
                            "Los hechos del grafo y los fragmentos vectoriales en una sola lista numerada; cada elemento lleva su etiqueta de tipo.",
                            [{"title": f"[{n}] {it.text}", "detail": KIND_LABEL[it.kind]} for n, it in enumerate(items, 1)], 0.0))
        return Retrieval(items, stages, llms, sub.subgraph, sub.sql, sub.direction)

    def _global(self, question: str, source: str) -> Retrieval:
        """Versión SIMPLIFICADA de la búsqueda global de Microsoft GraphRAG: MAPA (qué dice cada resumen de comunidad sobre la
        pregunta) y REDUCCIÓN (la respuesta final se redacta a partir de esos extractos)."""
        self._graph(source)
        communities = self.graphs.load_communities(source)
        if not communities or not any(c.summary for c in communities):
            raise NeedsSetup("Las comunidades no tienen resumen: ve a la pestaña Grafo y pulsa «Resumir comunidades» (usa el LLM).")
        t0 = time.perf_counter()
        llms: list[LLMResult] = []
        items: list[Item] = []
        view: list[dict] = []
        for c in communities:
            r = generate(prompts.render("mapa_comunidad", "v1", pregunta=question, resumen=c.summary), max_tokens=200, purpose="mapa")
            llms.append(r)
            partial = " ".join(r.text.split())
            useful = bool(partial) and not partial.upper().startswith("NADA")
            view.append({"title": c.title, "detail": partial if useful else "(no aporta a esta pregunta)", "meta": {"useful": useful}})
            if useful:
                items.append(Item(id=f"comunidad-{c.id}", kind="community", text=partial, label=c.title, score=1.0,
                                  meta={"members": c.members, "summary": c.summary}))
        stages = [Stage("map", "Mapa: qué dice cada comunidad",
                        f"Un LLM lee el resumen de cada una de las {len(communities)} comunidades y extrae lo que sirve para la pregunta (o dice NADA).",
                        view, _ms(t0), {"calls": len(llms), "cost_usd": round(sum(x.cost_usd for x in llms), 6)})]
        return Retrieval(items, stages, llms)

    # ═════════════════════════ pregunta completa ═════════════════════════
    def ask(self, question: str, prompt_version: str = "v1", **options) -> Answer:
        t_start = time.perf_counter()
        r = self.retrieve(question, **options)
        timings = {s.key: s.ms for s in r.stages}
        if not r.items:
            timings["total"] = _ms(t_start)
            return Answer(question, NOT_FOUND, [], set(), set(), True, None, None, r, timings)
        prompt = prompts.render("rag_respuesta", prompt_version, question=question,
                                items=[{"kind": KIND_LABEL[i.kind], "label": i.label, "text": i.text} for i in r.items])
        t0 = time.perf_counter()
        llm = generate(prompt, max_tokens=600, purpose="answer")
        timings["generate"] = _ms(t0)
        text = llm.text.strip()
        cited = cited_numbers(text)
        timings["total"] = _ms(t_start)
        return Answer(question, text, r.items, cited, {n for n in cited if not 1 <= n <= len(r.items)},
                      text.startswith("No encuentro"), prompt, llm, r, timings)

    def status(self) -> dict:
        s = get_settings()
        try:
            chunks, contextual, store_error = self.store.count(), self.store.count_contextual(), None
            graphs = self.graphs.sources()
            communities = {src: sum(bool(c.summary) for c in self.graphs.load_communities(src)) for src in graphs}
        except Exception as exc:  # noqa: BLE001 - el estado debe poder mostrarse aunque el almacén falle
            chunks = contextual = 0
            graphs, communities, store_error = {}, {}, f"{type(exc).__name__}: {exc}"
        return {"llm_provider": s.llm_provider, "llm_model": s.llm_model, "embeddings_provider": s.embeddings_provider,
                "embeddings_model": s.embeddings_model, "store": self.store.name, "chunks": chunks, "contextual_chunks": contextual,
                "graphs": graphs, "summaries": communities, "store_error": store_error}


def _chunk_stage_item(item: Item) -> dict:
    return {"title": f"{item.label} · {item.score:.3f}", "detail": (f"contexto: {item.meta['context']} · " if item.meta.get("context") else "") + item.text}


def _merge_subgraphs(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return a or b
    nodes = {n["id"]: n for n in b["nodes"]}
    nodes.update({n["id"]: n for n in a["nodes"]})
    seen, edges = set(), []
    for e in [*a["edges"], *b["edges"]]:
        key = (e["subject"], e["relation"], e["object"])
        if key not in seen:
            seen.add(key)
            edges.append(e)
    return {"nodes": sorted(nodes.values(), key=lambda n: n["id"]), "edges": edges}
