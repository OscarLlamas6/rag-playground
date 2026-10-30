"""El flujo del RAG intermedio, con una TRAZA de cada etapa.

    pregunta
       │
       ├─► búsqueda vectorial (significado) ─┐
       │                                     ├─► fusión RRF ─► [padre-hijo: hijo → sección] ─► [reranking] ─► top-k
       └─► búsqueda BM25 (palabras exactas) ─┘        ▲
                       └── ambas con FILTRO por metadatos (audiencia) ──┘
                                                                          │
                                       prompt con citas ─► LLM ─► verificar citas

Cada etapa devuelve lo que vio y cuánto tardó, para que la interfaz pueda MOSTRAR qué hizo cada pieza. Sigue siendo
one-shot: cada pregunta es independiente (sin historial; la reescritura de consultas de seguimiento queda para cuando
haya conversación).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from . import prompts
from .cache import load_chunks, save_chunks
from .chunking import chunk_document, text_to_embed
from .config import get_settings
from .domain import Chunk, Hit
from .embeddings import embed, embed_query
from .fusion import rrf
from .ingest import load_documents
from .lexical import BM25
from .llm import NOT_FOUND, LLMResult, generate
from .store import StoreError, VectorStore, create_store

MODES = ("vector", "bm25", "hybrid")


def cited_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        numbers.update(int(n) for n in re.split(r"\s*,\s*", group))
    return numbers


def audience_filter(audience: str | None) -> list[str] | None:
    """'negocio' → fragmentos de Negocio Y de uso general. None → sin filtro."""
    return None if audience in (None, "", "all") else [audience, "general"]


def expand_to_parents(hits: list[Hit]) -> list[Hit]:
    """Cambia cada hijo por su PADRE (la sección entera), sin repetir padres y respetando el orden."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        pid = h.metadata.get("parent_id", h.id)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(Hit(pid, h.metadata.get("parent_text", h.text), h.metadata, h.score))
    return out


def hit_view(hit: Hit, **extra) -> dict:
    return {"id": hit.id, "source": hit.metadata.get("source", "?"), "section": hit.metadata.get("section", "?"),
            "audience": hit.metadata.get("audience", "?"), "type": hit.metadata.get("type", "?"),
            "score": round(hit.score, 4), "text": hit.text, **extra}


@dataclass
class Stage:
    key: str
    title: str
    note: str
    hits: list[dict]
    ms: float
    llm: dict | None = None


def _llm_view(r: LLMResult | None) -> dict | None:
    return None if r is None else {"provider": r.provider, "model": r.model, "tokens_in": r.tokens_in,
                                   "tokens_out": r.tokens_out, "cost_usd": r.cost_usd, "seconds": round(r.seconds, 3)}


@dataclass
class IngestReport:
    documents: int
    chunks: int
    dimension: int
    strategy: str
    store: str
    seconds: float
    per_document: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    hits: list[Hit]                 # lo que recibe el modelo
    stages: list[Stage]
    rerank_llm: LLMResult | None = None


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: set[int]
    invalid: set[int]
    abstained: bool
    prompt: str | None
    llm: LLMResult | None
    rerank_llm: LLMResult | None
    stages: list[Stage]
    timings_ms: dict[str, float]

    @property
    def uncited(self) -> bool:
        return not self.abstained and not self.cited

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in (self.llm, self.rerank_llm) if r)


def rerank_with_llm(question: str, hits: list[Hit], max_words: int = 90) -> tuple[list[Hit], LLMResult]:
    """Reordena los candidatos con un modelo que lee la pregunta Y cada fragmento JUNTOS.

    Es el reranking "casero" con un LLM general. En producción se usa un cross-encoder especializado (bge-reranker) o una
    API (Cohere, Voyage, Jina): más rápido, barato y fiable. Si el modelo responde algo ilegible, se conserva el orden.
    """
    short = [Hit(h.id, " ".join(h.text.split()[:max_words]), h.metadata, h.score) for h in hits]
    result = generate(prompts.render("rerank", "v1", question=question, hits=short), max_tokens=60)
    order: list[int] = []
    for n in re.findall(r"\d+", result.text):
        i = int(n) - 1
        if 0 <= i < len(hits) and i not in order:
            order.append(i)
    order += [i for i in range(len(hits)) if i not in order]      # los que el modelo olvidó, al final
    return [hits[i] for i in order], result


class IngestBusy(RuntimeError):
    """Ya hay una ingesta en curso: reconstruye el índice completo y no se puede hacer dos veces a la vez."""


class RagService:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store: VectorStore = store or create_store()
        self._ingest_lock = threading.Lock()
        self._bm25: BM25 | None = None
        self._chunks: list[Chunk] = []

    # ───────────── índice léxico ─────────────
    def _build_bm25(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25(chunks, text_to_embed) if chunks else None

    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._build_bm25(load_chunks(get_settings().cache_file))     # tras reiniciar el servidor
        if self._bm25 is None:
            raise StoreError("El índice léxico está vacío: ejecuta primero la ingesta.")
        return self._bm25

    # ───────────── ingesta ─────────────
    def ingest(self, strategy: str = "parent_child", max_words: int = 35, overlap: int = 1) -> IngestReport:
        if not self._ingest_lock.acquire(blocking=False):
            raise IngestBusy("Ya hay una ingesta en curso: espera a que termine.")
        try:
            return self._ingest(strategy, max_words, overlap)
        finally:
            self._ingest_lock.release()

    def _ingest(self, strategy: str, max_words: int, overlap: int) -> IngestReport:
        t0 = time.perf_counter()
        settings = get_settings()
        documents = load_documents(settings.docs_dir)
        chunks: list[Chunk] = []
        per_document: dict[str, dict] = {}
        warnings: list[str] = []
        for doc in documents:
            pieces = chunk_document(doc, strategy=strategy, max_words=max_words, overlap=overlap)
            per_document[doc.source] = {"type": doc.type, "sections": len(doc.sections), "chunks": len(pieces)}
            warnings += [f"{doc.source}: {w}" for w in doc.warnings]
            chunks.extend(pieces)
        vectors = embed([text_to_embed(c) for c in chunks], kind="document")
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        self.store.reset(len(vectors[0]))
        self.store.upsert(chunks)
        save_chunks(settings.cache_file, chunks)
        self._build_bm25(chunks)
        return IngestReport(len(documents), len(chunks), len(vectors[0]), strategy, self.store.name,
                            time.perf_counter() - t0, per_document, warnings)

    # ───────────── recuperación ─────────────
    def search(self, question: str, mode: str = "hybrid", k: int = 3, candidates: int = 8, audience: str | None = None,
               expand_parent: bool | None = None, rerank: bool = False) -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"mode debe ser uno de: {', '.join(MODES)}")
        audiences = audience_filter(audience)
        stages: list[Stage] = []
        lists: dict[str, list[Hit]] = {}

        if mode in ("vector", "hybrid"):
            t0 = time.perf_counter()
            hits = self.store.search(embed_query(question), candidates, audiences)
            lists["vector"] = hits
            stages.append(Stage("vector", "Búsqueda vectorial", "Por significado: similitud coseno entre la pregunta y cada fragmento.",
                                [hit_view(h) for h in hits], (time.perf_counter() - t0) * 1000))
        if mode in ("bm25", "hybrid"):
            t0 = time.perf_counter()
            accept = (lambda meta: meta.get("audience") in audiences) if audiences else None
            hits = self.bm25().search(question, candidates, accept)
            lists["bm25"] = hits
            stages.append(Stage("bm25", "Búsqueda BM25", "Por palabras exactas: gana el fragmento con las palabras más raras de la pregunta.",
                                [hit_view(h) for h in hits], (time.perf_counter() - t0) * 1000))

        if mode == "hybrid":
            t0 = time.perf_counter()
            fused = rrf(lists, k=candidates)
            ranked = [h for h, _ in fused]
            stages.append(Stage("rrf", "Fusión RRF", "Suma 1/(60+posición) de cada lista: sube el que está bien ubicado en ambas.",
                                [hit_view(h, ranks=r) for h, r in fused], (time.perf_counter() - t0) * 1000))
        else:
            ranked = lists[mode]

        if expand_parent is None:
            expand_parent = any("parent_id" in h.metadata for h in ranked)
        if expand_parent and any("parent_id" in h.metadata for h in ranked):
            ranked = expand_to_parents(ranked)
            stages.append(Stage("parent", "Padre-hijo", "Se buscó con fragmentos pequeños; se entrega la sección completa (sin repetir).",
                                [hit_view(h) for h in ranked], 0.0))

        rerank_llm = None
        if rerank and len(ranked) > 1:
            t0 = time.perf_counter()
            ranked, rerank_llm = rerank_with_llm(question, ranked[:candidates])
            stages.append(Stage("rerank", "Reranking con un LLM", "Un modelo lee la pregunta y cada candidato juntos y los reordena.",
                                [hit_view(h) for h in ranked], (time.perf_counter() - t0) * 1000, _llm_view(rerank_llm)))
        return Retrieval(ranked[:k], stages, rerank_llm)

    # ───────────── pregunta completa ─────────────
    def ask(self, question: str, mode: str = "hybrid", k: int = 3, candidates: int = 8, audience: str | None = None,
            expand_parent: bool | None = None, rerank: bool = False, prompt_version: str = "v1") -> Answer:
        t_start = time.perf_counter()
        r = self.search(question, mode, k, candidates, audience, expand_parent, rerank)
        timings = {s.key: s.ms for s in r.stages}
        if not r.hits:
            timings["total"] = (time.perf_counter() - t_start) * 1000
            return Answer(question, NOT_FOUND, [], set(), set(), True, None, None, r.rerank_llm, r.stages, timings)
        prompt = prompts.render("rag_respuesta", prompt_version, hits=r.hits, question=question)
        t0 = time.perf_counter()
        llm = generate(prompt)
        timings["generate"] = (time.perf_counter() - t0) * 1000
        text = llm.text.strip()
        cited = cited_numbers(text)
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return Answer(question, text, r.hits, cited, {n for n in cited if not 1 <= n <= len(r.hits)},
                      text.startswith("No encuentro"), prompt, llm, r.rerank_llm, r.stages, timings)

    def status(self) -> dict:
        s = get_settings()
        try:
            chunks, store_error = self.store.count(), None
        except Exception as exc:  # noqa: BLE001 - el estado debe poder mostrarse aunque el almacén falle
            chunks, store_error = 0, f"{type(exc).__name__}: {exc}"
        return {"llm_provider": s.llm_provider, "llm_model": s.llm_model,
                "embeddings_provider": s.embeddings_provider, "embeddings_model": s.embeddings_model,
                "store": self.store.name, "chunks": chunks, "store_error": store_error,
                "index": s.pinecone_index if self.store.name == "pinecone" else None,
                "namespace": s.pinecone_namespace if self.store.name == "pinecone" else None}
