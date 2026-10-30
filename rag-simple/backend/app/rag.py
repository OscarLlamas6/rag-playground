"""El flujo del RAG simple, de punta a punta.

    INGESTA (una vez)                                   CONSULTA (cada pregunta, una sola vuelta)
    documentos → fragmentar → vectorizar → guardar      pregunta → vectorizar → buscar k fragmentos
                                                                      ↓
                                                        prompt = reglas + fragmentos numerados + pregunta
                                                                      ↓
                                                        LLM → respuesta con citas [1] [2] → verificar citas

Es un flujo FIJO: no decide nada, no recuerda la conversación, no usa herramientas. Cada pregunta es independiente
(one-shot). Los agentes y la memoria son otro tema.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from . import prompts
from .config import get_settings
from .documents import chunk_document, load_documents, text_to_embed
from .domain import Chunk, Hit
from .embeddings import embed, embed_query
from .llm import NOT_FOUND, LLMResult, generate
from .store import VectorStore, create_store


def cited_numbers(text: str) -> set[int]:
    """Números citados en el texto: [1], [2] o [1, 3]."""
    numbers: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        numbers.update(int(n) for n in re.split(r"\s*,\s*", group))
    return numbers


@dataclass
class IngestReport:
    documents: int
    chunks: int
    dimension: int
    strategy: str
    max_words: int
    overlap: int
    store: str
    seconds: float
    per_document: dict[str, int] = field(default_factory=dict)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: set[int]                       # citas que el modelo escribió
    invalid: set[int]                     # citas a fragmentos que NO existen (posible invención)
    abstained: bool                       # dijo "no encuentro"
    prompt: str | None
    llm: LLMResult | None                 # None si no se llamó al modelo (nada superó el umbral)
    timings_ms: dict[str, float]
    skipped_reason: str | None = None

    @property
    def uncited(self) -> bool:
        """Respondió con datos pero sin citar ningún fragmento: no se puede verificar."""
        return not self.abstained and not self.cited


class IngestBusy(RuntimeError):
    """Ya hay una ingesta en curso: reconstruye el índice completo y no se puede hacer dos veces a la vez."""


class RagService:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store: VectorStore = store or create_store()
        self._ingest_lock = threading.Lock()

    # ───────────── ingesta ─────────────
    def ingest(self, strategy: str = "sentences", max_words: int = 70, overlap: int = 1) -> IngestReport:
        if not self._ingest_lock.acquire(blocking=False):
            raise IngestBusy("Ya hay una ingesta en curso: espera a que termine.")
        try:
            return self._ingest(strategy, max_words, overlap)
        finally:
            self._ingest_lock.release()

    def _ingest(self, strategy: str, max_words: int, overlap: int) -> IngestReport:
        """Lee los documentos, los fragmenta, los vectoriza y los guarda. Reconstruye el índice completo."""
        t0 = time.perf_counter()
        settings = get_settings()
        documents = load_documents(settings.docs_dir)
        chunks: list[Chunk] = []
        per_document: dict[str, int] = {}
        for doc in documents:
            pieces = chunk_document(doc, strategy=strategy, max_words=max_words, overlap=overlap)
            per_document[doc.source] = len(pieces)
            chunks.extend(pieces)
        vectors = embed([text_to_embed(c) for c in chunks], kind="document")
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        self.store.reset(len(vectors[0]))
        self.store.upsert(chunks)
        return IngestReport(len(documents), len(chunks), len(vectors[0]), strategy, max_words, overlap,
                            self.store.name, time.perf_counter() - t0, per_document)

    # ───────────── consulta ─────────────
    def retrieve(self, question: str, k: int = 4) -> tuple[list[Hit], dict[str, float]]:
        t0 = time.perf_counter()
        vector = embed_query(question)
        t1 = time.perf_counter()
        hits = self.store.search(vector, k=k)
        t2 = time.perf_counter()
        return hits, {"embed_query": (t1 - t0) * 1000, "search": (t2 - t1) * 1000}

    def ask(self, question: str, k: int = 4, threshold: float | None = None, prompt_version: str = "v1",
            include_prompt: bool = True) -> Answer:
        """Una pregunta, una respuesta: recuperar → (¿vale la pena?) → generar → verificar citas.

        `threshold`: si el mejor fragmento tiene menos similitud, NI SIQUIERA se llama al modelo y se responde
        "no encuentro". Ahorra dinero, pero un umbral descarta lo ajeno, no lo que se parece y no está: para
        eso sirve el prompt. Calíbralo mirando los puntajes de tus propias preguntas.
        """
        t_start = time.perf_counter()
        hits, timings = self.retrieve(question, k)
        if not hits or (threshold is not None and hits[0].score < threshold):
            timings["total"] = (time.perf_counter() - t_start) * 1000
            reason = "No hay fragmentos indexados." if not hits else f"El mejor fragmento ({hits[0].score:.2f}) no supera el umbral ({threshold:.2f})."
            return Answer(question, NOT_FOUND, hits, set(), set(), True, None, None, timings, skipped_reason=reason)

        prompt = prompts.render("rag_respuesta", prompt_version, hits=hits, question=question)
        t0 = time.perf_counter()
        llm = generate(prompt)
        timings["generate"] = (time.perf_counter() - t0) * 1000
        text = llm.text.strip()
        cited = cited_numbers(text)
        invalid = {n for n in cited if not 1 <= n <= len(hits)}
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return Answer(question, text, hits, cited, invalid, text.startswith("No encuentro"),
                      prompt if include_prompt else None, llm, timings)

    def status(self) -> dict:
        s = get_settings()
        try:
            chunks = self.store.count()
            store_error = None
        except Exception as exc:  # noqa: BLE001 - el estado debe poder mostrarse aunque el almacén falle
            chunks, store_error = 0, f"{type(exc).__name__}: {exc}"
        return {
            "llm_provider": s.llm_provider, "llm_model": s.llm_model,
            "embeddings_provider": s.embeddings_provider, "embeddings_model": s.embeddings_model,
            "store": self.store.name, "chunks": chunks, "store_error": store_error,
            "prompt_versions": prompts.versions("rag_respuesta"),
        }
