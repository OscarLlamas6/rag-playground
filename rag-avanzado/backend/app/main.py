"""API HTTP del RAG avanzado (FastAPI).

    GET  /api/health            responde sin tocar nada externo (healthcheck de Docker)
    GET  /api/status            proveedores, almacén, fragmentos indexados, grafos construidos
    GET  /api/documents         el catálogo ya leído en secciones
    GET  /api/examples          las preguntas de ejemplo (las de la evaluación)
    POST /api/ingest            fragmenta, (opcional) escribe la frase de contexto, vectoriza y guarda en PostgreSQL
    POST /api/graph/build       construye el grafo: 'curated' (gratis) o 'extracted' (LLM, medido contra el curado)
    POST /api/graph/summarize   resume cada comunidad con un LLM (necesario para el modo graph_global)
    GET  /api/graph             el grafo completo, estadísticas y comunidades
    POST /api/ask               UNA pregunta con la traza de cada etapa, el subgrafo y el SQL del recorrido
    POST /api/evaluate          compara configuraciones sobre las preguntas multi-salto

Arranque:  uv run uvicorn app.main:app --reload --port 8020     (documentación interactiva en /docs)
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from jinja2 import TemplateNotFound

from . import evaluation
from .config import get_settings
from .embeddings import EmbeddingError
from .ingest import load_documents
from .llm import LLMError
from .pg import StoreError
from .rag import IngestBusy, NeedsSetup, RagService, item_view
from .schemas import AskRequest, EvaluateRequest, GraphBuildRequest, IngestRequest, SummarizeRequest

app = FastAPI(title="RAG avanzado", version="0.1.0",
              description="GraphRAG con PostgreSQL + pgvector: grafo con CTE recursivas, comunidades, descomposición y recuperación contextual. One-shot, sin historial, sin agentes.")
_service: RagService | None = None


def service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


@app.exception_handler(EmbeddingError)
@app.exception_handler(LLMError)
@app.exception_handler(StoreError)
async def dependency_error(_request, exc: Exception):
    """Un servicio externo no está disponible o no está configurado: 503 con un mensaje que explica qué revisar."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(NeedsSetup)
@app.exception_handler(IngestBusy)
async def setup_error(_request, exc: Exception):
    """Falta un paso previo (ingestar, construir el grafo, resumir comunidades) o hay otra operación en curso: 409."""
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error(_request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict:
    """Responde sin tocar ningún servicio externo: sirve para los healthchecks de Docker."""
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    return service().status()


@app.get("/api/documents")
def documents() -> list[dict]:
    return [{"source": d.source, "title": d.title,
             "sections": [{"title": t, "words": len(x.split()), "text": x} for t, x in d.sections]}
            for d in load_documents(get_settings().docs_dir)]


@app.get("/api/examples")
def examples() -> list[dict]:
    return [{"id": q["id"], "question": q["pregunta"], "type": q["tipo"], "origin": q["origen"], "hops": q["saltos"]}
            for q in evaluation.load_questions()]


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    return asdict(service().ingest(req.strategy, req.max_words, req.overlap, req.contextual, req.heading, req.force_context))


@app.post("/api/graph/build")
def graph_build(req: GraphBuildRequest) -> dict:
    try:
        return service().build_graph(req.source, req.prompt_version, req.force)
    except TemplateNotFound as exc:
        raise HTTPException(status_code=422, detail=f"No existe la versión de prompt '{req.prompt_version}'.") from exc


@app.post("/api/graph/summarize")
def graph_summarize(req: SummarizeRequest) -> dict:
    return service().summarize_communities(req.source, req.force)


@app.get("/api/graph")
def graph_view(source: str = "curated") -> dict:
    if source not in ("curated", "extracted"):
        raise HTTPException(status_code=422, detail="source debe ser 'curated' o 'extracted'.")
    return service().graph_view(source)


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        a = service().ask(req.question, prompt_version=req.prompt_version, mode=req.mode, k=req.k, hops=req.hops, direction=req.direction,
                          decompose=req.decompose, contextual=req.contextual, graph=req.graph, max_facts=req.max_facts,
                          link_threshold=req.link_threshold, teams=req.teams)
    except TemplateNotFound as exc:
        raise HTTPException(status_code=422, detail=f"No existe la versión de prompt '{req.prompt_version}'.") from exc
    r = a.retrieval
    cited_kinds = sorted({a.items[n - 1].kind for n in a.cited if 1 <= n <= len(a.items)})
    return {
        "question": a.question, "answer": a.text, "mode": req.mode, "graph": req.graph, "abstained": a.abstained, "uncited": a.uncited,
        "invalid_citations": sorted(a.invalid), "cited_kinds": cited_kinds,
        "items": [item_view(it, number=i, cited=i in a.cited) for i, it in enumerate(a.items, start=1)],
        "stages": [asdict(s) for s in r.stages], "subgraph": r.subgraph, "sql": r.sql, "direction": r.direction,
        "usage": a.llm.view() if a.llm else None, "cost_usd": round(a.cost_usd, 6),
        "timings_ms": {k: round(v, 1) for k, v in a.timings_ms.items()}, "prompt": a.prompt,
    }


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    return evaluation.evaluate(service(), req.configs, req.generate, req.questions)


@app.get("/api/evaluate/configs")
def evaluation_configs() -> list[dict]:
    return [{"name": c.name, "label": c.label, "mode": c.mode, "needs_llm_in_retrieval": c.llm_in_retrieval} for c in evaluation.CONFIGS]



