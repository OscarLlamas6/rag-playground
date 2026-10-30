"""API HTTP del RAG intermedio (FastAPI).

    GET  /api/status      proveedores, almacén y cuántos fragmentos hay indexados
    GET  /api/documents   los documentos de origen (md, html, docx) ya convertidos en secciones
    GET  /api/examples    preguntas de ejemplo con lo que muestra cada una
    POST /api/ingest      lee, fragmenta (con metadatos), vectoriza, guarda en Pinecone y arma el índice BM25
    POST /api/ask         UNA pregunta con la traza de cada etapa: vectorial, BM25, RRF, padre-hijo, reranking
    POST /api/evaluate    compara los modos de búsqueda sobre las preguntas de prueba

Arranque:  uv run uvicorn app.main:app --reload --port 8010     (documentación interactiva en /docs)
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
from .rag import IngestBusy, RagService, hit_view
from .schemas import AskRequest, EvaluateRequest, IngestRequest
from .store import StoreError

app = FastAPI(title="RAG intermedio", version="0.1.0",
              description="Ingesta multiformato, padre-hijo, BM25 + vectores con RRF, filtros y reranking. One-shot, sin historial.")
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


def _require_index(svc: RagService) -> None:
    """409 solo si el almacén está de verdad vacío. Si no se puede consultar, el error se propaga (503 con su mensaje):
    antes se confundía "no puedo contar" con "no hay nada" y se mandaba a ingestar de nuevo sin necesidad."""
    try:
        chunks = svc.store.count()
    except StoreError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StoreError(f"No se pudo consultar el almacén ({type(exc).__name__}): {exc}") from exc
    if chunks == 0:
        raise HTTPException(status_code=409, detail="El almacén está vacío: ejecuta primero la ingesta.")


@app.get("/api/health")
def health() -> dict:
    """Responde sin tocar ningún servicio externo: sirve para los healthchecks de Docker."""
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    return service().status()


@app.get("/api/documents")
def documents() -> list[dict]:
    return [{"source": d.source, "title": d.title, "type": d.type, "warnings": d.warnings,
             "sections": [{"title": t, "words": len(x.split()), "text": x} for t, x in d.sections]}
            for d in load_documents(get_settings().docs_dir)]


@app.get("/api/examples")
def examples() -> list[dict]:
    return [{"id": q["id"], "question": q["pregunta"], "type": q["tipo"], "audience": (q.get("filtro") or {}).get("publico")}
            for q in evaluation.load_questions() if q["tipo"] in evaluation.TYPES]


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        return asdict(service().ingest(req.strategy, req.max_words, req.overlap))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IngestBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    svc = service()
    _require_index(svc)
    try:
        a = svc.ask(req.question, mode=req.mode, k=req.k, candidates=req.candidates, audience=req.audience,
                    expand_parent=req.expand_parent, rerank=req.rerank, prompt_version=req.prompt_version)
    except TemplateNotFound as exc:
        raise HTTPException(status_code=422, detail=f"No existe la versión de prompt '{req.prompt_version}'.") from exc
    usage = None
    if a.llm:
        usage = {"provider": a.llm.provider, "model": a.llm.model, "tokens_in": a.llm.tokens_in, "tokens_out": a.llm.tokens_out,
                 "seconds": round(a.llm.seconds, 3)}
    return {
        "question": a.question, "answer": a.text, "abstained": a.abstained, "uncited": a.uncited,
        "invalid_citations": sorted(a.invalid),
        "hits": [{**hit_view(h), "number": i, "cited": i in a.cited} for i, h in enumerate(a.hits, start=1)],
        "stages": [asdict(s) for s in a.stages],
        "usage": usage, "cost_usd": round(a.cost_usd, 6),
        "timings_ms": {k: round(v, 1) for k, v in a.timings_ms.items()},
        "prompt": a.prompt,
    }


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    svc = service()
    _require_index(svc)
    return evaluation.evaluate_modes(svc, k=req.k, candidates=req.candidates, with_rerank=req.with_rerank)
