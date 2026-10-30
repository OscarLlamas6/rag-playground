"""API HTTP del RAG simple (FastAPI).

    GET  /api/status      proveedores, almacén y cuántos fragmentos hay indexados
    GET  /api/documents   los documentos de origen y sus secciones
    GET  /api/examples    preguntas de ejemplo (las del set de evaluación)
    POST /api/ingest      fragmenta, vectoriza y guarda los documentos
    POST /api/ask         UNA pregunta, UNA respuesta con citas (sin historial)
    POST /api/evaluate    mide la recuperación (gratis) y, si se pide, la generación

Arranque:  uv run uvicorn app.main:app --reload --port 8000     (documentación interactiva en /docs)
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from jinja2 import TemplateNotFound

from . import evaluation
from .config import get_settings
from .documents import load_documents
from .embeddings import EmbeddingError
from .llm import LLMError
from .rag import IngestBusy, RagService
from .schemas import AskRequest, AskResponse, EvaluateRequest, HitOut, IngestRequest, UsageOut
from .store import StoreError

app = FastAPI(title="RAG simple", version="0.1.0", description="Ingesta, recuperación y generación con citas. One-shot, sin historial.")
_service: RagService | None = None


def service() -> RagService:
    """Un único servicio por proceso (mantiene la conexión al almacén)."""
    global _service
    if _service is None:
        _service = RagService()
    return _service


@app.exception_handler(EmbeddingError)
@app.exception_handler(LLMError)
@app.exception_handler(StoreError)
async def dependency_error(_request, exc: Exception):
    """Un servicio externo no está disponible: 503 con un mensaje que explica qué revisar."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict:
    """Responde sin tocar ningún servicio externo: sirve para los healthchecks de Docker."""
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    return service().status()


@app.get("/api/documents")
def documents() -> list[dict]:
    return [
        {"source": d.source, "title": d.title,
         "sections": [{"title": t, "words": len(x.split()), "text": x} for t, x in d.sections]}
        for d in load_documents(get_settings().docs_dir)
    ]


@app.get("/api/examples")
def examples() -> list[dict]:
    return [{"id": q["id"], "question": q["pregunta"], "out_of_scope": bool(q.get("fuera_de_alcance"))}
            for q in evaluation.load_questions()]


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        return asdict(service().ingest(req.strategy, req.max_words, req.overlap))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IngestBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    svc = service()
    if svc.status()["chunks"] == 0:
        raise HTTPException(status_code=409, detail="El almacén está vacío: ejecuta primero la ingesta.")
    try:
        a = svc.ask(req.question, k=req.k, threshold=req.threshold, prompt_version=req.prompt_version)
    except TemplateNotFound as exc:
        raise HTTPException(status_code=422, detail=f"No existe la versión de prompt '{req.prompt_version}'.") from exc
    return AskResponse(
        question=a.question, answer=a.text, abstained=a.abstained, uncited=a.uncited,
        invalid_citations=sorted(a.invalid),
        hits=[HitOut(number=i, id=h.id, text=h.text, source=h.metadata.get("source", "?"),
                     section=h.metadata.get("section", "?"), score=round(h.score, 4), cited=i in a.cited)
              for i, h in enumerate(a.hits, start=1)],
        usage=UsageOut(provider=a.llm.provider, model=a.llm.model, tokens_in=a.llm.tokens_in, tokens_out=a.llm.tokens_out,
                       cost_usd=a.llm.cost_usd, seconds=round(a.llm.seconds, 3)) if a.llm else None,
        timings_ms={k: round(v, 1) for k, v in a.timings_ms.items()},
        skipped_reason=a.skipped_reason, prompt=a.prompt,
    )


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    svc = service()
    if svc.status()["chunks"] == 0:
        raise HTTPException(status_code=409, detail="El almacén está vacío: ejecuta primero la ingesta.")
    return evaluation.evaluate(svc, k=req.k, with_generation=req.with_generation)
