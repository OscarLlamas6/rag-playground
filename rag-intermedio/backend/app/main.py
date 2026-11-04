"""API HTTP del RAG intermedio (FastAPI).

    GET  /api/status      proveedores, almacén y cuántos fragmentos hay indexados
    GET  /api/documents   los documentos de origen (md, html, docx) ya convertidos en secciones, y los PDF (sin convertir)
    GET  /api/media       audios y videos de MEDIA_DIR (duración, tipo, URL)
    GET  /api/media/{nombre}  el archivo, con soporte de Range (para saltar a un instante); solo nombres de esa carpeta
    GET  /api/pdf/{nombre}    el PDF de la carpeta de documentos (se abre en la página citada con #page=N)
    GET  /api/examples    preguntas de ejemplo con lo que muestra cada una
    POST /api/ingest      lee, fragmenta (con metadatos), vectoriza, guarda en Pinecone y arma el índice BM25
    POST /api/ask         UNA pregunta con la traza de cada etapa: vectorial, BM25, RRF, padre-hijo, reranking
    POST /api/evaluate    compara los modos de búsqueda sobre las preguntas de prueba

Arranque:  uv run uvicorn app.main:app --reload --port 8010     (documentación interactiva en /docs)
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from jinja2 import TemplateNotFound

from . import evaluation, media, pdf
from .config import get_settings
from .embeddings import EmbeddingError
from .extras import MediaError, capabilities
from .ingest import TEXT_EXTENSIONS, read_document
from .locations import format_time
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
@app.exception_handler(MediaError)
async def dependency_error(_request, exc: Exception):
    """Un servicio externo no está disponible o no está configurado (o falta el grupo `media`): 503 con un mensaje que explica qué revisar."""
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
    """Md, html y docx se leen al momento. Los PDF NO se convierten aquí (Docling tarda): salen listados y se leen al ingestar."""
    folder = get_settings().docs_dir
    out = []
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        ext = path.suffix.lower()
        if ext in TEXT_EXTENSIONS:
            d = read_document(path)
            out.append({"source": d.source, "title": d.title, "type": d.type, "warnings": d.warnings,
                        "sections": [{"title": t, "words": len(x.split()), "text": x} for t, x in d.sections]})
        elif ext in pdf.EXTENSIONS:
            out.append({"source": path.name, "title": path.stem.replace("_", " ").capitalize(), "type": "pdf", "sections": [],
                        "size_kb": round(path.stat().st_size / 1024), "url": f"/api/pdf/{path.name}",
                        "warnings": ["Se lee con Docling al ingestar (OCR automático si es un escaneo)."]})
    if not out:
        raise HTTPException(status_code=404, detail=f"No hay documentos en {folder}.")
    return out


@app.get("/api/media")
def media_list() -> list[dict]:
    s = get_settings()
    out = []
    for path in media.list_media(s.media_dir):
        seconds = media.duration_of(path)
        out.append({"name": path.name, "type": media.kind_of(path), "mime": media.mime_of(path), "size_kb": round(path.stat().st_size / 1024),
                    "duration_s": seconds, "duration": format_time(seconds) if seconds is not None else None,
                    "url": f"/api/media/{path.name}"})
    return out


def _serve(folder, resolver, name: str, what: str, mime: str | None = None) -> FileResponse:
    """Sirve un archivo por NOMBRE (nunca por ruta). FileResponse atiende las cabeceras Range (206) y declara Accept-Ranges."""
    try:
        path = resolver(folder, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail=f"No existe {what} '{name}'.")
    return FileResponse(path, media_type=mime or media.mime_of(path), filename=None, headers={"Accept-Ranges": "bytes"})


@app.get("/api/media/{name}")
def media_file(name: str) -> FileResponse:
    return _serve(get_settings().media_dir, media.resolve_media, name, "el medio")


def _resolve_pdf(folder, name: str):
    if not name or "/" in name or "\\" in name or "\x00" in name or name.startswith("."):
        raise ValueError("Nombre de archivo no válido.")
    return next((p for p in folder.iterdir() if p.is_file() and p.name == name and p.suffix.lower() in pdf.EXTENSIONS), None) \
        if folder.is_dir() else None


@app.get("/api/pdf/{name}")
def pdf_file(name: str) -> FileResponse:
    return _serve(get_settings().docs_dir, _resolve_pdf, name, "el PDF", "application/pdf")


@app.get("/api/examples")
def examples() -> list[dict]:
    return [{"id": q["id"], "question": q["pregunta"], "type": q["tipo"], "audience": (q.get("filtro") or {}).get("publico")}
            for q in evaluation.load_questions() if q["tipo"] in evaluation.TYPES]


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        return asdict(service().ingest(req.strategy, req.max_words, req.overlap, req.max_tokens))
    except (ValueError, FileNotFoundError) as exc:
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
