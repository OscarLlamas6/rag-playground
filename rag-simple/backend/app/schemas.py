"""Contratos de la API (lo que entra y lo que sale). FastAPI genera la documentación en /docs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .documents import STRATEGIES


class IngestRequest(BaseModel):
    strategy: str = Field("sentences", description=f"Estrategia de fragmentado: {', '.join(STRATEGIES)}")
    max_words: int = Field(70, ge=10, le=400)
    overlap: int = Field(1, ge=0, le=5)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=10, description="Cuántos fragmentos recibe el modelo")
    threshold: float | None = Field(None, ge=0, le=1, description="Similitud mínima del mejor fragmento para llamar al modelo")
    prompt_version: str = "v1"


class EvaluateRequest(BaseModel):
    k: int = Field(4, ge=1, le=10)
    with_generation: bool = Field(False, description="Incluye la capa de generación (llama al modelo: cuesta)")


class HitOut(BaseModel):
    number: int                 # el número con el que el prompt lo numera: [1], [2]…
    id: str
    text: str
    source: str
    section: str
    score: float
    cited: bool                 # ¿el modelo lo citó?


class UsageOut(BaseModel):
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    seconds: float


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    uncited: bool
    invalid_citations: list[int]
    hits: list[HitOut]
    usage: UsageOut | None
    timings_ms: dict[str, float]
    skipped_reason: str | None
    prompt: str | None
