"""Contratos de la API (lo que entra y lo que sale). FastAPI genera la documentación en /docs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .chunking import STRATEGIES


class IngestRequest(BaseModel):
    strategy: str = Field("parent_child", description=f"Estrategia de fragmentado: {', '.join(STRATEGIES)}")
    max_words: int = Field(35, ge=10, le=400, description="Palabras máximas por fragmento (por hijo, en padre-hijo)")
    overlap: int = Field(1, ge=0, le=5)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    k: int = Field(3, ge=1, le=10, description="Cuántos fragmentos recibe el modelo")
    candidates: int = Field(8, ge=2, le=30, description="Candidatos que aporta cada buscador (y que reordena el reranking)")
    audience: Literal["residencial", "negocio"] | None = Field(None, description="Filtro por metadatos: audiencia (+ lo de uso general)")
    rerank: bool = Field(False, description="Reordenar los candidatos con un LLM (suma una llamada)")
    expand_parent: bool | None = Field(None, description="Entregar el padre en vez del hijo; por defecto, si el índice es padre-hijo")
    prompt_version: str = "v1"


class EvaluateRequest(BaseModel):
    k: int = Field(3, ge=1, le=10)
    candidates: int = Field(8, ge=2, le=30)
    with_rerank: bool = Field(False, description="Incluye 'híbrida + reranking' (una llamada al modelo por pregunta: cuesta)")
