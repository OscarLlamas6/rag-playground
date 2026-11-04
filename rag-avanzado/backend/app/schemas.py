"""Contratos de la API (lo que entra y lo que sale). FastAPI genera la documentación en /docs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .ingest import STRATEGIES

Source = Literal["curated", "extracted"]


class IngestRequest(BaseModel):
    strategy: str = Field("sentences", description=f"Estrategia de fragmentado: {', '.join(STRATEGIES)}")
    max_words: int = Field(25, ge=10, le=400, description="Palabras máximas por fragmento")
    overlap: int = Field(0, ge=0, le=5)
    heading: bool = Field(True, description="Anteponer el título de la sección al texto antes de vectorizar (encabezado contextual barato)")
    contextual: bool = Field(False, description="Contextual Retrieval: un LLM escribe una frase de contexto por fragmento (1 llamada cada uno; se cachea)")
    force_context: bool = Field(False, description="Ignorar la caché de frases de contexto")


class GraphBuildRequest(BaseModel):
    source: Source = "curated"
    prompt_version: str = Field("v1", description="Versión del prompt de extracción (solo para 'extracted')")
    force: bool = Field(False, description="Ignorar la caché de extracción")


class SummarizeRequest(BaseModel):
    source: Source = "curated"
    force: bool = False


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: Literal["vector", "graph_local", "graph_global", "hybrid_graph"] = "hybrid_graph"
    k: int = Field(4, ge=1, le=12, description="Fragmentos que recibe el modelo")
    hops: int = Field(4, ge=1, le=6, description="Saltos máximos del recorrido del grafo")
    direction: Literal["auto", "reverse", "forward", "both"] = Field(
        "auto", description="Sentido del recorrido. auto = reglas fijas por palabras de la pregunta (no es un agente)")
    decompose: bool = Field(False, description="Descomponer la pregunta en subpreguntas (solo vector e hybrid_graph)")
    contextual: bool = Field(False, description="Buscar con los vectores que incluyen la frase de contexto (requiere ingesta contextual)")
    graph: Source = "curated"
    max_facts: int = Field(30, ge=1, le=100)
    link_threshold: float = Field(0.65, ge=0.0, le=1.0, description="Similitud mínima para vincular una entidad por embedding")
    teams: bool | None = Field(None, description="Incluir los equipos que administran los servicios del recorrido; por defecto, si la pregunta habla de equipos")
    prompt_version: str = "v1"


class EvaluateRequest(BaseModel):
    configs: list[str] | None = Field(None, description="Nombres de las configuraciones; por defecto, las gratuitas (o todas si generate=true)")
    generate: bool = Field(False, description="Genera la respuesta con el LLM y mide su recall (cuesta una llamada por pregunta y configuración)")
    questions: list[str] | None = Field(None, description="Identificadores de preguntas (por defecto, todas)")
