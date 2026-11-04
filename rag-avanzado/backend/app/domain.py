"""Los objetos que viajan por todo el RAG avanzado.

    Document / Chunk   el catálogo ya leído y partido en fragmentos (con su vector y, si se pidió, su frase de contexto)
    Triple             un hecho del grafo (sujeto, relación, objeto) con su PROCEDENCIA: sección y oración de donde salió
    GraphData          las entidades y tripletas de un grafo (curado o extraído)
    Item               un elemento NUMERADO del contexto que recibe el modelo: un fragmento, un hecho o una comunidad
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    source: str
    title: str
    sections: list[tuple[str, str]]


@dataclass
class Chunk:
    """Un fragmento. `context` es la frase que añade el LLM en la técnica contextual (vacía si no se usó).

    `vector` se calcula con el texto normal; `vector_ctx`, con la frase de contexto delante (solo si hubo contexto).
    El texto que lee el modelo al responder es SIEMPRE `text`: el contexto sirve solo para encontrarlo.
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    context: str = ""
    vector: list[float] | None = None
    vector_ctx: list[float] | None = None


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    section: str = ""      # sección del catálogo de donde salió
    evidence: str = ""     # la oración exacta que lo dice

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class GraphData:
    """Un grafo: `entities` = nombre → {"tipo": ..., "alias": [...]}, `triples` con procedencia."""

    entities: dict[str, dict]
    triples: list[Triple]
    source: str = "curated"


@dataclass
class Item:
    """Un elemento del contexto. `kind` es `chunk` (fragmento de texto), `fact` (tripleta del grafo) o `community`
    (resumen de una comunidad). Los tres se numeran [1], [2]… en el prompt y se pueden citar."""

    id: str
    kind: str
    text: str
    label: str = ""                              # de dónde viene (sección, relación, comunidad)
    score: float = 0.0
    meta: dict = field(default_factory=dict)
