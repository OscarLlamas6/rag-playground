"""Evaluación: ¿ayuda el grafo? Cada configuración se mide con las MISMAS preguntas.

Las preguntas son multi-salto (la respuesta es una lista de entidades que hay que construir siguiendo relaciones). Cada una trae
`esperadas`: las entidades que debe contener una buena respuesta. Se mide por separado cada capa, como en las otras sesiones:

    cobertura del contexto  ¿llegaron al prompt las entidades esperadas?            (la RECUPERACIÓN; gratis salvo descomponer y global)
    recall de la respuesta  ¿las nombró el modelo?                                  (la GENERACIÓN; usa el LLM)
    precisión               ¿lo que nombró era esperado? (sin contar lo que ya decía la pregunta)
    completas               respuestas con recall 1.0 Y precisión 1.0 (todo y solo lo esperado)
    citas válidas           respuestas que citan y cuyas citas apuntan a elementos reales
    costo y latencia        por pregunta, promedio

Una entidad "aparece" en un texto si se nombra (o su alias): es una medición por CÓDIGO, sin juez LLM. Mide qué entidades
aparecen, no si el razonamiento es correcto. Con pocas preguntas, una diferencia de una o dos es RUIDO.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from . import graph as G
from .config import get_settings
from .rag import NeedsSetup, RagService

TYPES = ("impacto", "dependencias", "encadenada", "equipos", "global", "directa")


@dataclass(frozen=True)
class Config:
    name: str
    label: str
    mode: str
    k: int = 4
    hops: int = 4
    decompose: bool = False
    contextual: bool = False
    graph: str = "curated"
    llm_in_retrieval: bool = False      # descomponer y la búsqueda global llaman al LLM aunque no se genere respuesta

    def options(self) -> dict:
        return {"mode": self.mode, "k": self.k, "hops": self.hops, "decompose": self.decompose, "contextual": self.contextual,
                "graph": self.graph}


CONFIGS = (
    Config("vector_k4", "Vectorial (k=4)", "vector"),
    Config("vector_k8", "Vectorial (k=8)", "vector", k=8),
    Config("vector_ctx", "Vectorial (k=4) + contextual", "vector", contextual=True),
    Config("vector_dec", "Vectorial (k=4) + descomposición", "vector", decompose=True, llm_in_retrieval=True),
    Config("graph_local", "Grafo local (curado, 4 saltos)", "graph_local"),
    Config("graph_local_h2", "Grafo local (curado, 2 saltos)", "graph_local", hops=2),
    Config("graph_local_ext", "Grafo local (extraído, 4 saltos)", "graph_local", graph="extracted"),
    Config("hybrid", "Híbrido: vectores + grafo", "hybrid_graph"),
    Config("hybrid_dec", "Híbrido + descomposición", "hybrid_graph", decompose=True, llm_in_retrieval=True),
    Config("graph_global", "Grafo global (comunidades)", "graph_global", llm_in_retrieval=True),
)
BY_NAME = {c.name: c for c in CONFIGS}


def load_questions() -> list[dict]:
    questions: list[dict] = []
    for path in get_settings().eval_files:
        if path.exists():
            for q in json.loads(path.read_text(encoding="utf-8")):
                questions.append({"origen": "curso", **q})
    return questions


def coverage(entities: dict[str, dict], text: str, expected: list[str], question: str) -> dict:
    """Qué fracción de las entidades esperadas aparece en un texto (recall) y qué fracción de las entidades mencionadas es esperada
    (precisión). Las entidades que nombra la propia pregunta no cuentan: no hay mérito en repetirlas."""
    asked = {e for e, _ in G.link_by_name(entities, question)}
    mentioned = {e for e, _ in G.link_by_name(entities, text)} - asked
    wanted = set(expected)
    hits = mentioned & wanted
    return {"recall": len(hits) / len(wanted), "precision": len(hits) / len(mentioned) if mentioned else 0.0,
            "missing": sorted(wanted - mentioned), "extra": sorted(mentioned - wanted)}


def available(svc: RagService, config: Config) -> str | None:
    """Por qué una configuración no se puede correr ahora (o None si sí)."""
    if svc.chunk_count() == 0 and config.mode in ("vector", "hybrid_graph"):
        return "falta ingestar"
    if config.contextual and not svc.store.count_contextual():
        return "la ingesta no es contextual (reingesta con «contextual»)"
    sources = svc.graphs.sources()
    if config.mode != "vector" and config.graph not in sources:
        return f"falta construir el grafo «{config.graph}»"
    if config.mode == "graph_global" and not any(c.summary for c in svc.graphs.load_communities(config.graph)):
        return "faltan los resúmenes de las comunidades"
    return None


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def evaluate(svc: RagService, configs: list[str] | None = None, generate: bool = False, question_ids: list[str] | None = None) -> dict:
    if svc.chunk_count() == 0 and not svc.graphs.sources():
        raise NeedsSetup("No hay nada que evaluar: ejecuta la ingesta y construye el grafo.")
    entities = G.load_inventory(get_settings().graph_file)[0]
    questions = [q for q in load_questions() if not question_ids or q["id"] in question_ids]
    if configs:
        unknown = [c for c in configs if c not in BY_NAME]
        if unknown:
            raise ValueError(f"Configuraciones desconocidas: {', '.join(unknown)}. Válidas: {', '.join(BY_NAME)}.")
        chosen = [BY_NAME[c] for c in configs]
    else:   # sin elegir: lo gratuito, o todo si se pide generar respuestas
        chosen = [c for c in CONFIGS if generate or not c.llm_in_retrieval]
    skipped = {c.name: why for c in chosen if (why := available(svc, c))}
    runnable = [c for c in chosen if c.name not in skipped]

    if any(c.mode != "graph_global" for c in runnable):       # vectorizar todas las preguntas juntas, una sola vez
        from .embeddings import embed_query
        for q in questions:
            embed_query(q["pregunta"])

    rows = [{"id": q["id"], "question": q["pregunta"], "type": q["tipo"], "origin": q["origen"], "expected": q["esperadas"], "results": {}}
            for q in questions]
    summary: dict[str, dict] = {}
    for config in runnable:
        per_q: list[dict] = []
        for q, row in zip(questions, rows):
            t0 = time.perf_counter()
            if generate:
                a = svc.ask(q["pregunta"], **config.options())
                items, answer, cost = a.items, a.text, a.cost_usd
            else:
                r = svc.retrieve(q["pregunta"], **config.options())
                items, answer, cost, a = r.items, "", r.cost_usd, None
            ms = (time.perf_counter() - t0) * 1000
            context = "\n".join(i.text for i in items)
            ctx = coverage(entities, context, q["esperadas"], q["pregunta"])
            res = {"context_recall": round(ctx["recall"], 3), "context_missing": ctx["missing"],
                   "context_words": len(context.split()), "items": len(items), "cost_usd": round(cost, 6), "ms": round(ms)}
            if a is not None:
                ans = coverage(entities, answer, q["esperadas"], q["pregunta"])
                res.update({"answer": answer, "answer_recall": round(ans["recall"], 3), "answer_precision": round(ans["precision"], 3),
                            "answer_missing": ans["missing"], "answer_extra": ans["extra"],
                            "complete": ans["recall"] == 1.0 and ans["precision"] == 1.0,
                            "abstained": a.abstained, "valid_citations": bool(a.cited) and not a.invalid})
            row["results"][config.name] = res
            per_q.append({"type": q["tipo"], **res})
        by_type = {}
        for t in TYPES:
            sel = [x for x in per_q if x["type"] == t]
            if sel:
                by_type[t] = {"n": len(sel), "context_recall": _mean([x["context_recall"] for x in sel]),
                              **({"answer_recall": _mean([x["answer_recall"] for x in sel])} if generate else {})}
        entry = {"label": config.label, "mode": config.mode, "n": len(per_q),
                 "context_recall": _mean([x["context_recall"] for x in per_q]),
                 "context_full": sum(x["context_recall"] == 1.0 for x in per_q),
                 "context_words": round(sum(x["context_words"] for x in per_q) / max(1, len(per_q))),
                 "cost_usd": round(sum(x["cost_usd"] for x in per_q), 6), "avg_ms": round(sum(x["ms"] for x in per_q) / max(1, len(per_q))),
                 "by_type": by_type}
        if generate:
            entry.update({"answer_recall": _mean([x["answer_recall"] for x in per_q]),
                          "answer_precision": _mean([x["answer_precision"] for x in per_q]),
                          "complete": sum(x["complete"] for x in per_q), "valid_citations": sum(x["valid_citations"] for x in per_q),
                          "abstained": sum(x["abstained"] for x in per_q)})
        summary[config.name] = entry
    s = get_settings()
    return {"summary": {"n": len(questions), "generate": generate, "configs": summary, "skipped": skipped,
                        "model": f"{s.llm_provider} · {s.llm_model}", "embeddings": f"{s.embeddings_provider} · {s.embeddings_model}",
                        "cost_usd": round(sum(c["cost_usd"] for c in summary.values()), 6),
                        "types": {t: sum(q["tipo"] == t for q in questions) for t in TYPES if any(q["tipo"] == t for q in questions)},
                        "warning": f"Con {len(questions)} preguntas, una o dos de diferencia son ruido: mira los patrones por tipo, no un punto."},
            "rows": rows}
