"""Evaluación de la RECUPERACIÓN: ¿qué modo de búsqueda funciona mejor con MIS documentos y MIS preguntas?

Cada pregunta de prueba trae una `clave`: el texto que debe aparecer en un fragmento para que ESE fragmento responda.
Así se puede medir cualquier estrategia de fragmentado, aunque los fragmentos cambien. No llama al modelo de lenguaje
(salvo el reranking, si se pide): es gratis con embeddings locales.

    hit@1     ¿el PRIMER fragmento contiene la respuesta?
    hit@k     ¿alguno de los k primeros la contiene?
    MRR       promedio de 1 / posición del primer acierto (1 = siempre primero; 0.5 = suele ser el segundo; 0 = nunca)
    contexto  palabras promedio que se enviarían al modelo: más palabras = más tokens = más costo

Se desglosa por TIPO de pregunta porque cada técnica ataca un punto ciego distinto: las paráfrasis (`semantica`) las gana
el significado; los códigos exactos (`lexica`) las gana BM25; la híbrida busca cubrir ambas. Con pocas preguntas, 1 o 2
aciertos de diferencia son RUIDO: sirve para comparar ideas, no para prometer exactitud.
"""

from __future__ import annotations

import json

from .config import get_settings
from .lexical import normalize
from .rag import RagService

TYPES = ("semantica", "lexica", "tabla", "filtro")        # `conversacional` necesita historial: no aplica a un RAG one-shot
LABELS = {"vector": "Vectorial", "bm25": "BM25", "hybrid": "Híbrida (RRF)", "hybrid+rerank": "Híbrida + reranking"}


def load_questions() -> list[dict]:
    return json.loads(get_settings().eval_file.read_text(encoding="utf-8"))


def contains_key(text: str, key: str) -> bool:
    return normalize(key) in normalize(text)


def evaluate_modes(service: RagService, k: int = 3, candidates: int = 8, with_rerank: bool = False) -> dict:
    questions = [q for q in load_questions() if q["tipo"] in TYPES]
    modes = [("vector", False), ("bm25", False), ("hybrid", False)] + ([("hybrid", True)] if with_rerank else [])
    results: dict[str, dict] = {}
    rows = [{"id": q["id"], "question": q["pregunta"], "type": q["tipo"], "audience": (q.get("filtro") or {}).get("publico"),
             "positions": {}} for q in questions]
    cost = 0.0

    for mode, rerank in modes:
        name = "hybrid+rerank" if rerank else mode
        per_type: dict[str, dict] = {t: {"n": 0, "hit1": 0} for t in TYPES}
        hit1 = hitk = 0
        reciprocal = words = 0.0
        for q, row in zip(questions, rows):
            r = service.search(q["pregunta"], mode=mode, k=k, candidates=candidates,
                               audience=(q.get("filtro") or {}).get("publico"), rerank=rerank)
            if r.rerank_llm:
                cost += r.rerank_llm.cost_usd
            position = next((i for i, h in enumerate(r.hits, start=1) if contains_key(h.text, q["clave"])), None)
            row["positions"][name] = position
            hit1 += position == 1
            hitk += position is not None
            reciprocal += 1 / position if position else 0.0
            words += sum(len(h.text.split()) for h in r.hits)
            per_type[q["tipo"]]["n"] += 1
            per_type[q["tipo"]]["hit1"] += position == 1
        n = len(questions)
        results[name] = {"label": LABELS[name], "hit1": hit1, "hitk": hitk, "mrr": round(reciprocal / n, 3),
                         "context_words": round(words / n), "by_type": per_type}
    return {"summary": {"k": k, "candidates": candidates, "n": len(questions), "cost_usd": round(cost, 6), "modes": results},
            "rows": rows}
