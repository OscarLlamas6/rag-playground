"""Evaluación mínima: un RAG tiene DOS capas y se miden por separado.

    pregunta ─► RECUPERACIÓN ─► fragmentos ─► GENERACIÓN ─► respuesta
                 hit@k, MRR                    dato clave, citas válidas, "no sé" bien dicho

Si el fragmento correcto no llega al prompt, ningún prompt puede arreglarlo; por eso la recuperación se mide
sin llamar al modelo (gratis y en segundos). La generación sí cuesta: se activa solo si se pide.

Con pocas preguntas, 1 o 2 aciertos de diferencia son RUIDO. Esto sirve para comparar ideas y detectar
regresiones al cambiar el prompt o el fragmentado, no para prometer una exactitud.
"""

from __future__ import annotations

import json

from .config import get_settings
from .rag import RagService


def load_questions() -> list[dict]:
    return json.loads(get_settings().eval_file.read_text(encoding="utf-8"))


def _position(hits, question: dict) -> int | None:
    """Posición (desde 1) del primer fragmento que viene de la sección correcta; None si no aparece."""
    for i, h in enumerate(hits, start=1):
        if h.metadata.get("source") == question["fuente"] and h.metadata.get("section") == question["seccion"]:
            return i
    return None


def evaluate(service: RagService, k: int = 4, with_generation: bool = False) -> dict:
    questions = load_questions()
    rows: list[dict] = []
    in_scope = [q for q in questions if not q.get("fuera_de_alcance")]
    hit1 = hitk = 0
    reciprocal = 0.0
    gen = {"fact": 0, "valid_citations": 0, "abstain_ok": 0}
    out_scope = [q for q in questions if q.get("fuera_de_alcance")]
    cost = 0.0

    for q in questions:
        row: dict = {"id": q["id"], "question": q["pregunta"], "out_of_scope": bool(q.get("fuera_de_alcance"))}
        hits, _ = service.retrieve(q["pregunta"], k)
        row["top_score"] = hits[0].score if hits else None
        if not row["out_of_scope"]:
            pos = _position(hits, q)
            row.update(expected=f"{q['fuente']} › {q['seccion']}", position=pos,
                       got=[f"{h.metadata.get('section')} ({h.score:.2f})" for h in hits[:2]])
            hit1 += pos == 1
            hitk += pos is not None
            reciprocal += 1 / pos if pos else 0.0
        if with_generation:
            answer = service.ask(q["pregunta"], k=k, include_prompt=False)
            row["answer"] = answer.text
            if answer.llm:
                cost += answer.llm.cost_usd
            if row["out_of_scope"]:
                row["abstained_ok"] = answer.abstained
                gen["abstain_ok"] += answer.abstained
            else:
                fact = q["clave"].lower() in answer.text.lower()
                valid = bool(answer.cited) and not answer.invalid
                row.update(fact_ok=fact, citations_ok=valid)
                gen["fact"] += fact
                gen["valid_citations"] += valid
        rows.append(row)

    n = len(in_scope)
    summary = {
        "k": k, "in_scope": n, "out_of_scope": len(out_scope),
        "retrieval": {"hit1": hit1, "hitk": hitk, "mrr": round(reciprocal / n, 3) if n else 0.0},
    }
    if with_generation:
        summary["generation"] = {**gen, "cost_usd": round(cost, 6)}
    return {"summary": summary, "rows": rows}
