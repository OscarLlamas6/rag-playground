"""Fusión de rankings: Reciprocal Rank Fusion (RRF).

Problema: el coseno va de 0 a 1 y BM25 puede ir de 0 a 20 o más. Son escalas incomparables: sumarlas no tiene sentido.
Solución de RRF: IGNORAR los puntajes y usar solo la POSICIÓN. Cada lista aporta 1 / (c + posición) a cada fragmento;
se suman los aportes. Un fragmento bien ubicado en AMBAS listas sube al primer lugar: es una votación entre buscadores.

    ejemplo (c = 60)   BM25: A, C, B    vectorial: B, A, D
        A = 1/61 + 1/62 = 0.0325      B = 1/63 + 1/61 = 0.0323      C = 1/62 = 0.0161      D = 1/63 = 0.0159
"""

from __future__ import annotations

from collections import defaultdict

from .domain import Hit


def rrf(lists: dict[str, list[Hit]], k: int = 5, c: int = 60, weights: dict[str, float] | None = None) -> list[tuple[Hit, dict[str, int]]]:
    """Fusiona listas ordenadas. Devuelve (hit con el puntaje RRF, posiciones en cada lista) de mayor a menor."""
    weights = weights or {}
    total: dict[str, float] = defaultdict(float)
    sample: dict[str, Hit] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, hits in lists.items():
        for position, hit in enumerate(hits, start=1):
            total[hit.id] += weights.get(name, 1.0) / (c + position)
            sample.setdefault(hit.id, hit)
            ranks[hit.id][name] = position
    ordered = sorted(total, key=lambda i: -total[i])[:k]
    return [(Hit(i, sample[i].text, sample[i].metadata, total[i]), ranks[i]) for i in ordered]
