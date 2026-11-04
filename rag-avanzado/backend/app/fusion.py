"""Fusión de rankings: Reciprocal Rank Fusion (RRF).

Se usa aquí para juntar las listas de las SUBPREGUNTAS de la descomposición: cada subpregunta recupera su propia lista y lo que
aparece bien ubicado en varias listas sube al primer lugar. Se ignoran los puntajes (escalas distintas) y se usa solo la POSICIÓN:
cada lista aporta 1 / (c + posición) a cada elemento y se suman los aportes.

    ejemplo (c = 60)   sub1: A, C, B    sub2: B, A, D
        A = 1/61 + 1/62 = 0.0325      B = 1/63 + 1/61 = 0.0323      C = 1/62 = 0.0161      D = 1/63 = 0.0159
"""

from __future__ import annotations

from collections import defaultdict

from .domain import Item


def rrf(lists: dict[str, list[Item]], k: int = 5, c: int = 60) -> list[tuple[Item, dict[str, int]]]:
    """Fusiona listas ordenadas. Devuelve (elemento con el puntaje RRF, posición en cada lista) de mayor a menor."""
    total: dict[str, float] = defaultdict(float)
    sample: dict[str, Item] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, items in lists.items():
        for position, item in enumerate(items, start=1):
            total[item.id] += 1 / (c + position)
            sample.setdefault(item.id, item)
            ranks[item.id][name] = position
    ordered = sorted(total, key=lambda i: (-total[i], i))[:k]
    return [(Item(i, sample[i].kind, sample[i].text, sample[i].label, total[i], sample[i].meta), ranks[i]) for i in ordered]
