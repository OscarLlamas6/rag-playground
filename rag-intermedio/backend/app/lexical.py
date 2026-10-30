"""Búsqueda léxica (por PALABRAS): BM25.

La búsqueda semántica entiende paráfrasis ("se me olvidó la clave" ≈ "restablecer contraseña") pero casi no distingue
identificadores exactos: para un modelo de embeddings, OPT-1102 y OPT-1107 son casi el mismo texto. BM25, el algoritmo
clásico de Lucene/Elasticsearch/OpenSearch, hace lo contrario: puntúa más a un fragmento cuando contiene las palabras
de la consulta, sobre todo si son RARAS en el corpus, y no entiende sinónimos.

    puntaje(fragmento) = Σ por cada palabra de la consulta:  IDF(palabra) × frecuencia saturada / normalización de largo

    IDF            una palabra rara pesa mucho ("OPT-1107"), una común casi nada ("el")
    saturación(k1) la 5.ª aparición aporta menos que la 1.ª (rendimientos decrecientes)
    largo (b)      no favorece a los fragmentos largos solo por tener más palabras
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable

from .domain import Chunk, Hit

STOPWORDS = frozenset(
    "a al algo ante como con cual cuales cuando cuanto cuantos cuanta cuantas de del desde donde el ella ellos en entre es "
    "esta este esto estos fue ha han hay la las le les lo los me mi mis muy no nos o para pero por que se si sin sobre su sus "
    "te tiene tienen tu un una uno unos unas y ya yo hago hace puedo puede pueden debe deben son ser estar".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def normalize(text: str) -> str:
    """Minúsculas, sin acentos y con espacios simples."""
    plain = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return " ".join(plain.lower().split())


def tokenize(text: str, prefix: int | None = 5) -> list[str]:
    """Texto → lista de términos. Conserva códigos con guion o punto COMPLETOS ('e-217', '192.168.1.1') y también sus
    partes; quita palabras vacías; y recorta las palabras a `prefix` letras ('cancelar' y 'cancelación' → 'cance'),
    un "stemming" ingenuo que junta formas de una misma palabra. No se aplica a tokens con dígitos (códigos)."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(normalize(text)):
        for part in [raw] + (re.split(r"[-.]", raw) if re.search(r"[-.]", raw) else []):
            if not part or part in STOPWORDS:
                continue
            tokens.append(part[:prefix] if prefix and part.isalpha() and len(part) > prefix else part)
    return tokens


class BM25:
    def __init__(self, chunks: list[Chunk], text_of: Callable[[Chunk], str], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self._tf = [Counter(tokenize(text_of(c))) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        n = len(chunks)
        self._avg = (sum(self._len) / n) if n else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        # IDF "suavizado": nunca es negativo, aunque un término esté en casi todos los fragmentos
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5, accept: Callable[[dict], bool] | None = None) -> list[Hit]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            if accept and not accept(chunk.metadata):
                continue
            score = 0.0
            for t in terms:
                f = self._tf[i].get(t)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [Hit(self.chunks[i].id, self.chunks[i].text, self.chunks[i].metadata, s) for s, i in scored[:k]]
