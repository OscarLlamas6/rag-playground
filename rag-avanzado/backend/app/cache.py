"""Caché en disco de las llamadas al LLM que son COSTOSAS y repetibles: la frase de contexto de cada fragmento,
las tripletas extraídas de cada sección y el resumen de cada comunidad.

La clave es un hash de (tarea, proveedor, modelo, prompt completo): si cambias el modelo, el texto del catálogo o la versión
del prompt, la clave cambia y se vuelve a llamar; si no, se reutiliza el resultado (0 s y US$ 0). Las respuestas finales NO
se cachean: la latencia y el costo que se miden en la evaluación deben ser reales.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from .config import get_settings
from .llm import LLMResult, generate

_LOCK = threading.Lock()


def _file(task: str) -> Path:
    return get_settings().cache_dir / f"llm_{task}.json"


def _key(task: str, prompt: str) -> str:
    s = get_settings()
    return hashlib.sha256(f"{task}|{s.llm_provider}|{s.llm_model}|{prompt}".encode()).hexdigest()


def cached_generate(task: str, prompt: str, max_tokens: int = 400, force: bool = False) -> tuple[LLMResult, bool]:
    """Devuelve (resultado, vino_del_cache). Con `force` ignora la caché y la reescribe."""
    path, key = _file(task), _key(task, prompt)
    with _LOCK:
        store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not force and key in store:
        s = get_settings()
        return LLMResult(store[key], s.llm_model, s.llm_provider, 0, 0, 0.0, 0.0), True
    result = generate(prompt, max_tokens=max_tokens, purpose=task)
    with _LOCK:
        store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        store[key] = result.text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    return result, False


def cache_size(task: str) -> int:
    path = _file(task)
    return len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0
