"""El modelo que REDACTA la respuesta (el de generación; distinto del modelo de embeddings).

Proveedores (variable LLM_PROVIDER): anthropic, openai, local (LM Studio u otro servidor compatible con OpenAI)
y fake (respuesta extractiva sin modelo: para probar el flujo y para las pruebas automáticas).

Cada llamada devuelve el texto, los tokens, el costo y los segundos: en un RAG real el costo de cada pregunta
se mide, no se adivina.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from .config import get_settings
from .embeddings import strip_accents

# USD por millón de tokens (entrada, salida). Referencia a octubre de 2026: confirma en la página del proveedor.
PRICES = {
    # Anthropic
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
    # OpenAI
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}   # un modelo que no está aquí (y que no es local) se muestra con costo 0: agrégalo para ver su costo real
NOT_FOUND = "No encuentro esa información en los documentos."


class LLMError(RuntimeError):
    """El modelo de generación no está disponible (llave faltante, servidor apagado…)."""


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    seconds: float
    cost_usd: float


def _cost(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in PRICES:   # un modelo local no cobra por token
        return 0.0
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _anthropic(prompt: str, model: str, max_tokens: int) -> LLMResult:
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMError("Falta ANTHROPIC_API_KEY (LLM_PROVIDER=anthropic).")
    client = anthropic.Anthropic()
    params = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    try:
        # `effort: low` limita el razonamiento interno: sus tokens cuentan dentro de max_tokens y, en una tarea
        # simple como esta, solo encarecen la respuesta (o la dejan vacía si agotan el límite).
        response = client.messages.create(**params, output_config={"effort": "low"})
    except TypeError:
        response = client.messages.create(**params)
    seconds = time.perf_counter() - t0
    text = "".join(b.text for b in response.content if b.type == "text")
    u = response.usage
    tokens_in = u.input_tokens + (getattr(u, "cache_read_input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    return LLMResult(text, model, "anthropic", tokens_in, u.output_tokens, seconds, _cost(model, tokens_in, u.output_tokens))


def _openai(prompt: str, model: str, max_tokens: int) -> LLMResult:
    import openai

    if not os.getenv("OPENAI_API_KEY"):
        raise LLMError("Falta OPENAI_API_KEY (LLM_PROVIDER=openai).")
    client = openai.OpenAI()
    extra = {"reasoning": {"effort": "none"}} if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")) else {}
    t0 = time.perf_counter()
    response = client.responses.create(model=model, input=prompt, max_output_tokens=max(16, max_tokens), **extra)
    seconds = time.perf_counter() - t0
    u = response.usage
    return LLMResult(response.output_text, model, "openai", u.input_tokens, u.output_tokens, seconds,
                     _cost(model, u.input_tokens, u.output_tokens))


def _local(prompt: str, model: str, max_tokens: int) -> LLMResult:
    import openai

    client = openai.OpenAI(base_url=get_settings().local_url, api_key="no-se-usa", max_retries=0, timeout=300)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0
    )
    seconds = time.perf_counter() - t0
    u = response.usage
    return LLMResult(response.choices[0].message.content or "", model, "local",
                     getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0, seconds, 0.0)


def _fake(prompt: str, question_hint: str | None = None) -> LLMResult:
    """Respuesta extractiva: elige, entre los fragmentos del prompt, la oración que más palabras comparte con la
    pregunta y la cita. Si ninguna comparte palabras, responde "no encuentro". No es un modelo: es un doble de prueba."""
    t0 = time.perf_counter()
    if "<fragmentos>" in prompt:                      # prompt de reranking: ordenar por palabras compartidas con la pregunta
        return _fake_rerank(prompt, t0)
    question = (re.search(r"Pregunta del cliente:\s*(.+)\s*$", prompt, re.S) or [None, question_hint or ""])[1]
    words = lambda s: {w[:6] for w in re.findall(r"[a-z0-9]+", strip_accents(s).lower()) if len(w) > 3 or w.isdigit()}  # noqa: E731
    wanted = words(question)
    best, best_score = (None, 0)
    for number, block in re.findall(r"\[(\d+)\] \(fuente:[^\n]*\)\n(.*?)(?=\n\[\d+\] \(fuente:|\n</contexto>)", prompt, re.S):
        for sentence in re.split(r"(?<=[.!?])\s+", block.strip()):
            score = len(wanted & words(sentence))
            if score > best_score:
                best, best_score = (f"{sentence.strip()} [{number}]", score)
    text = best if best and best_score >= 2 else NOT_FOUND
    return LLMResult(text, "fake-extractive", "fake", len(prompt.split()), len(text.split()), time.perf_counter() - t0, 0.0)


def _fake_rerank(prompt: str, t0: float) -> LLMResult:
    question = (re.search(r"Pregunta:\s*(.+)", prompt) or [None, ""])[1]
    words = lambda s: {w[:6] for w in re.findall(r"[a-z0-9]+", strip_accents(s).lower()) if len(w) > 3 or w.isdigit()}  # noqa: E731
    wanted = words(question)
    blocks = re.findall(r"\[(\d+)\] (.*?)(?=\n\[\d+\] |\n</fragmentos>)", prompt, re.S)
    order = sorted(blocks, key=lambda b: -len(wanted & words(b[1])))
    text = ", ".join(n for n, _ in order)
    return LLMResult(text, "fake-extractive", "fake", len(prompt.split()), len(text.split()), time.perf_counter() - t0, 0.0)


def generate(prompt: str, max_tokens: int = 500) -> LLMResult:
    """Llama al modelo configurado. 💰 Con anthropic/openai consume dinero (fracciones de centavo por pregunta)."""
    s = get_settings()
    try:
        if s.llm_provider == "anthropic":
            return _anthropic(prompt, s.llm_model, max_tokens)
        if s.llm_provider == "openai":
            return _openai(prompt, s.llm_model, max_tokens)
        if s.llm_provider == "local":
            return _local(prompt, s.llm_model, max_tokens)
        return _fake(prompt)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - mensaje claro para cualquier otro fallo del SDK
        raise LLMError(f"El modelo '{s.llm_provider}' falló ({type(exc).__name__}): {exc}") from exc
