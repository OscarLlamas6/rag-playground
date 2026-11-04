"""Los modelos de lenguaje del RAG avanzado.

Un solo punto de entrada, `generate(prompt, purpose=...)`, para TODAS las llamadas: la respuesta final y también las que
ayudan en el camino (frase de contexto, extracción de tripletas, descomposición, resúmenes de comunidades). El modelo es el
mismo; lo que cambia es el prompt (cada uno vive en `prompts/<nombre>/vN.j2`) y `purpose`, que solo usa el proveedor `fake`.

Proveedores (variable LLM_PROVIDER): anthropic, openai, local (LM Studio u otro servidor compatible con OpenAI) y fake
(un doble de prueba con reglas, sin modelo: sirve para probar el flujo offline y para las pruebas automáticas).

Cada llamada devuelve el texto, los tokens, el costo y los segundos: en un RAG real el costo de cada pregunta se mide.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from .config import get_settings
from .embeddings import strip_accents

# USD por millón de tokens (entrada, salida). Confirma los precios vigentes en la página del proveedor.
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
PURPOSES = ("answer", "contexto", "extraer", "descomponer", "resumen", "mapa")


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

    def view(self) -> dict:
        return {"provider": self.provider, "model": self.model, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_usd": round(self.cost_usd, 6), "seconds": round(self.seconds, 3)}


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

    client = openai.OpenAI(base_url=get_settings().local_url, api_key="no-se-usa", max_retries=0, timeout=600)
    t0 = time.perf_counter()
    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0)
            break
        except openai.BadRequestError:
            # LM Studio descarga un modelo para cargar otro (p. ej. al alternar embeddings y LLM) y responde 400 «Model is
            # unloaded» hasta que lo recarga solo: se espera un momento y se reintenta.
            if attempt == 5:
                raise
            time.sleep(5)
    seconds = time.perf_counter() - t0
    u = response.usage
    return LLMResult(response.choices[0].message.content or "", model, "local",
                     getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0, seconds, 0.0)


# ───────────────────────── el doble de prueba (proveedor fake) ─────────────────────────
# No es un modelo: son reglas deterministas que producen salidas con la FORMA correcta para cada tarea. Sirven para ver la
# interfaz y para probar el flujo sin red. Las cifras que importan salen con un modelo real.

_ART = r"(?:el|la|los|las)"


def _words(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z0-9]+", strip_accents(text).lower()) if len(w) > 3 or w.isdigit()}


def _fake_answer(prompt: str) -> str:
    """Extractivo: elige los elementos del contexto (hechos, fragmentos, comunidades) que más palabras comparten con la
    pregunta y los cita. Si ninguno comparte al menos dos palabras, responde "no encuentro"."""
    question = (re.search(r"Pregunta:\s*(.+)\s*$", prompt, re.S) or [None, ""])[1]
    wanted = _words(question)
    scored: list[tuple[int, str, str]] = []
    for number, body in re.findall(r"\[(\d+)\] \((?:hecho|fragmento|comunidad)[^\n]*\)\n(.*?)(?=\n\[\d+\] \(|\n</contexto>)", prompt, re.S):
        score = len(wanted & _words(body))
        if score >= 2:
            scored.append((score, number, " ".join(body.split())))
    if not scored:
        return NOT_FOUND
    best = sorted(scored, key=lambda s: (-s[0], int(s[1])))[:4]
    return " ".join(f"{text} [{number}]" for _, number, text in sorted(best, key=lambda s: int(s[1])))


def _fake_context(prompt: str) -> str:
    section = (re.search(r"Sección del fragmento:\s*(.+)", prompt) or [None, "el catálogo"])[1].strip()
    return f"Este fragmento pertenece a la sección «{section}» del catálogo de servicios de TI de Andes Fibra."


def _fake_extract(prompt: str) -> str:
    """Lee la sección con expresiones regulares y deja los defectos típicos de un modelo, para que se vea el trabajo de la
    normalización y de la validación: conserva artículos ("la Red troncal"), y en dos secciones añade una tripleta inventada
    (entidad que no existe) y otra con el sujeto cambiado (una arista falsa)."""
    m = re.search(r"Texto de la sección «(.+?)»:\n(.*)\Z", prompt, re.S)
    if not m:
        return '{"tripletas": []}'
    title, text = m.group(1), m.group(2)
    triples: list[dict] = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        dep = re.search(r"depende (de la|del|de el|de los|de las|de) (.+?)\.?$", sentence)
        if dep:
            first = {"del": "el ", "de la": "la ", "de el": "el ", "de los": "los ", "de las": "las "}.get(dep.group(1), "")
            parts = re.split(r",\s*|\s+y\s+(?=" + _ART + r"\s)", dep.group(2))
            for i, obj in enumerate(p.strip() for p in parts if p.strip()):
                triples.append({"sujeto": title, "relacion": "depende_de", "objeto": (first if i == 0 else "") + obj})
        run = re.search(r"[Ss]e ejecuta en (.+?)\.?$", sentence)
        if run:
            triples.append({"sujeto": title, "relacion": "corre_en", "objeto": run.group(1).strip()})
        adm = re.search(r"[Ll]o administra (?:el |la )?(.+?)\.?$", sentence)
        if adm:
            triples.append({"sujeto": title, "relacion": "administrado_por", "objeto": adm.group(1).strip()})
    if title == "Portal Mi Andes":
        triples.append({"sujeto": title, "relacion": "depende_de", "objeto": "Servicio de correo corporativo"})   # entidad inventada
    if title == "Chat de soporte":
        triples.append({"sujeto": "Sistema de tickets", "relacion": "depende_de", "objeto": title})               # arista falsa
    return json.dumps({"tripletas": triples}, ensure_ascii=False)


def _fake_decompose(prompt: str) -> str:
    question = (re.search(r"Pregunta del usuario:\s*(.+)\s*$", prompt, re.S) or [None, ""])[1].strip()
    parts = [p.strip(" ,;¿?") for p in re.split(r"\s+y\s+(?=qué|cuál|quién|cuáles|quiénes|cómo)|;", question) if p.strip(" ,;¿?")]
    subs = [("¿" + p + "?") if not p.startswith("¿") else p for p in parts] or [question]
    return json.dumps({"subpreguntas": subs[:3]}, ensure_ascii=False)


def _fake_summary(prompt: str) -> str:
    members = re.findall(r"^- (.+?) \(", prompt.split("Miembros:")[-1].split("Relaciones internas:")[0], re.M)
    teams = re.search(r"Equipos que los administran: (.+)", prompt)
    text = f"Este grupo reúne {len(members)} elementos: {', '.join(members)}."
    return text + (f" Los administran: {teams.group(1).strip()}" if teams else "")


def _fake_map(prompt: str) -> str:
    question = (re.search(r"Pregunta:\s*(.+)", prompt) or [None, ""])[1]
    summary = (re.search(r"Resumen de la comunidad:\s*(.+?)\s*\n\s*\nRespuesta", prompt, re.S) or [None, ""])[1]
    return " ".join(summary.split()) if len(_words(question) & _words(summary)) >= 1 else "NADA"


def _fake(prompt: str, purpose: str) -> LLMResult:
    t0 = time.perf_counter()
    handler = {"answer": _fake_answer, "contexto": _fake_context, "extraer": _fake_extract, "descomponer": _fake_decompose,
               "resumen": _fake_summary, "mapa": _fake_map}[purpose]
    text = handler(prompt)
    return LLMResult(text, "fake-reglas", "fake", len(prompt.split()), len(text.split()), time.perf_counter() - t0, 0.0)


def generate(prompt: str, max_tokens: int = 500, purpose: str = "answer") -> LLMResult:
    """Llama al modelo configurado. Con anthropic/openai consume dinero (fracciones de centavo por llamada)."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose debe ser uno de: {', '.join(PURPOSES)}")
    s = get_settings()
    try:
        if s.llm_provider == "anthropic":
            return _anthropic(prompt, s.llm_model, max_tokens)
        if s.llm_provider == "openai":
            return _openai(prompt, s.llm_model, max_tokens)
        if s.llm_provider == "local":
            return _local(prompt, s.llm_model, max_tokens)
        return _fake(prompt, purpose)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - mensaje claro para cualquier otro fallo del SDK
        hint = ""
        if s.llm_provider == "local":
            hint = f" Abre LM Studio, activa el servidor local y carga '{s.llm_model}'. URL usada: {s.local_url}."
            if os.getenv("RUNNING_IN_DOCKER") == "1":
                hint += " Desde un contenedor usa docker-compose.host.yml o «Serve on Local Network» (ver el README)."
        raise LLMError(f"El modelo '{s.llm_provider}' falló ({type(exc).__name__}): {str(exc)[:200]}.{hint}") from exc
