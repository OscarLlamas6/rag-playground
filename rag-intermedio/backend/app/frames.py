"""EXPERIMENTAL · descripciones de fotogramas de un video, alineadas por tiempo con la transcripción.

Desactivado por defecto (`FRAME_CAPTIONS=false`). Cuando se activa, cada `FRAME_INTERVAL_SECONDS` se extrae un fotograma del video
con ffmpeg, se lo describe un modelo CON VISIÓN y cada descripción se indexa como un fragmento más (`content = "frame"`), con
el mismo `start_s`/`end_s` que cualquier fragmento de audio: una pregunta puede traer la transcripción Y lo que se veía en pantalla.

    local      un servidor compatible con OpenAI (LM Studio): chat.completions con una parte `image_url` (data URL base64)
    openai     la Responses API: una parte `input_image` con la data URL
    anthropic  la Messages API: un bloque `image` con `source.type = base64`

Honestidad, medida con el video de prueba (una persona escribiendo notas a mano en un cristal, 640×360):
  * el modelo local (gemma de 12B) lee MAL la escritura manuscrita a esa resolución ("AGENTS!" salió como "PACIENTS");
  * tarda unos 16 s por fotograma y gasta tokens de razonamiento;
  * la transcripción es lo que sostiene el RAG; las descripciones de fotogramas son un complemento y pueden ser ruido.
Sirve para videos con diapositivas, diagramas o pantallas legibles. Revisa lo que produce antes de confiar en ello.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings
from .extras import MediaError, ffmpeg_path
from .locations import format_time

PROMPT = ("Describe en español, en una o dos oraciones, lo que se ve en este fotograma de un video: diapositivas, pizarras, "
          "pantallas o notas escritas (transcribe el texto solo si lo lees con claridad). Si no hay nada informativo, responde "
          "únicamente: sin contenido.")
_CACHE_VERSION = "v1"


@dataclass
class Caption:
    t: float                # segundo del video al que corresponde el fotograma
    text: str


# ───────────────────────── fotogramas (ffmpeg) ─────────────────────────
def frame_times(duration_s: float, interval_s: float) -> list[float]:
    """Un instante cada `interval_s`, empezando a mitad del primer intervalo (el segundo 0 suele ser negro o un título)."""
    if duration_s <= 0 or interval_s <= 0:
        return []
    times, t = [], interval_s / 2
    while t < duration_s:
        times.append(round(t, 1))
        t += interval_s
    return times


def _extract_frame(video: Path, t: float) -> bytes:
    """Un fotograma JPEG en el segundo `t` (ffmpeg, escalado a 640 px de ancho como máximo)."""
    exe = ffmpeg_path()
    if not exe:
        raise MediaError("Describir fotogramas necesita ffmpeg (sudo apt install ffmpeg / brew install ffmpeg).")
    proc = subprocess.run([exe, "-nostdin", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
                           "-vf", "scale='min(640,iw)':-2", "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                          capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise MediaError(f"ffmpeg no pudo extraer el fotograma {format_time(t)} de {video.name}.")
    return proc.stdout


# ───────────────────────── describir con un modelo con visión ─────────────────────────
def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _describe_local(jpeg: bytes, settings: Settings) -> str:
    import openai

    client = openai.OpenAI(base_url=settings.local_url, api_key="no-se-usa", max_retries=0, timeout=300)
    response = client.chat.completions.create(
        model=settings.vision_model, max_tokens=1200, temperature=0,      # el modelo local razona antes de contestar: margen de tokens
        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                               {"type": "image_url", "image_url": {"url": _data_url(jpeg)}}]}])
    return response.choices[0].message.content or ""


def _describe_openai(jpeg: bytes, settings: Settings) -> str:
    import openai

    if not os.getenv("OPENAI_API_KEY"):
        raise MediaError("Falta OPENAI_API_KEY (VISION_PROVIDER=openai).")
    response = openai.OpenAI().responses.create(
        model=settings.vision_model, max_output_tokens=300,
        input=[{"role": "user", "content": [{"type": "input_text", "text": PROMPT},
                                            {"type": "input_image", "image_url": _data_url(jpeg)}]}])
    return response.output_text


def _describe_anthropic(jpeg: bytes, settings: Settings) -> str:
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise MediaError("Falta ANTHROPIC_API_KEY (VISION_PROVIDER=anthropic).")
    response = anthropic.Anthropic().messages.create(
        model=settings.vision_model, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": PROMPT}]}])
    return "".join(b.text for b in response.content if b.type == "text")


def describe_frame(jpeg: bytes, settings: Settings) -> str:
    """Descripción de UN fotograma con el proveedor de visión configurado (VISION_PROVIDER / VISION_MODEL)."""
    describe = {"local": _describe_local, "openai": _describe_openai, "anthropic": _describe_anthropic}[settings.vision_provider]
    try:
        return describe(jpeg, settings).strip()
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"El modelo de visión '{settings.vision_provider}' falló ({type(exc).__name__}): {exc}") from exc


# ───────────────────────── descripciones de un video, con caché ─────────────────────────
def caption_video(video: Path, duration_s: float, file_digest: str, settings: Settings) -> list[Caption]:
    """Las descripciones de todos los fotogramas (cada `frame_interval_s`), guardadas en disco por video + intervalo + modelo.

    💰 Con openai o anthropic cada fotograma es una llamada de pago; con el modelo local son ~16 s por fotograma."""
    key = hashlib.sha256(f"{file_digest}|{settings.frame_interval_s}|{settings.vision_provider}|{settings.vision_model}|{PROMPT}".encode()).hexdigest()[:24]
    cache = settings.media_cache_dir / "frames" / f"{key}-{_CACHE_VERSION}.json"
    if cache.exists():
        return [Caption(**c) for c in json.loads(cache.read_text(encoding="utf-8"))]
    captions: list[Caption] = []
    for t in frame_times(duration_s, settings.frame_interval_s):
        text = describe_frame(_extract_frame(video, t), settings)
        if text and "sin contenido" not in text.lower()[:40]:
            captions.append(Caption(t, " ".join(text.split())))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(c) for c in captions], ensure_ascii=False), encoding="utf-8")
    return captions


def captions_to_sections(captions: list[Caption], media_name: str, interval_s: float) -> tuple[list[tuple[str, str]], list[dict]]:
    """Cada descripción pasa a ser una sección con el rango de tiempo que representa (la marca ± medio intervalo)."""
    sections, metas = [], []
    for c in captions:
        start, end = max(0.0, c.t - interval_s / 2), c.t + interval_s / 2
        sections.append((f"{format_time(start)}–{format_time(end)} · fotograma", f"Descripción del fotograma en {format_time(c.t)}: {c.text}"))
        metas.append({"media": media_name, "start_s": round(start, 2), "end_s": round(end, 2), "content": "frame"})
    return sections, metas
