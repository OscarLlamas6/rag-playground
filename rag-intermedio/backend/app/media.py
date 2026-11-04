"""Audio y video: transcribir con faster-whisper y partir la transcripción en fragmentos con TIEMPO.

    audio (.mp3 .wav .m4a .flac)  ─┐
                                   ├─► faster-whisper ─► segmentos (inicio, fin, texto) ─► ventanas de ~50 s ─► fragmentos
    video (.mp4 .mkv .mov .webm) ──┘   (el video pasa antes por ffmpeg para sacar solo el audio)

Por qué faster-whisper DIRECTO y no el pipeline de audio de Docling: el DoclingDocument que produce Docling-ASR no conserva
las marcas de tiempo, y sin ellas no se puede citar "minuto 12:30". Aquí cada segmento guarda `start` y `end`.

La transcripción es lo caro (decenas de segundos por una charla), así que se guarda en disco por hash del archivo + modelo +
idioma + vocabulario: volver a ingestar es instantáneo, y cambiar de modelo transcribe de nuevo.

faster-whisper, av y ffmpeg son OPCIONALES (grupo `media`); se importan solo al transcribir. Fija `av<16`.

Qué NO hace por ti: el reconocimiento de voz se equivoca con vocabulario técnico (p. ej. «ChatGBT» por «ChatGPT»). La
transcripción hay que revisarla; `WHISPER_PROMPT` (nombres propios y siglas) ayuda, y un modelo mayor también.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import Settings, get_settings
from .domain import Document
from .extras import MediaError, ffmpeg_path, require
from .locations import format_range

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm")
EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime", ".webm": "video/webm"}
_CACHE_VERSION = "v1"
_NAME_RE = re.compile(r"^[^/\\\x00]+$")

log = logging.getLogger(__name__)


def kind_of(path: Path | str) -> str:
    """'audio' o 'video' según la extensión."""
    return "video" if Path(path).suffix.lower() in VIDEO_EXTENSIONS else "audio"


def mime_of(path: Path | str) -> str:
    ext = Path(path).suffix.lower()
    return MIME.get(ext) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"


# ───────────────────────── carpeta de medios: listar y resolver nombres con seguridad ─────────────────────────
def list_media(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith(".")]


def resolve_media(folder: Path, name: str) -> Path | None:
    """El archivo de la carpeta de medios con ese nombre, o None. Se busca ENTRE LOS ARCHIVOS DE LA CARPETA (no se arma una
    ruta con el texto recibido): `../../etc/passwd`, rutas absolutas y nombres con separadores no pueden escapar de la carpeta.

    Lanza ValueError si el nombre es inválido (con separadores, vacío o con bytes nulos)."""
    if not name or not _NAME_RE.match(name) or name in (".", "..") or len(name) > 255:
        raise ValueError("Nombre de archivo no válido.")
    for path in list_media(folder):
        if path.name == name:
            return path
    return None


@lru_cache(maxsize=256)
def _probe(path_str: str, mtime: float, size: int) -> float | None:
    if not ffmpeg_path():
        return None
    ffprobe = (ffmpeg_path() or "").replace("ffmpeg", "ffprobe")
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path_str],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        return round(float(out), 1)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def duration_of(path: Path) -> float | None:
    """Duración en segundos con ffprobe (viene con ffmpeg), o None si no se puede saber."""
    stat = path.stat()
    return _probe(str(path), stat.st_mtime, stat.st_size)


# ───────────────────────── transcripción ─────────────────────────
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model: str
    seconds: float                      # lo que tardó la transcripción real (la primera vez)
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(file_digest: str, model: str, language: str | None, prompt: str | None) -> str:
    """Identifica una transcripción: mismo archivo + mismo modelo + mismo idioma + mismo vocabulario = misma transcripción."""
    tag = hashlib.sha256((prompt or "").encode()).hexdigest()[:8] if prompt else "sin-prompt"
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return f"{file_digest[:24]}-{safe_model}-{language or 'auto'}-{tag}-{_CACHE_VERSION}"


def _cache_file(settings: Settings, key: str) -> Path:
    return settings.media_cache_dir / "transcripts" / f"{key}.json"


def _extract_audio(video: Path, target_dir: Path) -> Path:
    """ffmpeg: del video solo se conserva el audio (mono, 16 kHz, el formato que espera Whisper)."""
    exe = ffmpeg_path()
    if not exe:
        raise MediaError(f"Para transcribir el video {video.name} hace falta ffmpeg (sudo apt install ffmpeg / brew install ffmpeg).")
    out = target_dir / "audio.wav"
    proc = subprocess.run([exe, "-nostdin", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-y", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        detail = (proc.stderr or "").strip().splitlines()[-1:] or ["sin detalle"]
        raise MediaError(f"ffmpeg no pudo extraer el audio de {video.name}: {detail[0]} (¿el video no tiene pista de audio?).")
    return out


_models: dict[str, object] = {}


def _whisper_model(name: str):
    if name not in _models:
        import os

        from faster_whisper import WhisperModel

        _models[name] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=min(16, os.cpu_count() or 4))
    return _models[name]


def _transcribe_raw(path: Path, model: str, language: str | None, prompt: str | None) -> tuple[list[Segment], str, float, float]:
    """Transcribe de verdad. Devuelve (segmentos, idioma, probabilidad del idioma, duración del audio).

    Es el único punto que toca faster-whisper y ffmpeg: las pruebas lo sustituyen por un doble."""
    require("Transcribir audio y video", "faster_whisper", "av")
    try:
        whisper = _whisper_model(model)
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"No se pudo cargar el modelo Whisper '{model}' ({type(exc).__name__}: {exc}). La primera vez se descarga "
                         "de Hugging Face (base ≈ 140 MB, small ≈ 460 MB): revisa tu conexión.") from exc
    with tempfile.TemporaryDirectory(prefix="rag-media-") as tmp:
        source = _extract_audio(path, Path(tmp)) if kind_of(path) == "video" else path
        try:
            segments, info = whisper.transcribe(str(source), language=language, vad_filter=True, beam_size=1,
                                                initial_prompt=prompt)
            out = [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments if s.text.strip()]
        except Exception as exc:  # noqa: BLE001
            raise MediaError(f"Whisper no pudo transcribir {path.name} ({type(exc).__name__}: {exc}). "
                             "Si el error menciona `metadata_errors`, tienes `av` 16 o superior: este proyecto necesita av<16.") from exc
    return out, info.language, float(info.language_probability), float(info.duration)


def transcribe(path: Path, settings: Settings | None = None) -> Transcript:
    """Transcripción con caché en disco. La primera vez tarda; las siguientes son instantáneas."""
    s = settings or get_settings()
    key = cache_key(file_hash(path), s.whisper_model, s.whisper_language, s.whisper_prompt)
    cached = _cache_file(s, key)
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
        data["segments"] = [Segment(**x) for x in data["segments"]]
        return Transcript(**{**data, "from_cache": True})
    t0 = time.perf_counter()
    segments, language, probability, duration = _transcribe_raw(path, s.whisper_model, s.whisper_language, s.whisper_prompt)
    transcript = Transcript(segments, language, probability, duration, s.whisper_model, round(time.perf_counter() - t0, 1))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(asdict(transcript), ensure_ascii=False), encoding="utf-8")
    return transcript


# ───────────────────────── segmentos → ventanas (Python puro) ─────────────────────────
@dataclass
class Window:
    start: float
    end: float
    text: str


def group_segments(segments: list[Segment], window_s: float = 50.0, overlap_s: float = 8.0) -> list[Window]:
    """Agrupa segmentos consecutivos en ventanas de hasta `window_s` segundos, con un pequeño solape.

    Un segmento de Whisper dura unos segundos: como fragmento es demasiado corto para tener sentido. Una ventana de ~50 s
    (≈ 120 palabras) es un buen fragmento. El solape repite al inicio de cada ventana los últimos `overlap_s` segundos de la
    anterior, para que una idea que cae en el borde no se parta sin remedio. Una ventana siempre trae al menos un segmento
    (uno más largo que `window_s` forma su propia ventana); `start`/`end` son los del primer y el último segmento.
    """
    if window_s <= 0:
        raise ValueError("window_s debe ser positivo")
    if not 0 <= overlap_s < window_s:
        raise ValueError("overlap_s debe estar entre 0 y window_s")
    segs = [x for x in segments if x.text.strip()]
    windows: list[Window] = []
    i = 0
    while i < len(segs):
        j = i + 1
        while j < len(segs) and segs[j].end - segs[i].start <= window_s:
            j += 1
        group = segs[i:j]
        windows.append(Window(group[0].start, group[-1].end, " ".join(x.text.strip() for x in group)))
        if j >= len(segs):
            break
        k = j                                                      # el solape: retroceder mientras el segmento caiga en los
        while k - 1 > i and segs[k - 1].start >= group[-1].end - overlap_s:   # últimos `overlap_s` segundos de la ventana
            k -= 1
        i = k                                                      # siempre k > i: la ventana siguiente avanza
    return windows


# ───────────────────────── lectura → Document ─────────────────────────
def read_media(path: Path, settings: Settings | None = None) -> Document:
    s = settings or get_settings()
    kind = kind_of(path)
    title = path.stem.replace("_", " ").strip()
    try:
        t0 = time.perf_counter()
        transcript = transcribe(path, s)
        elapsed = time.perf_counter() - t0
    except MediaError as exc:        # incluye MediaDepsMissing: un archivo con problemas no tumba toda la ingesta
        return Document(path.name, title, kind, [], [f"{kind.capitalize()} omitido. {exc}"], skipped=True)
    windows = group_segments(transcript.segments, s.media_window_s, s.media_overlap_s)
    sections = [(format_range(w.start, w.end), w.text) for w in windows]
    metas = [{"media": path.name, "start_s": round(w.start, 2), "end_s": round(w.end, 2)} for w in windows]
    warnings = list(transcript.warnings)
    if not windows:
        warnings.append("No se detectó voz en el archivo: no hay nada que indexar.")
    else:
        warnings.append(f"Transcripción automática ({transcript.model}): puede equivocarse con nombres propios, siglas y "
                        "vocabulario técnico; revísala antes de confiar en ella.")
    if s.frame_captions and kind == "video":                       # EXPERIMENTAL: ver frames.py
        try:
            from . import frames

            captions = frames.caption_video(path, transcript.duration, file_hash(path), s)
            extra_sections, extra_metas = frames.captions_to_sections(captions, path.name, s.frame_interval_s)
            sections, metas = sections + extra_sections, metas + extra_metas
            warnings.append(f"EXPERIMENTAL: {len(captions)} fotogramas descritos con {s.vision_provider}/{s.vision_model}; "
                            "un modelo con visión lee mal la escritura manuscrita a baja resolución: revísalos.")
        except MediaError as exc:
            warnings.append(f"Fotogramas omitidos: {exc}")
    info = {"language": transcript.language, "language_probability": round(transcript.language_probability, 2),
            "duration_s": round(transcript.duration, 1), "model": transcript.model, "segments": len(transcript.segments),
            "seconds": round(elapsed, 1), "transcribe_seconds": transcript.seconds, "from_cache": transcript.from_cache}
    return Document(path.name, title, kind, sections, warnings, metas, info)


def load_media(folder: Path, settings: Settings | None = None) -> list[Document]:
    """Todos los audios y videos de la carpeta (que puede no existir: entonces no hay medios)."""
    return [read_media(p, settings) for p in list_media(folder)]
