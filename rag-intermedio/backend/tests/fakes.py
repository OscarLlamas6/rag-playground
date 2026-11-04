"""Dobles de Docling y de faster-whisper: la suite por defecto no descarga modelos ni necesita el grupo `media`."""

from __future__ import annotations

from pathlib import Path

from app import media, pdf

REAL_RUN_DOCLING = pdf._run_docling              # el verdadero, por si una prueba quiere comprobar la degradación sin el grupo `media`
REAL_TRANSCRIBE_RAW = media._transcribe_raw
REAL_HYBRID = pdf._native_hybrid_pieces


class FakeDocling:
    """Sustituye a `pdf._run_docling`. Los PDF cuyo nombre contiene 'escaneada' no tienen capa de texto: solo el OCR los lee."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, int]] = []
        self.total_pages: dict[str, int] = {}          # nombre → páginas del archivo (por defecto 3)

    def __call__(self, path: Path, ocr: bool, max_pages: int, settings) -> pdf.RawPdf:
        self.calls.append((path.name, ocr, max_pages))
        total = self.total_pages.get(path.name, 1 if "escaneada" in path.name else 3)
        read = min(total, max_pages) if max_pages > 0 else total
        items: list[pdf.PdfItem] = []
        if "escaneada" not in path.name or ocr:
            stem = path.stem
            items.append(pdf.PdfItem("heading", f"Informe {stem}", 1, 1))
            for page in range(1, read + 1):
                items.append(pdf.PdfItem("text", f"Contenido de la página {page} de {stem}: tema{page} y datos del equipo AX-{page}00.", page))
        return pdf.RawPdf(items, read, total, 0.1, ocr, native={"fake": path.name})


def fake_hybrid_pieces(native, tokenizer: str, max_tokens: int) -> list[pdf.HybridPiece]:
    """Un fragmento por página, con encabezado (lo que haría el HybridChunker, sin Docling)."""
    name = native["fake"]
    return [pdf.HybridPiece(f"Informe {name}\nContenido de la página {p} de {name}", [f"Informe {name}"], p, None) for p in (1, 2)]


class FakeWhisper:
    """Sustituye a `media._transcribe_raw`: 8 segmentos de 15 s (2 minutos) con texto reconocible."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, path: Path, model: str, language: str | None, prompt: str | None):
        self.calls.append((path.name, model, language))
        segments = [media.Segment(i * 15.0, i * 15.0 + 14.0, f"Parte {i} de {path.stem}: el servidor{i} reinicia el servicio.")
                    for i in range(8)]
        return segments, language or "en", 0.97, 120.0
