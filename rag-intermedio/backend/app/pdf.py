"""PDF con Docling: texto con estructura, OCR automático y página en cada sección.

    1) Se convierte SIN OCR (rápido: lee la capa de texto y detecta títulos, tablas y listas).
    2) Si casi no salió texto (PDF escaneado, o diseño tipo lámina que Docling trata como imagen), se REINTENTA con OCR.
    3) El resultado se guarda en disco (por hash del archivo + opciones): volver a ingestar es instantáneo.

Todo lo que toca a Docling vive en `_run_docling` y `_native_hybrid_pieces`; el resto (decidir si hace falta OCR, armar
secciones, citar páginas) es Python puro y se prueba con un Docling simulado. Docling y torch son OPCIONALES (grupo
`media`) y se importan solo al convertir.

Qué NO hace por ti: el OCR comete errores ("OpenAl" por "OpenAI"), las tablas complejas pueden salir desordenadas y un tope de
páginas (DOCLING_MAX_PAGES) deja fuera el resto del documento. Por eso cada archivo reporta si usó OCR, cuántas páginas leyó
y qué advertencias dejó.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .domain import Document
from .extras import INSTALL_HINT, MediaDepsMissing, MediaError, require

log = logging.getLogger(__name__)

EXTENSIONS = (".pdf",)
EMPTY_PAGE_SHARE = 0.5          # si la mitad o más de las páginas casi no tienen texto, se intenta con OCR
_CACHE_VERSION = "v1"


@dataclass
class PdfItem:
    """Un trozo de contenido del PDF, ya sin la maquinaria de Docling: encabezado, párrafo/lista o tabla, con su página."""

    kind: str                   # "heading" | "text" | "table"
    text: str
    page: int | None = None
    level: int = 1              # solo para encabezados (1 = el más alto)


@dataclass
class RawPdf:
    """Lo que devuelve una conversión: los items, cuántas páginas se leyeron de cuántas hay, y el objeto nativo."""

    items: list[PdfItem]
    pages_read: int
    pages_total: int
    seconds: float
    ocr: bool
    native: object = None       # DoclingDocument (para el HybridChunker)
    from_cache: bool = False


@dataclass
class HybridPiece:
    """Un fragmento del HybridChunker: `text` ya viene contextualizado (encabezados + texto), que es lo que se vectoriza."""

    text: str
    headings: list[str] = field(default_factory=list)
    page: int | None = None
    page_end: int | None = None


# ───────────────────────── decisión de OCR (Python puro) ─────────────────────────
def page_chars(items: list[PdfItem]) -> dict[int, int]:
    """Caracteres de texto por página (solo páginas con alguna posición conocida)."""
    counts: dict[int, int] = {}
    for item in items:
        if item.page is not None:
            counts[item.page] = counts.get(item.page, 0) + len(item.text.strip())
    return counts


def needs_ocr(items: list[PdfItem], pages_read: int, min_chars: int = 40) -> tuple[bool, str]:
    """¿Conviene reintentar con OCR? Sí si la mitad o más de las páginas leídas traen menos de `min_chars` caracteres.

    Se decide por PÁGINA y no por documento: un informe de 100 páginas con la portada en imagen no necesita OCR, pero uno cuyas
    páginas son casi todas imagen sí. Devuelve (decisión, motivo).
    """
    if pages_read <= 0:
        return True, "no se leyó ninguna página"
    counts = page_chars(items)
    empty = sum(1 for page in range(1, pages_read + 1) if counts.get(page, 0) < min_chars)
    share = empty / pages_read
    reason = f"{empty} de {pages_read} páginas con menos de {min_chars} caracteres"
    return share >= EMPTY_PAGE_SHARE, reason


# ───────────────────────── secciones con página (Python puro) ─────────────────────────
def items_to_sections(items: list[PdfItem], fallback_title: str) -> tuple[list[tuple[str, str]], list[dict]]:
    """Agrupa los items en secciones: cada encabezado abre una. El título es la ruta de encabezados ('Capítulo › Apartado').

    Devuelve las secciones y, en paralelo, su posición: `page` (donde empieza) y `page_end` (donde termina) si difiere.
    """
    sections: list[tuple[str, str]] = []
    metas: list[dict] = []
    stack: list[tuple[int, str]] = []
    heading = fallback_title
    buffer: list[str] = []
    pages: list[int] = []

    def close() -> None:
        body = "\n".join(buffer).strip()
        if body:
            meta: dict = {}
            if pages:
                meta["page"] = min(pages)
                if max(pages) != min(pages):
                    meta["page_end"] = max(pages)
            sections.append((heading, body))
            metas.append(meta)

    for item in items:
        text = item.text.strip()
        if not text:
            continue
        if item.kind == "heading":
            close()
            while stack and stack[-1][0] >= item.level:
                stack.pop()
            stack.append((item.level, " ".join(text.split())))
            heading = " › ".join(t for _, t in stack)
            buffer, pages = [], ([item.page] if item.page else [])
        else:
            buffer.append(text)
            if item.page:
                pages.append(item.page)
    close()
    return sections, metas


def build_document(path: Path, raw: RawPdf, warnings: list[str], info: dict) -> Document:
    title = path.stem.replace("_", " ").capitalize()
    first = next((i.text for i in raw.items if i.kind == "heading"), "")
    sections, metas = items_to_sections(raw.items, fallback_title=first or title)
    return Document(path.name, title, "pdf", sections, warnings, metas, info, native=raw.native)


# ───────────────────────── lectura (con la decisión de OCR) ─────────────────────────
def read_pdf(path: Path, settings: Settings | None = None) -> Document:
    """Lee un PDF: primero sin OCR y, si casi no hay texto, con OCR. Si falta el grupo `media`, devuelve un documento omitido."""
    s = settings or get_settings()
    t0 = time.perf_counter()
    warnings: list[str] = []
    try:
        raw = _run_docling(path, ocr=(s.docling_ocr == "force"), max_pages=s.docling_max_pages, settings=s)
        used_ocr, reason = raw.ocr, ""
        if s.docling_ocr == "auto" and not raw.ocr:
            retry, reason = needs_ocr(raw.items, raw.pages_read, s.docling_min_chars)
            if retry:
                warnings.append(f"Sin texto ({reason}): se aplicó OCR.")
                raw = _run_docling(path, ocr=True, max_pages=s.docling_max_pages, settings=s)
                used_ocr = True
    except MediaDepsMissing as exc:
        doc = Document(path.name, path.stem, "pdf", [], [f"PDF omitido. {exc}"], skipped=True)
        return doc
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - un PDF dañado no debe tumbar toda la ingesta
        log.warning("Docling falló con %s: %s", path.name, exc)
        return Document(path.name, path.stem, "pdf", [], [f"PDF omitido: Docling no pudo leerlo ({type(exc).__name__}: {exc})."],
                        skipped=True)

    if used_ocr:
        warnings.append("Texto obtenido por OCR: puede tener errores (p. ej. «OpenAl» por «OpenAI»); revísalo antes de confiar en él.")
        if not raw.items:
            warnings.append("Ni siquiera con OCR salió texto.")
    elif not raw.items:
        warnings.append("No salió texto y el OCR no se intentó (DOCLING_OCR=off).")
    if raw.pages_read < raw.pages_total:
        warnings.append(f"Solo se leyeron las primeras {raw.pages_read} de {raw.pages_total} páginas "
                        f"(tope DOCLING_MAX_PAGES={s.docling_max_pages}); el resto no se indexa.")
    info = {"ocr": used_ocr, "pages": raw.pages_read, "pages_total": raw.pages_total,
            "seconds": round(time.perf_counter() - t0, 1), "from_cache": raw.from_cache}
    return build_document(path, raw, warnings, info)


# ───────────────────────── Docling de verdad (perezoso) ─────────────────────────
_converters: dict[bool, object] = {}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _converter(ocr: bool):
    """Un DocumentConverter por modo (cargar los modelos cuesta segundos: se reutiliza)."""
    if ocr not in _converters:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = ocr
        _converters[ocr] = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return _converters[ocr]


def table_to_lines(markdown: str) -> str:
    """Una tabla en Markdown ('| a | b |' + fila de guiones) pasa a una línea por fila ('a | b'), como hace el lector de Word."""
    rows = []
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not any(cells) or all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _label(item) -> str:
    return str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", "")))


def extract_items(dl_doc) -> list[PdfItem]:
    """Recorre el DoclingDocument en orden de lectura y lo reduce a encabezados, texto y tablas con su página."""
    items: list[PdfItem] = []
    for node, _depth in dl_doc.iterate_items():
        label = _label(node)
        page = node.prov[0].page_no if getattr(node, "prov", None) else None
        if label == "title":
            items.append(PdfItem("heading", node.text, page, 1))
        elif label == "section_header":
            items.append(PdfItem("heading", node.text, page, int(getattr(node, "level", 1)) + 1))
        elif label == "table":
            try:
                text = table_to_lines(node.export_to_markdown(doc=dl_doc))
            except Exception:  # noqa: BLE001
                text = getattr(node, "text", "") or ""
            items.append(PdfItem("table", text, page))
        elif label in ("picture", "chart"):
            continue
        else:
            text = getattr(node, "text", "") or ""
            if text.strip():
                items.append(PdfItem("text", ("- " + text) if label == "list_item" else text, page))
    return items


def _run_docling(path: Path, ocr: bool, max_pages: int, settings: Settings) -> RawPdf:
    """Convierte con Docling (o recupera la conversión guardada en disco)."""
    require("Leer PDF con Docling", "docling")
    from docling_core.types.doc import DoclingDocument

    cache_dir = settings.media_cache_dir / "docling"
    key = f"{_file_hash(path)[:24]}-{_CACHE_VERSION}-p{max_pages}-ocr{int(ocr)}"
    body, meta_file = cache_dir / f"{key}.json", cache_dir / f"{key}.meta.json"
    if body.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        doc = DoclingDocument.load_from_json(body)
        return RawPdf(extract_items(doc), meta["pages_read"], meta["pages_total"], meta["seconds"], ocr, doc, from_cache=True)

    t0 = time.perf_counter()
    kwargs = {"page_range": (1, max_pages)} if max_pages > 0 else {}
    result = _converter(ocr).convert(path, **kwargs)
    doc = result.document
    pages_read = len(doc.pages)
    pages_total = getattr(getattr(result, "input", None), "page_count", None) or pages_read
    seconds = time.perf_counter() - t0
    items = extract_items(doc)
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(body)
    meta_file.write_text(json.dumps({"pages_read": pages_read, "pages_total": pages_total, "seconds": round(seconds, 1)}), encoding="utf-8")
    return RawPdf(items, pages_read, pages_total, seconds, ocr, doc)


# ───────────────────────── HybridChunker (perezoso) ─────────────────────────
def hybrid_chunks(doc: Document, tokenizer: str, max_tokens: int) -> list[HybridPiece]:
    """Fragmenta con el HybridChunker de Docling: respeta la jerarquía del documento y parte lo que pasa de `max_tokens`.

    OJO con el nombre: "híbrido" aquí NO es la búsqueda híbrida (BM25 + vectores). Es chunking híbrido = estructura del
    documento (encabezados, tablas, listas) + límite de tokens.
    """
    if doc.native is None:
        raise MediaError(f"{doc.source}: no hay documento de Docling para fragmentar.")
    return _native_hybrid_pieces(doc.native, tokenizer, max_tokens)


def _native_hybrid_pieces(native, tokenizer: str, max_tokens: int) -> list[HybridPiece]:
    require("El chunking docling_hybrid", "docling", "transformers")
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    try:
        hf = AutoTokenizer.from_pretrained(tokenizer)
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"No se pudo cargar el tokenizer '{tokenizer}' ({type(exc).__name__}). La primera vez se descarga de "
                         "Hugging Face: revisa tu conexión, o cambia DOCLING_TOKENIZER por uno que ya tengas.") from exc
    chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=hf, max_tokens=max_tokens))
    pieces: list[HybridPiece] = []
    for chunk in chunker.chunk(dl_doc=native):
        pages = sorted({p.page_no for it in chunk.meta.doc_items for p in getattr(it, "prov", [])})
        pieces.append(HybridPiece(chunker.contextualize(chunk), list(chunk.meta.headings or []),
                                  pages[0] if pages else None, pages[-1] if len(pages) > 1 else None))
    return pieces
