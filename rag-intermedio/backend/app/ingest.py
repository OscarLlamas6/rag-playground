"""Ingesta multiformato: leer Markdown, HTML y Word y convertirlos en SECCIONES de texto limpio.

Un RAG real casi nunca recibe un .md ordenado: recibe páginas web, contratos en Word, boletines en PDF. El primer
paso (y el que más se subestima) es EXTRAER el texto conservando su estructura. Si aquí entra basura, todo lo que
sigue (fragmentar, vectorizar, recuperar) trabaja con basura.

    .md    separa por encabezados `## `
    .html  quita lo que NO es contenido (<script>, <style>, menús, pie de página) y separa por <h2>
    .docx  respeta los estilos de título (Heading 1/2) y convierte las TABLAS en líneas de texto

PDF: este playground no lo incluye. Un PDF con texto se lee igual (una sección por página con `pypdf`); un PDF
escaneado es una imagen y necesita OCR antes de entrar al RAG.
"""

from __future__ import annotations

from pathlib import Path

from .domain import Document

EXTENSIONS = (".md", ".html", ".htm", ".docx")


def _from_markdown(path: Path) -> Document:
    title = path.stem.replace("_", " ").capitalize()
    sections: list[tuple[str, str]] = []
    heading, buffer = "Introducción", []

    def close() -> None:
        if "".join(buffer).strip():
            sections.append((heading, "\n".join(buffer).strip()))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            close()
            heading, buffer = line[3:].strip(), []
        elif line.startswith("# "):
            title = line[2:].strip()
        else:
            buffer.append(line)
    close()
    return Document(path.name, title, "md", sections)


def _from_html(path: Path) -> Document:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else path.stem)
    # 1) quitar el "ruido": no es contenido, y si se queda se vectoriza como si lo fuera
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.body or soup

    # 2) recorrer los elementos EN ORDEN y abrir una sección nueva en cada <h2>
    sections: list[tuple[str, str]] = []
    heading, buffer = "Introducción", []

    def close() -> None:
        if "".join(buffer).strip():
            sections.append((heading, "\n".join(buffer).strip()))

    for el in body.find_all(["h1", "h2", "h3", "p", "li"]):
        text = " ".join(el.get_text().split())
        if not text or el.name == "h1":
            continue
        if el.name in ("h2", "h3"):
            close()
            heading, buffer = text, []
        else:
            buffer.append(("- " + text) if el.name == "li" else text)
    close()
    return Document(path.name, title, "html", sections)


def _from_docx(path: Path) -> Document:
    from docx import Document as Docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Docx(str(path))
    title, h1 = path.stem.replace("_", " ").capitalize(), ""
    heading, buffer = "Introducción", []
    sections: list[tuple[str, str]] = []

    def close() -> None:
        if "".join(buffer).strip():
            sections.append((heading, "\n".join(buffer).strip()))

    for child in doc.element.body.iterchildren():          # párrafos y tablas, en el orden del documento
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            text, style = p.text.strip(), (p.style.name if p.style is not None else "")
            if not text:
                continue
            if style == "Title":
                title = text
            elif style == "Heading 1":
                close()
                h1, heading, buffer = text, text, []
            elif style == "Heading 2":
                close()
                heading, buffer = (f"{h1} › {text}" if h1 else text), []
            else:
                buffer.append(text)
        elif child.tag.endswith("}tbl"):
            # una tabla no es texto corrido: una línea por fila ("celda | celda | celda")
            for row in Table(child, doc).rows:
                buffer.append(" | ".join(c.text.strip() for c in row.cells))
    close()
    return Document(path.name, title, "docx", sections)


def read_document(path: Path) -> Document:
    readers = {".md": _from_markdown, ".html": _from_html, ".htm": _from_html, ".docx": _from_docx}
    ext = path.suffix.lower()
    if ext not in readers:
        raise ValueError(f"Formato no soportado: {ext}. Usa: {', '.join(EXTENSIONS)}")
    doc = readers[ext](path)
    if not doc.sections:
        doc.warnings.append("No se extrajo ninguna sección (¿archivo vacío o escaneado?).")
    return doc


def load_documents(folder: Path) -> list[Document]:
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No hay documentos {', '.join(EXTENSIONS)} en {folder}")
    return [read_document(p) for p in files]
