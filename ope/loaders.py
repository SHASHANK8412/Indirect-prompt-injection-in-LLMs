"""Document loaders: PDF / TXT / DOCX / pasted text -> Document.

PDF text is extracted the way an upload pipeline would (hidden text included):
the OPE must handle it downstream, not by silently dropping it here.
DOCX is read with the standard library (engineering addition 7).
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from .schemas import Document


def docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
        text = text.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pdf_text(data: bytes) -> str:
    import pdfplumber

    from pdfrender import pages_text
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return pages_text(pdf.pages)


def load_bytes(doc_id: str, filename: str, data: bytes) -> Document:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return Document(doc_id, pdf_text(data), filename, "pdf")
    if ext == ".docx":
        return Document(doc_id, docx_text(data), filename, "docx")
    if ext in (".txt", ".md", ".text", ""):
        return Document(doc_id, data.decode("utf-8", errors="replace"), filename, "txt")
    raise ValueError(f"unsupported file type: {ext} (use PDF, DOCX or TXT)")


def load_path(path: str | Path, doc_id: str | None = None) -> Document:
    p = Path(path)
    return load_bytes(doc_id or p.stem, p.name, p.read_bytes())


def from_text(doc_id: str, text: str, name: str = "") -> Document:
    return Document(doc_id, text, name or doc_id, "text")
