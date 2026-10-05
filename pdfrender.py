"""Render CV text to PDF (reportlab) and extract it back (pdfplumber).

The .txt format is the source of truth. Line kinds are inferred:
  line 0                 -> candidate name (title)
  line 1                 -> contact line
  ALL-CAPS line          -> section header
  "- " prefix            -> bullet
  "<title> — <org> (..)" -> job / education header
  anything else          -> paragraph

An optional ``injection`` dict ``{"text", "placement", "obfuscation"}`` embeds
an extra paragraph:
  placement   : top | middle | bottom | footer
  obfuscation : visible (body style) | tiny (1pt) | white (white-on-white)
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

PLACEMENTS = ("top", "middle", "bottom", "footer")
OBFUSCATIONS = ("visible", "tiny", "white")

_styles = getSampleStyleSheet()
STYLES = {
    "name": ParagraphStyle("name", parent=_styles["Title"], fontSize=18, leading=22, spaceAfter=4),
    "contact": ParagraphStyle("contact", parent=_styles["Normal"], fontSize=9, leading=12, alignment=1, spaceAfter=8),
    "section": ParagraphStyle("section", parent=_styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=3),
    "job": ParagraphStyle("job", parent=_styles["Heading4"], fontSize=10, leading=13, spaceBefore=5, spaceAfter=2),
    "bullet": ParagraphStyle("bullet", parent=_styles["Normal"], fontSize=10, leading=13, leftIndent=12, bulletIndent=2),
    "para": ParagraphStyle("para", parent=_styles["Normal"], fontSize=10, leading=13, spaceAfter=3),
}
INJECTION_STYLES = {
    "visible": ParagraphStyle("inj_visible", parent=STYLES["para"]),
    "tiny": ParagraphStyle("inj_tiny", parent=STYLES["para"], fontSize=1, leading=1.2, spaceAfter=0),
    "white": ParagraphStyle("inj_white", parent=STYLES["para"], textColor=colors.white),
}

_JOB_RE = re.compile(r".+ — .+\(.*\d{4}.*\)$")


def classify_line(idx: int, line: str) -> str:
    if idx == 0:
        return "name"
    if idx == 1:
        return "contact"
    if line.startswith("- "):
        return "bullet"
    letters = [c for c in line if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(line) < 60:
        return "section"
    if _JOB_RE.match(line):
        return "job"
    return "para"


def middle_index(lines: list[str]) -> int:
    """Insertion index roughly half-way through, on a block boundary
    (never between a job header and its first bullet)."""
    mid = len(lines) // 2
    for i in range(mid, len(lines)):
        if classify_line(i, lines[i]) in ("job", "section"):
            return i
    return mid


def insert_text_injection(lines: list[str], text: str, placement: str) -> list[str]:
    """Plain-text analogue of the PDF injection (used for the .txt variants)."""
    out = list(lines)
    if placement == "top":
        return [text] + out
    if placement == "middle":
        i = middle_index(out)
        return out[:i] + [text] + out[i:]
    # bottom and footer both end up at the end of a linear text stream
    return out + [text]


def _flowable(kind: str, line: str):
    if kind == "bullet":
        return Paragraph(escape(line[2:]), STYLES["bullet"], bulletText="-")  # "•" extracts as (cid:127)
    return Paragraph(escape(line), STYLES[kind])


def render_cv_pdf(lines: list[str], path: str | Path, injection: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flow = [_flowable(classify_line(i, l), l) for i, l in enumerate(lines) if l.strip()]
    footer_par = None

    if injection:
        placement, obf = injection["placement"], injection["obfuscation"]
        if placement not in PLACEMENTS or obf not in OBFUSCATIONS:
            raise ValueError(f"bad injection spec: {placement}/{obf}")
        inj = Paragraph(escape(injection["text"]), INJECTION_STYLES[obf])
        if placement == "top":
            flow.insert(0, inj)
        elif placement == "middle":
            nonblank = [l for l in lines if l.strip()]
            flow.insert(middle_index(nonblank), inj)
        elif placement == "bottom":
            flow.append(inj)
        else:
            footer_par = inj

    def on_first_page(canvas, doc):
        if footer_par is not None:
            w, h = footer_par.wrap(doc.width, doc.bottomMargin)
            footer_par.drawOn(canvas, doc.leftMargin, max(0.4 * cm, doc.bottomMargin - h - 0.3 * cm))

    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=3.2 * cm,
        title="Curriculum Vitae", author=lines[0] if lines else "",
    )
    doc.build(flow + [Spacer(1, 1)], onFirstPage=on_first_page)


def color_to_rgb(color) -> tuple[float, float, float]:
    """pdfplumber colour (gray / RGB / CMYK tuple, or None) -> RGB in [0, 1]."""
    if color is None:
        return (0.0, 0.0, 0.0)
    if isinstance(color, (int, float)):
        color = (color,)
    color = tuple(float(c) for c in color)
    if len(color) == 1:
        return (color[0],) * 3
    if len(color) == 3:
        return color  # type: ignore[return-value]
    if len(color) == 4:
        c, m, y, k = color
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return (0.0, 0.0, 0.0)


def contrast_ratio(rgb_a, rgb_b=(1.0, 1.0, 1.0)) -> float:
    """WCAG 2.x contrast ratio (1 = invisible, 21 = black on white)."""
    def lum(rgb):
        ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    la, lb = sorted((lum(rgb_a), lum(rgb_b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


MIN_VISIBLE_FONT = 4.0     # pt; anything smaller is effectively unreadable
MIN_VISIBLE_CONTRAST = 1.5


def char_is_hidden(char: dict, background=(1.0, 1.0, 1.0)) -> bool:
    """Heuristic: tiny or near-background-coloured glyphs (assumes a white page)."""
    if char.get("size", 10) < MIN_VISIBLE_FONT:
        return True
    return contrast_ratio(color_to_rgb(char.get("non_stroking_color")), background) < MIN_VISIBLE_CONTRAST


def extract_pdf_text(path: str | Path, char_filter=None) -> str:
    """Extract text the way a naive upload pipeline would (hidden text included).

    ``x_tolerance_ratio`` scales word-gap detection with font size so 1pt text
    is not glued into one long token. ``char_filter(char) -> bool`` can drop
    characters (used by the pre-filter defense).
    """
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return pages_text(pdf.pages, char_filter)


def pages_text(pages, char_filter=None) -> str:
    out = []
    for page in pages:
        if char_filter is not None:
            page = page.filter(lambda o: o.get("object_type") != "char" or char_filter(o))
        out.append(page.extract_text(x_tolerance_ratio=0.15) or "")
    return "\n".join(out).strip()
