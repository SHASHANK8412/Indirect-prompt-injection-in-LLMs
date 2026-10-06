"""Layer 1 - Isolation.

Separates the user's INSTRUCTIONAL_CORE from UNTRUSTED_CONTENT and keeps every
document individually identifiable. Document text is data; any boundary marker
that appears inside a document is escaped so the document cannot close its own
boundary and pose as part of the instruction core (engineering addition 5).
"""
from __future__ import annotations

import re

from .schemas import Document, IsolatedDocument, IsolatedRequest

BOUNDARY_TAGS = ("INSTRUCTIONAL_CORE", "UNTRUSTED_CONTENT", "DOCUMENT", "SECURITY_META", "SECURITY_VERIFICATION",
                 "SECURITY_NOTES")
_TAG_RE = re.compile(r"<\s*(/?)\s*(" + "|".join(BOUNDARY_TAGS) + r")\b([^>]*)>", re.I)


def escape_boundaries(text: str) -> str:
    """Turn '<UNTRUSTED_CONTENT>' etc. inside data into an inert '[escaped-tag ...]'."""
    return _TAG_RE.sub(lambda m: f"[escaped-tag {m.group(1)}{m.group(2).upper()}]", text)


def _safe_attr(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _.\-]", "_", value)[:80]


class IsolationLayer:
    def isolate(self, user_prompt: str, documents: list[Document]) -> IsolatedRequest:
        if not user_prompt.strip():
            raise ValueError("user prompt is empty")
        ids = [d.id for d in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("document ids must be unique")
        return IsolatedRequest(
            instructional_core=escape_boundaries(user_prompt.strip()),
            documents=[IsolatedDocument(_safe_attr(d.id), _safe_attr(d.name or d.id), escape_boundaries(d.content.strip()))
                       for d in documents],
        )

    @staticmethod
    def render_core(req: IsolatedRequest) -> str:
        return f"<INSTRUCTIONAL_CORE>\n{req.instructional_core}\n</INSTRUCTIONAL_CORE>"

    @staticmethod
    def render_content(req: IsolatedRequest) -> str:
        blocks = [f'<DOCUMENT id="{d.id}" name="{d.name}">\n{d.content}\n</DOCUMENT>' for d in req.documents]
        return "<UNTRUSTED_CONTENT>\n" + "\n\n".join(blocks) + "\n</UNTRUSTED_CONTENT>"
