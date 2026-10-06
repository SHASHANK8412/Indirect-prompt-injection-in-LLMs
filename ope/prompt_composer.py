"""Prompt composer: SECURITY_META + INSTRUCTIONAL_CORE + UNTRUSTED_CONTENT + SECURITY_VERIFICATION.

The only place where prompt text is assembled. The result is a provider-neutral
chat message list consumed by any LLMClient.
"""
from __future__ import annotations

from .isolation import IsolationLayer
from .meta_instruction import MetaInstructionLayer
from .schemas import IsolatedRequest


class PromptComposer:
    def __init__(self, meta: MetaInstructionLayer | None = None):
        self.meta = meta or MetaInstructionLayer()

    def compose(self, req: IsolatedRequest, security_notes: str | None = None) -> list[dict]:
        parts = [self.meta.header(),                  # outer edge: primacy
                 IsolationLayer.render_core(req),
                 IsolationLayer.render_content(req)]
        if security_notes:
            parts.append(security_notes)
        parts.append(self.meta.footer())              # outer edge: recency
        return [{"role": "user", "content": "\n\n".join(parts)}]

    @staticmethod
    def compose_baseline(user_prompt: str, documents) -> list[dict]:
        """Undefended condition: documents followed by the user prompt, as a plain upload would look."""
        body = "\n\n".join(f"[Attached document: {d.id}]\n{d.content.strip()}\n[End of attached document]"
                           for d in documents)
        return [{"role": "user", "content": f"{body}\n\n{user_prompt}"}]
