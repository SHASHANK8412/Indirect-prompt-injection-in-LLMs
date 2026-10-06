"""Layer 3b - Remediation.

When the audit finds a compromised draft: suppress it, reconstruct a secure prompt
from the ORIGINAL user instruction (documents still strictly data, plus a
quarantined list of the detected statements), re-run the LLM and audit again.
If the bias persists after the allowed attempts, report remediation_failed: the
paper's "aware-but-bypassed" case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audit import AuditLayer
from .prompt_composer import PromptComposer
from .schemas import AuditResult, Document, IsolatedRequest


@dataclass
class RemediationOutcome:
    draft: str
    audit: AuditResult
    success: bool
    attempts: int
    audits: list[AuditResult] = field(default_factory=list)
    responses: list = field(default_factory=list)


class RemediationLayer:
    def __init__(self, composer: PromptComposer, max_attempts: int = 1):
        self.composer = composer
        self.max_attempts = max_attempts

    def remediate(self, client, req: IsolatedRequest, documents: list[Document], audit: AuditLayer,
                  first_audit: AuditResult, temperature: float = 0.0, seed: int | None = None) -> RemediationOutcome:
        flagged = dict(first_audit.flagged_statements)
        audits, responses = [], []
        draft, last = "", first_audit
        for attempt in range(1, self.max_attempts + 1):
            notes = self.composer.meta.remediation_notes(flagged)
            r = client.complete(self.composer.compose(req, security_notes=notes), temperature=temperature, seed=seed)
            responses.append(r)
            draft = r.text
            last = audit.audit(req.instructional_core, documents, draft, flagged=flagged or None)
            audits.append(last)
            if not last.output_compromised:
                return RemediationOutcome(draft, last, True, attempt, audits, responses)
            flagged = flagged or last.flagged_statements
        return RemediationOutcome(draft, last, False, self.max_attempts, audits, responses)
