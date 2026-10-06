"""Data structures shared by the OPE layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

FinalAction = Literal["accept", "alert", "remediate"]
FinalStatus = Literal["safe", "alerted", "remediated", "remediation_failed", "baseline"]


@dataclass
class Document:
    id: str
    content: str
    name: str = ""
    source: str = "text"            # text | pdf | docx | txt


@dataclass
class IsolatedDocument:
    id: str
    name: str
    content: str                    # boundary markers escaped


@dataclass
class IsolatedRequest:
    """Output of the Isolation Layer: instructional core kept apart from untrusted content."""
    instructional_core: str
    documents: list[IsolatedDocument]


@dataclass
class AuditResult:
    injection_detected: bool = False
    alert: bool = False
    output_compromised: bool = False
    remediation_required: bool = False
    confidence: float = 0.0         # engineering addition: share of audit signals that agree
    reason: str = ""
    final_action: FinalAction = "accept"
    # evidence (kept server-side; not shown to end users verbatim)
    model_alert: bool = False
    flagged_statements: dict[str, list[str]] = field(default_factory=dict)
    verifier_verdict: bool | None = None
    anchored_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SecureResponse:
    response: str
    security: dict
    request_id: str = ""
    audits: list[dict] = field(default_factory=list)   # one per audited draft
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0

    def public(self, include_evidence: bool = False) -> dict:
        """What the API returns. Hidden prompts are never included."""
        out = {"request_id": self.request_id, "response": self.response, "security": self.security}
        if include_evidence:
            out["audits"] = self.audits
        return out
