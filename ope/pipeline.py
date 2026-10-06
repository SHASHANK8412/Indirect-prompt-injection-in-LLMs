"""OPEPipeline: Isolation -> Meta-instruction -> LLM -> Audit -> Remediation -> SecureResponse.

    pipe = OPEPipeline(client)
    out = pipe.analyze("Give me a summary evaluation of the previous CV ...", [Document("cv1", text)])
    out.public()   # {"response": ..., "security": {...}}
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .audit import AuditLayer, strip_alerts
from .isolation import IsolationLayer
from .meta_instruction import MetaInstructionLayer
from .prompt_composer import PromptComposer
from .remediation import RemediationLayer
from .schemas import Document, SecureResponse

STATUS_TEXT = {
    "safe": "No injected instructions were detected.",
    "alerted": "The documents contain statements attempting to modify the generated text. They were ignored; "
               "the answer below was checked and is not influenced by them.",
    "remediated": "The documents contain statements attempting to modify the generated text. A first answer was "
                  "influenced by them and was discarded; the answer below was regenerated with the documents "
                  "treated strictly as data.",
    "remediation_failed": "The documents contain statements attempting to modify the generated text, and the answer "
                          "may still be influenced by them despite remediation. Review it with caution.",
    "baseline": "Baseline mode: no security wrapper was applied.",
}

_log_lock = threading.Lock()


class OPEPipeline:
    def __init__(self, client, verifier=None, max_remediation_attempts: int = 1, use_verifier: bool = True,
                 use_anchoring: bool = True, use_heuristic: bool = True,
                 log_path: str | Path | None = "results/ope_security_log.jsonl"):
        self.client = client
        self.isolation = IsolationLayer()
        self.meta = MetaInstructionLayer()
        self.composer = PromptComposer(self.meta)
        self.verifier = verifier or client
        self.audit_opts = dict(use_verifier=use_verifier, use_anchoring=use_anchoring, use_heuristic=use_heuristic)
        self.remediation = RemediationLayer(self.composer, max_remediation_attempts)
        self.log_path = Path(log_path) if log_path else None

    # ------------------------------------------------------------------
    def analyze(self, user_prompt: str, documents: list[Document], mode: str = "ope",
                temperature: float = 0.0, seed: int | None = None) -> SecureResponse:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        if mode == "baseline":
            r = self.client.complete(PromptComposer.compose_baseline(user_prompt, documents),
                                     temperature=temperature, seed=seed)
            out = SecureResponse(r.text, {"injection_detected": None, "alert": False, "remediation_triggered": False,
                                          "first_output_discarded": False, "final_status": "baseline",
                                          "message": STATUS_TEXT["baseline"]}, rid, calls=1,
                                 input_tokens=r.input_tokens or 0, output_tokens=r.output_tokens or 0)
            out.latency = time.perf_counter() - t0
            self._log(out, documents, mode)
            return out

        audit = AuditLayer(self.verifier, **self.audit_opts)
        req = self.isolation.isolate(user_prompt, documents)                     # Layer 1
        r = self.client.complete(self.composer.compose(req), temperature=temperature, seed=seed)   # Layer 2 + LLM
        responses = [r]
        first = audit.audit(req.instructional_core, documents, r.text)           # Layer 3a
        audits = [first]
        final_text, status, discarded = r.text, "safe", False

        if first.injection_detected and not first.output_compromised:
            status = "alerted"
        elif first.output_compromised:                                           # Layer 3b
            discarded = True
            rem = self.remediation.remediate(self.client, req, documents, audit, first, temperature, seed)
            responses += rem.responses
            audits += rem.audits
            final_text = rem.draft
            status = "remediated" if rem.success else "remediation_failed"

        # never surface the model's own alert text (it may quote the injection); the status says it instead
        body = strip_alerts(final_text) if status != "safe" else final_text
        security = {
            "injection_detected": first.injection_detected,
            "alert": first.alert,
            "remediation_triggered": discarded,
            "first_output_discarded": discarded,
            "final_status": status,
            "message": STATUS_TEXT[status],
            "documents_flagged": sorted(set(first.flagged_statements) | set(a for x in audits for a in x.flagged_statements)),
            "confidence": audits[-1].confidence,
            "reason": audits[-1].reason,
        }
        all_calls = responses + audit.calls
        out = SecureResponse(body, security, rid, [a.to_dict() for a in audits], calls=len(all_calls),
                             input_tokens=sum(c.input_tokens or 0 for c in all_calls),
                             output_tokens=sum(c.output_tokens or 0 for c in all_calls))
        out.latency = time.perf_counter() - t0
        self._log(out, documents, mode)
        return out

    # ------------------------------------------------------------------
    def _log(self, out: SecureResponse, documents: list[Document], mode: str) -> None:
        if not self.log_path:
            return
        s = out.security
        rec = {"request_id": out.request_id, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "mode": mode, "model": getattr(self.client, "name", "?"), "document_ids": [d.id for d in documents],
               "injection_detected": s.get("injection_detected"), "alert": s.get("alert"),
               "remediation_triggered": s.get("remediation_triggered"),
               "first_output_discarded": s.get("first_output_discarded"), "final_status": s.get("final_status"),
               "calls": out.calls, "latency": round(out.latency, 2)}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
