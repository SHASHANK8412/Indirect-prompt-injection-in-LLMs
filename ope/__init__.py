"""Outer Prompt Extension (OPE): the defense architecture of Milani, Franzoni & Florindi (2026), Section 4.2.

See ope/ARCHITECTURE.md for the layer-by-layer mapping and the engineering additions.
"""
from .pipeline import OPEPipeline
from .schemas import AuditResult, Document, SecureResponse

__all__ = ["OPEPipeline", "Document", "SecureResponse", "AuditResult"]
