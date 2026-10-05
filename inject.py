"""Produce attacked CV variants for a controlled prompt-injection evaluation.

For every CV: plain, +p.Pos, +p.Neg, +p.Pos-Neg, +p.Pos-Med, each as
  .txt            the injection inserted as plain text at the placement
  .pdf            the injection rendered with the chosen obfuscation
  .extracted.txt  what a PDF upload pipeline sees (pdfplumber text)

    python inject.py                                   # default: bottom + white
    python inject.py --placement footer --obfuscation tiny
    python inject.py --all-variants                    # every placement x obfuscation

Metadata goes to <out>/injections.csv, including a verification that hidden
text is machine-readable but visually (near-)invisible.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml

from dataset import load_manifest
from pdfrender import (OBFUSCATIONS, PLACEMENTS, char_is_hidden, color_to_rgb, contrast_ratio,
                       extract_pdf_text, insert_text_injection, pages_text, render_cv_pdf)

ATTACKS = {"pos": "p.Pos", "neg": "p.Neg", "posneg": "p.Pos-Neg", "posmed": "p.Pos-Med"}
TEMPLATE_KEYS = {v: k for k, v in ATTACKS.items()}
FIELDS = ["cv_id", "attack", "template", "direction", "placement", "obfuscation", "txt_path", "pdf_path",
          "extracted_path", "injection_text", "extracted_ok", "hidden_chars", "expected_chars",
          "min_font_size", "max_contrast", "visually_hidden", "verified"]


def load_templates(path: str | Path = "templates.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_template(text: str, name: str) -> str:
    return text.replace("[Name Surname]", name).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def verify_pdf(pdf_path: Path, injection_text: str, obfuscation: str) -> tuple[dict, str]:
    """Check the injection is extractable and (for hidden variants) near-invisible."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        extracted = pages_text(pdf.pages)
        chars = [c for p in pdf.pages for c in p.chars]
    # compare alphanumeric glyphs only (punctuation/space handling varies by renderer)
    hidden = [c for c in chars if char_is_hidden(c) and c["text"].isalnum()]
    expected = len(_norm(injection_text))
    res = {
        "extracted_ok": _norm(injection_text) in _norm(extracted),
        "hidden_chars": len(hidden),
        "expected_chars": expected if obfuscation != "visible" else 0,
        "min_font_size": round(min((c["size"] for c in hidden), default=float("nan")), 2),
        "max_contrast": round(max((contrast_ratio(color_to_rgb(c.get("non_stroking_color"))) for c in hidden),
                                  default=float("nan")), 3),
    }
    if obfuscation == "visible":
        res["visually_hidden"] = False
        res["verified"] = res["extracted_ok"] and len(hidden) == 0
    else:
        # all injected glyphs (and nothing else) should be hidden
        res["visually_hidden"] = abs(len(hidden) - expected) <= max(3, 0.02 * expected)
        res["verified"] = res["extracted_ok"] and res["visually_hidden"]
    return res, extracted


def build_variants(cv_dir="data/cvs", out_dir="data/attacked", templates_path="templates.yaml",
                   placements=("bottom",), obfuscations=("white",), verify=True) -> list[dict]:
    cv_dir, out_dir = Path(cv_dir), Path(out_dir)
    templates = load_templates(templates_path)
    rows: list[dict] = []
    for cv in load_manifest(cv_dir):
        lines = (cv_dir / cv["txt_path"]).read_text(encoding="utf-8").strip().splitlines()
        # plain baseline, run through the same render/extract pipeline
        plain = out_dir / "plain"
        plain.mkdir(parents=True, exist_ok=True)
        (plain / f"{cv['id']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        render_cv_pdf(lines, plain / f"{cv['id']}.pdf")
        (plain / f"{cv['id']}.extracted.txt").write_text(extract_pdf_text(plain / f"{cv['id']}.pdf"), encoding="utf-8")
        rows.append({"cv_id": cv["id"], "attack": "plain", "template": "", "direction": 0, "placement": "",
                     "obfuscation": "", "txt_path": f"plain/{cv['id']}.txt", "pdf_path": f"plain/{cv['id']}.pdf",
                     "extracted_path": f"plain/{cv['id']}.extracted.txt", "injection_text": "", "verified": True})

        for placement in placements:
            for obf in obfuscations:
                sub = out_dir / f"{placement}_{obf}"
                sub.mkdir(parents=True, exist_ok=True)
                for short, key in ATTACKS.items():
                    text = render_template(templates[key]["text"], cv["name"])
                    stem = f"{cv['id']}__{short}"
                    (sub / f"{stem}.txt").write_text("\n".join(insert_text_injection(lines, text, placement)) + "\n",
                                                     encoding="utf-8")
                    pdf = sub / f"{stem}.pdf"
                    render_cv_pdf(lines, pdf, {"text": text, "placement": placement, "obfuscation": obf})
                    row = {"cv_id": cv["id"], "attack": short, "template": key,
                           "direction": templates[key]["direction"], "placement": placement, "obfuscation": obf,
                           "txt_path": f"{sub.name}/{stem}.txt", "pdf_path": f"{sub.name}/{stem}.pdf",
                           "extracted_path": f"{sub.name}/{stem}.extracted.txt", "injection_text": text}
                    if verify:
                        check, extracted = verify_pdf(pdf, text, obf)
                        row.update(check)
                    else:
                        extracted = extract_pdf_text(pdf)
                    (sub / f"{stem}.extracted.txt").write_text(extracted, encoding="utf-8")
                    rows.append(row)

    with open(out_dir / "injections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cvs", default="data/cvs")
    ap.add_argument("--out", default="data/attacked")
    ap.add_argument("--templates", default="templates.yaml")
    ap.add_argument("--placement", choices=PLACEMENTS, default="bottom")
    ap.add_argument("--obfuscation", choices=OBFUSCATIONS, default="white")
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    placements = PLACEMENTS if a.all_variants else (a.placement,)
    obfs = OBFUSCATIONS if a.all_variants else (a.obfuscation,)
    rows = build_variants(a.cvs, a.out, a.templates, placements, obfs, verify=not a.no_verify)
    attacked = [r for r in rows if r["attack"] != "plain"]
    failed = [r for r in attacked if not r.get("verified", True)]
    print(f"wrote {len(rows)} variants ({len(attacked)} attacked) to {a.out}; injections.csv updated")
    if not a.no_verify:
        print(f"verification: {len(attacked) - len(failed)}/{len(attacked)} passed")
        for r in failed[:10]:
            print("  FAILED", r["pdf_path"], {k: r[k] for k in ("extracted_ok", "hidden_chars", "expected_chars")})


if __name__ == "__main__":
    main()
