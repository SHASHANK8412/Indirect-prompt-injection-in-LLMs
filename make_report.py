"""Build the replication report PDF from one or more results files.

    python make_report.py --results results/paper_gemini.jsonl results/paper_local.jsonl \
        --findings results/findings.md --out Replication_Report.pdf

Tables 4-7 are computed (replication vs paper). --findings is an optional markdown-ish text
file (paragraphs; lines starting with "- " become bullets) written after looking at the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

import analysis as A  # noqa: E402
from llm_clients import load_config  # noqa: E402
from metrics import compute_trials, false_alarm_rate, load_directions, score_candidates  # noqa: E402

FONT_DIR = Path("C:/Windows/Fonts")
try:
    for name, file in (("Seg", "segoeui.ttf"), ("Seg-B", "segoeuib.ttf"), ("Seg-I", "segoeuii.ttf"),
                       ("Seg-BI", "segoeuiz.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    from reportlab.lib.fonts import addMapping
    addMapping("Seg", 0, 0, "Seg"); addMapping("Seg", 1, 0, "Seg-B")
    addMapping("Seg", 0, 1, "Seg-I"); addMapping("Seg", 1, 1, "Seg-BI")
    BASE, BOLD = "Seg", "Seg-B"
except Exception:  # non-Windows fallback
    BASE, BOLD = "Helvetica", "Helvetica-Bold"

INK, INK2, ACCENT, LINE, SOFT = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#2a78d6", "#dcdad4", "#eef4fc"))
body = ParagraphStyle("b", fontName=BASE, fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5)
small = ParagraphStyle("s", parent=body, fontSize=8.2, leading=11, textColor=INK2)
h1 = ParagraphStyle("h1", fontName=BOLD, fontSize=19, leading=24, spaceAfter=4)
h2 = ParagraphStyle("h2", fontName=BOLD, fontSize=13, leading=17, textColor=ACCENT, spaceBefore=10, spaceAfter=5)
cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10.5, spaceAfter=0)
hdr = ParagraphStyle("h", parent=cell, fontName=BOLD, textColor=colors.white)


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return ListFlowable([ListItem(P(i), leftIndent=12) for i in items], bulletType="bullet", start="•",
                        leftIndent=12, bulletColor=ACCENT)


def df_table(df: pd.DataFrame, widths=None, pct_cols=()):
    def fmt(c, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        if isinstance(v, float):
            return f"{v:.1f}%" if c in pct_cols or ("(" in c and "dSI" not in c and c != "n") else f"{v:.2f}"
        return str(v)
    data = [[P(str(c), hdr) for c in df.columns]] + [[P(fmt(c, v), cell) for c, v in zip(df.columns, row)]
                                                     for row in df.itertuples(index=False)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.4, LINE)]
    for i, c in enumerate(df.columns):  # shade the paper's columns
        if str(c).endswith("(paper)"):
            st.append(("BACKGROUND", (i, 1), (i, -1), colors.HexColor("#f4f3f0")))
    t.setStyle(TableStyle(st))
    return t


def load_trials(paths: list[Path], cfg: dict):
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            seen = set()
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["key"] not in seen:
                        seen.add(r["key"])
                        records.append(r)
    cand = score_candidates(records)
    manifest = pd.read_csv(Path(cfg["experiment"]["cv_dir"]) / "manifest.csv")
    trials = compute_trials(cand, manifest, cfg["metrics"]["asr_threshold"], cfg["metrics"]["baseline_defense"],
                            load_directions())
    return records, cand, trials


def build(paths, out, findings_path=None, figdir=Path("results/report_figures")):
    cfg = load_config("config.yaml")
    records, cand, trials = load_trials(paths, cfg)
    figdir.mkdir(parents=True, exist_ok=True)
    models = sorted(trials["model"].unique())
    model_ids = {r["model"]: r["model_id"] for r in records}
    n_trials = trials.groupby("model").size().to_dict()
    S = [P("Indirect Prompt Injection in LLM CV Screening", h1),
         P("Replication report · Milani, Franzoni &amp; Florindi, <i>Neural Computing &amp; Applications</i> 38:530 "
           "(2026), doi:10.1007/s00521-026-12266-x", small),
         P("Code: github.com/SHASHANK8412/Indirect-prompt-injection-in-LLMs", small), Spacer(1, 6)]

    # ---- setup
    thr = cfg["metrics"]["asr_threshold"]
    S += [P("1. Setup", h2), bullets([
        "<b>Dataset:</b> 27 synthetic CVs laid out as in the paper's Table 1 (3 seniority levels × 3 achievement-"
        "density classes, one role per cell, 450–3,000 words). Fictional people only.",
        "<b>Attacks:</b> the paper's Definitions 3.1–3.4 verbatim, hidden as white text at the bottom of the PDF; the "
        "model receives the PDF's extracted text.",
        "<b>Prompts:</b> u1 / u2 (no defense) and the paper's u1.X / u2.X (defense <i>ope</i>), verbatim.",
        "<b>Protocol:</b> single-CV sessions for p.Pos and p.Neg; multi-CV sessions with one injected CV for "
        "p.Pos-Neg and p.Pos-Med. Fresh session per trial, temperature 0.",
        f"<b>Models:</b> {', '.join(f'{m} ({model_ids.get(m, m)})' for m in models)}, through official APIs or "
        "locally with Ollama.",
        f"<b>Success rule (ASR):</b> ΔSI ≥ {thr} in the attacker's direction; SI = the paper's keyword-weighted "
        "Sentiment Index.",
        f"<b>Size:</b> {', '.join(f'{m}: {n:,} attacked trials' for m, n in n_trials.items())}. "
        f"{len(records):,} model responses in total, including clean baselines.",
    ])]
    notes = []
    if "llama32" in models:
        notes.append("llama3.2:3b ran the single-CV protocol only: a 27-CV session (~70k tokens) does not fit a "
                     "small local model's context window.")
    if trials.groupby(["model", "defense", "attack"]).size().max() < 27 * 5:
        notes.append("This is a pilot with one trial per condition (the paper used N = 10), so confidence "
                     "intervals are wide.")
    if notes:
        S.append(P("<b>Note:</b> " + " ".join(notes), small))

    # ---- results vs paper
    comp = A.paper_comparison(trials)
    S += [P("2. Results compared with the paper", h2),
          P("Shaded columns are the paper's reported values. Percentages; 'defended' = u1.X / u2.X."),
          KeepTogether([P("<b>Table 4 · by model</b>", body),
                        df_table(comp["table4_model"].drop(columns=["paper model"]))]), Spacer(1, 8),
          KeepTogether([P("<b>Table 5 · by seniority (defended)</b>", body), df_table(comp["table5_seniority"])]),
          Spacer(1, 8),
          KeepTogether([P("<b>Table 6 · by achievement density</b>", body), df_table(comp["table6_ad"])]),
          Spacer(1, 8),
          KeepTogether([P("<b>Table 7 · by injection type (defended)</b>", body), df_table(comp["table7_attack"])])]

    # ---- undefended by attack type, per model
    und = trials[trials["defense"] == "none"]
    if len(und):
        by_att = (und.pivot_table(index="model", columns="attack", values="success", aggfunc="mean") * 100).round(1)
        by_att = by_att[[a for a in ("pos", "neg", "posneg", "posmed") if a in by_att.columns]]
        by_att.columns = [A.ATTACK_LABELS[c] for c in by_att.columns]
        S += [Spacer(1, 8), KeepTogether([P("<b>Undefended ASR by model and injection type (%)</b>", body),
                                          df_table(by_att.reset_index(), pct_cols=list(by_att.columns))])]

    # ---- charts
    A.fig_asr_bars(trials, figdir / "asr.png", 500)
    if (trials["defense"] != "none").any():
        A.fig_me_vs_dar(trials, figdir / "me_dar.png")
    S += [PageBreak(), P("3. Charts", h2), Image(str(figdir / "asr.png"), width=165 * mm, height=110 * mm)]
    if (figdir / "me_dar.png").exists():
        S.append(Image(str(figdir / "me_dar.png"), width=150 * mm, height=100 * mm))

    # ---- findings
    if findings_path and Path(findings_path).exists():
        S.append(P("4. Findings", h2))
        items, para = [], []
        for line in Path(findings_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("- "):
                items.append(line[2:])
            elif line.strip():
                if items:
                    S.append(bullets(items)); items = []
                S.append(P(line))
        if items:
            S.append(bullets(items))

    fa = false_alarm_rate(cand)
    S += [P("5. Limitations", h2), bullets([
        "Models are reached through APIs and local runtimes, not the web interfaces the paper scripted; "
        "Gemini 2.5 Flash is unavailable to new API accounts, so gemini-3.8-flash was used.",
        "The CVs are synthetic, built to Table 1's specification, not the authors' dataset.",
        "The paper does not define its ASR success criterion; this report uses a ΔSI threshold.",
        "SI is a keyword score; responses with few evaluative words make it noisy.",
        "Pilot scale (one trial per condition): treat differences of a few percentage points as noise.",
    ] + ([f"False alarms (alerts on clean CVs under the defense): "
          + ", ".join(f"{r.model} {100 * r.false_alarm_rate:.0f}%" for r in fa.itertuples())] if len(fa) else []))]

    def deco(c, doc):
        c.saveState(); c.setStrokeColor(ACCENT); c.setLineWidth(3)
        c.line(doc.leftMargin, A4[1] - 12 * mm, A4[0] - doc.rightMargin, A4[1] - 12 * mm)
        c.setFont(BASE, 7.5); c.setFillColor(INK2)
        c.drawString(doc.leftMargin, 9 * mm, "Indirect prompt injection in LLM CV screening · replication report")
        c.drawRightString(A4[0] - doc.rightMargin, 9 * mm, f"Page {doc.page}"); c.restoreState()

    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=17 * mm,
                      bottomMargin=16 * mm, title="IPI replication report", author="K V Shashank").build(
        S, onFirstPage=deco, onLaterPages=deco)
    return comp, trials


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--findings")
    ap.add_argument("--out", default="Replication_Report.pdf")
    a = ap.parse_args(argv)
    build([Path(p) for p in a.results], Path(a.out), a.findings)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
