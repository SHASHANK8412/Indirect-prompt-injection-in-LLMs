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


WORK_DONE = [
    ("Dataset", "27 synthetic CVs matching the paper's Table 1: 3 seniority levels × 3 achievement-density classes, "
                "one role per cell, 450–3,000 words, achievement density checked automatically (dataset.py)."),
    ("Attacks", "The paper's four injections (Definitions 3.1–3.4) verbatim, hidden in the PDF as white or 1pt text "
                "at 4 placements; every attacked PDF is verified to be machine-readable but invisible (inject.py)."),
    ("Experiment runner", "The paper's protocol: single-CV sessions for p.Pos/p.Neg, multi-CV sessions for "
                          "p.Pos-Neg/p.Pos-Med, fresh session per trial, temperature 0, resumable, with cost "
                          "estimates; OpenAI, Gemini, DeepSeek, Grok, Perplexity and local Ollama models (runner.py)."),
    ("Paper's defense", "u1.X / u2.X, the user prompt plus the paper's prompt-extension sentence, verbatim "
                        "(defense ope)."),
    ("OPE architecture", "Section 4.2's three layers as software: isolation, meta-instructions at the outer edges, "
                         "and an audit plus remediation loop (ope/ package; defenses ope_layered and ope_full)."),
    ("Extra defenses", "Beyond the paper: spotlighting/datamarking, a two-pass JSON sanitiser, and a hidden-text "
                       "and instruction pre-filter."),
    ("Metrics", "The paper's ASR, Sentiment Index, ΔSI, DAR, ME and AD; the SI scorer reproduces the paper's "
                "worked example exactly (3.5 → 5.0, ΔSI 1.5). Added: rank shift, false-alarm rate, LLM judge."),
    ("Analysis", "The paper's Tables 4–7 with CV-clustered bootstrap CIs, regression, charts, and an automatic "
                 "side-by-side comparison with the paper's reported values."),
    ("Web application", "Local app with OPE Analyze, Results, Explorer, Dataset and Runs pages."),
    ("Quality", "88 automated tests; README, architecture and paper-alignment documents; GitHub repository."),
]


def _pct(s):
    s = s.dropna()
    return f"{100 * s.mean():.0f}%" if len(s) else "–"


def expected_vs_actual(trials: pd.DataFrame, thr: float) -> pd.DataFrame:
    """The paper's main claims next to what this replication shows so far."""
    und, dfd = trials[trials["defense"] == "none"], trials[trials["defense"] == "ope"]
    per_model = lambda df: ", ".join(f"{m} {_pct(g['success'])}" for m, g in df.groupby("model")) or "–"  # noqa: E731
    dar = ", ".join(f"{m} {_pct(g['alert'].astype(float))}" for m, g in dfd.groupby("model")) or "–"
    sen = dfd.groupby("seniority")["success"].mean() if "seniority" in dfd else pd.Series(dtype=float)
    seniors = [s for s in ("Junior", "Mid", "Senior") if s in sen.index]
    aware = dfd[dfd["alert"].astype(bool)]
    aware_bypassed = f"{100 * (aware['success'] == 1).mean():.0f}% of alerted trials still succeeded" if len(aware) else "–"
    multi = trials[trials["attack"].isin(["posneg", "posmed"])]
    rows = [
        ("All models vulnerable without defense", "ASR 92–98% (Table 4)", per_model(und),
         "Agrees" if len(und) and und["success"].mean() >= 0.5 else "Partly" if len(und) else "Not tested"),
        ("OPE lowers ASR, unevenly across models", "12.5% (ChatGPT) to 45.5% (DeepSeek)", per_model(dfd),
         "Agrees" if dfd.groupby("model")["success"].mean().nunique() > 1 else ("Partly" if len(dfd) else "Not tested")),
        ("OPE raises alerts (DAR)", "50–91% by model", dar, "See values" if len(dfd) else "Not tested"),
        ("Detection ≠ resistance (aware-but-bypassed)", "p.Pos-Med: 85.5% DAR yet 32.2% ASR", aware_bypassed,
         "Agrees" if len(aware) and (aware["success"] == 1).any() else ("No case yet" if len(aware) else "Not tested")),
        ("Longer CVs harder to protect", "Defended ASR Junior 15% → Senior 33% (Table 5)",
         ", ".join(f"{s} {100 * sen[s]:.0f}%" for s in seniors) or "–",
         "Not yet tested" if "Senior" not in seniors else "See Table 5"),
        ("Multi-CV attacks hardest (p.Pos-Med)", "Defended ASR 32.2% (Table 7)",
         per_model(multi[multi["defense"] == "ope"]) if len(multi) else "–", "See Table 7" if len(multi) else "Not yet tested"),
    ]
    return pd.DataFrame(rows, columns=["Claim in the paper", "Expected (paper)", f"Actual (ΔSI ≥ {thr})", "Status"])


def threshold_sensitivity(records_paths, cfg) -> pd.DataFrame:
    rows = []
    for th in (0.5, 1.0):
        c2 = {**cfg, "metrics": {**cfg["metrics"], "asr_threshold": th}}
        _, _, t = load_trials(records_paths, c2)
        for (m, d), g in t.groupby(["model", "defense"]):
            rows.append({"model": m, "defense": d, "threshold": f"ΔSI ≥ {th}", "ASR": 100 * g["success"].mean()})
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["model", "defense"], columns="threshold", values="ASR").reset_index()


def build(paths, out, findings_path=None, figdir=Path("results/report_figures"), status_lines=None):
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
    thr = cfg["metrics"]["asr_threshold"]

    # ---- summary + work done
    S += [P("Summary", h2),
          P("This project replicates an empirical study of indirect prompt injection: instructions hidden in a CV "
            "that change how an LLM evaluates candidates. The full pipeline, the paper's defense and its "
            "three-layer architecture are implemented and tested. Experiments with the paper's protocol are in "
            "progress on free models; the tables below compare the results so far with the paper's."),
          P("1. What has been done", h2),
          df_table(pd.DataFrame(WORK_DONE, columns=["Component", "Work completed"]), widths=[34 * mm, 144 * mm])]

    # ---- setup
    S += [P("2. Experimental setup", h2), bullets([
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
    S += [PageBreak(), P("3. Expected (paper) vs actual (this replication)", h2),
          P("The paper's main claims, what it reports, and what the replication shows so far."),
          df_table(expected_vs_actual(trials, thr), widths=[45 * mm, 45 * mm, 60 * mm, 28 * mm]), Spacer(1, 8),
          KeepTogether([P("<b>How the success threshold changes ASR (%)</b>. The paper does not define the "
                          "threshold. Clean CVs already score SI ≈ 4, so inflating attacks rarely gain a full "
                          "point: ΔSI ≥ 1.0 undercounts them.", body),
                        df_table(threshold_sensitivity(paths, cfg), pct_cols=["ΔSI ≥ 0.5", "ΔSI ≥ 1.0"])])]
    S += [P("4. Detailed results compared with the paper", h2),
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
    S += [PageBreak(), P("5. Charts", h2), Image(str(figdir / "asr.png"), width=165 * mm, height=110 * mm)]
    if (figdir / "me_dar.png").exists():
        S.append(Image(str(figdir / "me_dar.png"), width=150 * mm, height=100 * mm))

    # ---- findings
    if findings_path and Path(findings_path).exists():
        S.append(P("6. Findings", h2))
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
    if status_lines:
        S += [P("7. Status and what remains", h2), bullets(status_lines)]
    S += [P("8. Limitations", h2), bullets([
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
    ap.add_argument("--status", help="text file, one status line per line")
    ap.add_argument("--out", default="Replication_Report.pdf")
    a = ap.parse_args(argv)
    status = [l.strip() for l in Path(a.status).read_text(encoding="utf-8").splitlines() if l.strip()] if a.status else None
    build([Path(p) for p in a.results], Path(a.out), a.findings, status_lines=status)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
