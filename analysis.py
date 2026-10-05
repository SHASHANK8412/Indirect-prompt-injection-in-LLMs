"""Tables, statistics, figures and the defense report.

    python analysis.py --results results/raw.jsonl          # everything -> results/analysis/

Reproduces the paper's tables from your own runs:
  Table 4  per model     : ASR without vs with each defense, ME, DAR
  Table 5  per seniority
  Table 6  per AD class  (with dSI)
  Table 7  per attack type (with aware-but-bypassed rate)
plus 95% bootstrap CIs, a logistic regression (cluster-robust by CV) and a
Bayesian mixed-effects logistic model (random intercept per CV), three charts,
and defense_report.md comparing defenses for Pos-Med and for Senior CVs.
analysis.ipynb calls the same functions.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import false_alarm_rate, score_file

# Reference categorical palette (fixed order) and sequential blue ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEFENSE_ORDER = ["none", "ope", "spotlight", "sanitize", "prefilter"]
ATTACK_LABELS = {"pos": "p.Pos", "neg": "p.Neg", "posneg": "p.Pos-Neg", "posmed": "p.Pos-Med"}
MARKERS = ["o", "s", "^", "D", "v"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, cluster: str = "cv_id", n_boot: int = 2000,
                         seed: int = 0) -> tuple[float, float]:
    """Resample whole CVs (trials of one CV are not independent)."""
    d = df.dropna(subset=[col])
    if d.empty:
        return (np.nan, np.nan)
    g = d.groupby(cluster)[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    est = s / c
    return (float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975)))


def fmt_ci(est, lo, hi, pct=True) -> str:
    if pd.isna(est):
        return "–"
    if pct:
        return f"{est:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{est:.2f} [{lo:.2f}, {hi:.2f}]"


def group_stats(trials: pd.DataFrame, by: list[str], n_boot: int = 2000) -> pd.DataFrame:
    """Long table: one row per group x defense with CIs."""
    rows = []
    for keys, d in trials.groupby(by + ["defense"], dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by + ["defense"], keys))
        defended = row["defense"] != "none"
        asr_lo, asr_hi = cluster_bootstrap_ci(d, "success", n_boot=n_boot)
        row.update(n=len(d), asr=d["success"].mean(), asr_lo=asr_lo, asr_hi=asr_hi,
                   delta_si=d["delta"].mean(), abs_delta_si=d["delta"].abs().mean())
        if defended:
            me_lo, me_hi = cluster_bootstrap_ci(d, "me", n_boot=n_boot)
            dar_lo, dar_hi = cluster_bootstrap_ci(d.assign(alert=d["alert"].astype(float)), "alert", n_boot=n_boot)
            row.update(me=d["me"].mean(), me_lo=me_lo, me_hi=me_hi,
                       dar=d["alert"].mean(), dar_lo=dar_lo, dar_hi=dar_hi,
                       aware_bypassed=(d["alert"].astype(bool) & (d["success"] == 1)).mean(),
                       bypass_given_alert=d.loc[d["alert"].astype(bool), "success"].mean())
        if "rank_success" in d and d["rank_success"].notna().any():
            row["asr_rank"] = d["rank_success"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["defense"] = pd.Categorical(out["defense"], [x for x in DEFENSE_ORDER if x in set(out["defense"])])
    return out.sort_values(by + ["defense"]).reset_index(drop=True)


def paper_table(trials: pd.DataFrame, by: str, n_boot: int = 2000, extra=("delta_si",)) -> pd.DataFrame:
    """Wide, paper-style table: ASR per defense (with CIs), plus ME and DAR per defended condition."""
    long = group_stats(trials, [by], n_boot)
    out = pd.DataFrame(index=pd.Index(sorted(long[by].dropna().unique(), key=_order_key), name=by))
    for d in long["defense"].cat.categories:
        sub = long[long["defense"] == d].set_index(by)
        out[f"ASR {d}"] = [fmt_ci(*sub.loc[i, ["asr", "asr_lo", "asr_hi"]]) if i in sub.index else "–" for i in out.index]
        if d == "none":
            for e in extra:
                if e in sub:
                    out[f"{e} none"] = sub[e].reindex(out.index).round(2)
            continue
        out[f"ME {d}"] = [fmt_ci(*sub.loc[i, ["me", "me_lo", "me_hi"]]) if i in sub.index else "–" for i in out.index]
        out[f"DAR {d}"] = sub["dar"].reindex(out.index).map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
        for e in extra:
            if e in sub:
                out[f"{e} {d}"] = sub[e].reindex(out.index).round(2)
    return out


_ORDER = {k: i for i, k in enumerate(["Junior", "Mid", "Senior", "High", "Medium", "Low",
                                      "pos", "neg", "posneg", "posmed"])}


def _order_key(v):
    return (_ORDER.get(v, 99), str(v))


def table4(trials, n_boot=2000):
    return paper_table(trials, "model", n_boot)


def table5(trials, n_boot=2000):
    return paper_table(trials, "seniority", n_boot)


def table6(trials, n_boot=2000):
    return paper_table(trials, "ad_class", n_boot, extra=("delta_si", "abs_delta_si"))


def table7(trials, n_boot=2000):
    t = paper_table(trials, "attack", n_boot, extra=("delta_si", "aware_bypassed"))
    t.index = t.index.map(lambda a: ATTACK_LABELS.get(a, a))
    return t


def regression(trials: pd.DataFrame) -> dict:
    """Logistic regression of success on design factors.

    1) GLM logit with cluster-robust SEs by CV (trials of one CV are correlated).
    2) Bayesian mixed-effects logit (random intercept per CV), variational fit.
    Factors with a single level in the data are dropped automatically.
    """
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = trials.dropna(subset=["success"]).copy()
    d["success"] = d["success"].astype(int)
    factors = [f for f in ("model", "seniority", "ad_class", "attack", "defense") if f in d and d[f].nunique() > 1]
    refs = {"defense": "none", "seniority": "Junior", "ad_class": "Low", "attack": "pos"}
    terms = [f"C({f}, Treatment('{refs[f]}'))" if f in refs and refs[f] in set(d[f]) else f"C({f})" for f in factors]
    formula = "success ~ " + (" + ".join(terms) if terms else "1")
    out: dict = {"formula": formula, "n": len(d)}
    if d["success"].nunique() < 2:
        out["note"] = "outcome has a single value; regression skipped"
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            glm = smf.logit(formula, d).fit(disp=False, cov_type="cluster", cov_kwds={"groups": d["cv_id"]},
                                            maxiter=200)
            out["logit"] = glm
            out["odds_ratios"] = pd.DataFrame({"OR": np.exp(glm.params), "CI_low": np.exp(glm.conf_int()[0]),
                                               "CI_high": np.exp(glm.conf_int()[1]), "p": glm.pvalues})
        except Exception as e:  # e.g. perfect separation (a defense with 0% ASR)
            out["logit_error"] = repr(e)
            try:
                glm = smf.logit(formula, d).fit_regularized(disp=False, alpha=0.5, maxiter=500)
                out["logit_regularized_params"] = glm.params
            except Exception as e2:
                out["logit_regularized_error"] = repr(e2)
        try:
            mixed = BinomialBayesMixedGLM.from_formula(formula, {"cv": "0 + C(cv_id)"}, d).fit_vb()
            out["mixed"] = mixed
        except Exception as e:
            out["mixed_error"] = repr(e)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_asr_bars(trials: pd.DataFrame, path=None, n_boot=1000):
    import matplotlib.pyplot as plt

    long = group_stats(trials, ["model"], n_boot)
    models = sorted(long["model"].unique())
    defs = list(long["defense"].cat.categories)
    width = 0.8 / len(defs)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models) * len(defs) / 2), 4))
    for i, d in enumerate(defs):
        sub = long[long["defense"] == d].set_index("model").reindex(models)
        x = np.arange(len(models)) + (i - (len(defs) - 1) / 2) * width
        err = np.vstack([sub["asr"] - sub["asr_lo"], sub["asr_hi"] - sub["asr"]]).clip(min=0)
        ax.bar(x, sub["asr"], width * 0.9, color=SERIES[DEFENSE_ORDER.index(d)], label=d,
               edgecolor="white", linewidth=1)
        ax.errorbar(x, sub["asr"], yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.set_xticks(np.arange(len(models)), models)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax, "Attack success rate by model and defense", "ASR")
    ax.set_xlabel("Error bars: 95% CI, bootstrap resampling whole CVs", color=INK_2, fontsize=8)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, title="defense",
              title_fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def fig_heatmap(trials: pd.DataFrame, defense: str = "none", path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    d = trials[trials["defense"] == defense]
    pv = d.pivot_table(index="model", columns="attack", values="success", aggfunc="mean")
    pv = pv[[a for a in ATTACK_LABELS if a in pv.columns]]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(1.6 * pv.shape[1] + 2, 0.6 * pv.shape[0] + 1.6))
    im = ax.imshow(pv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [ATTACK_LABELS[a] for a in pv.columns])
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for (i, j), v in np.ndenumerate(pv.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.55 else INK)
    ax.set_title(f"ASR by model × attack type (defense: {defense})", loc="left", fontsize=11, color=INK, pad=10)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cb.outline.set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


def fig_me_vs_dar(trials: pd.DataFrame, path=None):
    """One point per model x defense x attack. Points high on DAR but low on ME are
    'aware-but-bypassed': the model flagged the attack and was moved anyway."""
    import matplotlib.pyplot as plt

    d = trials[trials["defense"] != "none"]
    agg = d.groupby(["model", "defense", "attack"], observed=True).agg(
        dar=("alert", "mean"), me=("me", "mean"),
        aware_bypassed=("success", lambda s: (s.eq(1) & d.loc[s.index, "alert"].astype(bool)).mean()),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    defs = [x for x in DEFENSE_ORDER if x in set(agg["defense"])]
    for d_name in defs:
        sub = agg[agg["defense"] == d_name]
        k = DEFENSE_ORDER.index(d_name)
        ax.scatter(sub["dar"], sub["me"], s=70 + 600 * sub["aware_bypassed"], color=SERIES[k],
                   marker=MARKERS[k], alpha=0.85, edgecolor="white", linewidth=1.5, label=d_name)
    worst = agg.sort_values("aware_bypassed", ascending=False).head(2)
    for i, r in enumerate(worst.itertuples()):
        if r.aware_bypassed > 0:
            ax.annotate(f"{r.model} · {r.defense} · {ATTACK_LABELS.get(r.attack, r.attack)}: "
                        f"{r.aware_bypassed:.0%} alerted yet succeeded",
                        (r.dar, r.me), xytext=(-14, -30 - 18 * i), textcoords="offset points", ha="right",
                        fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6))
    ax.axvspan(0.5, 1.02, ymin=0, ymax=0.5, color=GRID, alpha=0.35, zorder=0)
    ax.text(0.99, 0.02, "aware-but-bypassed zone", ha="right", va="bottom", fontsize=8, color=INK_2,
            transform=ax.transAxes)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Detection alert rate (DAR)", color=INK_2)
    _style(ax, "Mitigation efficiency vs detection rate", "ME")
    ax.legend(frameon=False, fontsize=9, title="defense", title_fontsize=9, loc="upper left", markerscale=0.7)
    fig.text(0.01, 0.01, "One point per model × defense × attack. Marker size = share of trials that raised an "
             "alert AND still succeeded.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if path:
        fig.savefig(path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Defense comparison report
# ---------------------------------------------------------------------------
def _rank_defenses(sub: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    s = group_stats(sub, [], n_boot) if len(sub) else pd.DataFrame()
    if s.empty:
        return s
    s = s[s["defense"] != "none"].copy()
    s["defense"] = s["defense"].astype(str)
    return s.sort_values(["asr", "me"], ascending=[True, False])


def defense_report(trials: pd.DataFrame, cand: pd.DataFrame | None = None, n_boot: int = 2000,
                   source: str = "") -> str:
    fa = false_alarm_rate(cand).groupby("defense")["false_alarm_rate"].mean() if cand is not None else pd.Series()
    none_asr = lambda sub: sub.loc[sub["defense"] == "none", "success"].mean()  # noqa: E731
    lines = ["# Defense comparison", ""]
    if source:
        lines += [f"Source: `{source}` · models: {', '.join(sorted(trials['model'].unique()))} · "
                  f"{len(trials):,} attacked trials", ""]
    if set(trials["model"]) <= {"mock"}:
        lines += ["> **Warning:** these numbers come from the mock simulator and are not experimental results.", ""]
    lines += ["Ranking: lowest ASR first, ties broken by higher mitigation efficiency (ME). "
              "CIs are 95% bootstrap intervals that resample whole CVs.", ""]
    sections = [("p.Pos-Med attack (multi-document)", trials[trials["attack"] == "posmed"]),
                ("Senior CVs (~2,500 words)", trials[trials.get("seniority") == "Senior"]),
                ("All attacked trials", trials)]
    best = {}
    for title, sub in sections:
        lines += [f"## {title}", ""]
        r = _rank_defenses(sub, n_boot)
        if r.empty:
            lines += ["_No trials for this subset._", ""]
            continue
        lines += [f"Undefended ASR: **{none_asr(sub):.1%}**" if pd.notna(none_asr(sub)) else "Undefended ASR: –", "",
                  "| defense | n | ASR [95% CI] | ME [95% CI] | DAR | alerted but still succeeded | false alarms on clean input |",
                  "|---|---:|---|---|---:|---:|---:|"]
        for row in r.itertuples():
            lines.append(f"| {row.defense} | {row.n} | {fmt_ci(row.asr, row.asr_lo, row.asr_hi)} | "
                         f"{fmt_ci(row.me, row.me_lo, row.me_hi)} | {row.dar:.1%} | {row.aware_bypassed:.1%} | "
                         f"{fa.get(row.defense, np.nan):.1%} |")
        top = r.iloc[0]
        best[title] = top["defense"]
        overlap = len(r) > 1 and r.iloc[1]["asr_lo"] <= top["asr_hi"]
        lines += ["", f"**Best: `{top['defense']}`** (ASR {top['asr']:.1%}, ME {top['me']:.1%})."
                  + (f" Its CI overlaps with `{r.iloc[1]['defense']}`, so the difference is not conclusive."
                     if overlap else ""), ""]
    lines += ["## Caveats", "",
              "- ASR uses a dSI threshold (see config.yaml); the paper does not specify its success criterion.",
              "- SI is a lexicon score; check its agreement with the LLM-as-judge SI (`metrics.py --judge-model`).",
              "- `sanitize` costs one extra call per CV; `prefilter` relies on PDF styling and a heuristic classifier "
              "that a paraphrased or non-imperative injection may evade.", ""]
    return "\n".join(lines)


def run_all(results: Path, cv_dir="data/cvs", out_dir: Path | None = None, threshold=1.0,
            baseline_defense="matched", n_boot=2000) -> dict:
    import matplotlib
    matplotlib.use("Agg")

    out_dir = out_dir or results.parent / "analysis"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    cand, trials = score_file(results, cv_dir, threshold, baseline_defense)
    tables = {"table4_model": table4(trials, n_boot), "table5_seniority": table5(trials, n_boot),
              "table6_ad": table6(trials, n_boot), "table7_attack": table7(trials, n_boot)}
    md = []
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")
        md += [f"## {name}", "", t.to_markdown(), ""]
    (out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")

    reg = regression(trials)
    with open(out_dir / "regression.txt", "w", encoding="utf-8") as f:
        f.write(f"formula: {reg['formula']}\nn = {reg['n']}\n\n")
        for k in ("note", "logit_error", "mixed_error"):
            if k in reg:
                f.write(f"{k}: {reg[k]}\n")
        if "odds_ratios" in reg:
            for dname, v in trials.groupby("defense")["success"].mean().items():
                if v in (0, 1):
                    f.write(f"NOTE: defense '{dname}' has ASR {v:.0%} (perfect separation); its odds ratio "
                            "and CI are not meaningful - report the raw rate instead.\n")
            f.write("Logit, cluster-robust SE by CV - odds ratios\n" + reg["odds_ratios"].round(4).to_string() + "\n\n")
        if "mixed" in reg:
            f.write("Bayesian mixed-effects logit (random intercept per CV)\n" + str(reg["mixed"].summary()) + "\n")

    fig_asr_bars(trials, out_dir / "figures" / "asr_by_model.png", min(n_boot, 1000))
    for d in trials["defense"].unique():
        fig_heatmap(trials, d, out_dir / "figures" / f"heatmap_{d}.png")
    if (trials["defense"] != "none").any():
        fig_me_vs_dar(trials, out_dir / "figures" / "me_vs_dar.png")
    report = defense_report(trials, cand, n_boot, source=str(results))
    (out_dir / "defense_report.md").write_text(report, encoding="utf-8")
    return {"cand": cand, "trials": trials, "tables": tables, "regression": reg, "report": report, "out": out_dir}


def main(argv=None):
    from llm_clients import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    results = Path(a.results or cfg["experiment"]["results"])
    res = run_all(results, cfg["experiment"]["cv_dir"], Path(a.out) if a.out else None,
                  cfg["metrics"]["asr_threshold"], cfg["metrics"]["baseline_defense"], a.n_boot)
    pd.set_option("display.width", 200)
    print(res["tables"]["table4_model"].to_string())
    print(f"\nwrote tables, regression, figures and defense_report.md to {res['out']}")


if __name__ == "__main__":
    main()
