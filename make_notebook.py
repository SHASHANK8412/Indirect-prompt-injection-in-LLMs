"""Regenerate analysis.ipynb (keeps the notebook diff-friendly and in sync with analysis.py)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Indirect prompt injection in LLM CV screening: analysis\n"
       "Replication of Milani, Franzoni & Florindi (2026), *Neural Computing & Applications* 38:530, "
       "plus three extra defenses.\n\n"
       "Set `RESULTS` to the JSONL written by `runner.py`. All logic lives in `analysis.py` / `metrics.py`, "
       "so the notebook and the CLI (`python analysis.py`) give identical numbers."),
    code("from pathlib import Path\n"
         "import pandas as pd\n"
         "import analysis as A\n"
         "from metrics import score_file, false_alarm_rate, validity\n"
         "from llm_clients import load_config\n\n"
         "cfg = load_config('config.yaml')\n"
         "RESULTS = Path(cfg['experiment']['results'])   # e.g. Path('results/raw.jsonl')\n"
         "N_BOOT = 2000\n"
         "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)"),
    code("cand, trials = score_file(RESULTS, cfg['experiment']['cv_dir'], cfg['metrics']['asr_threshold'],\n"
         "                          cfg['metrics']['baseline_defense'])\n"
         "if set(trials['model']) <= {'mock'}:\n"
         "    print('WARNING: mock-simulator data - not experimental results')\n"
         "print(f\"{len(trials):,} attacked trials · models: {sorted(trials['model'].unique())} · \"\n"
         "      f\"defenses: {sorted(trials['defense'].unique())}\")\n"
         "trials.head()"),
    md("## Table 4: per model\nASR without vs with each defense (95% CI, bootstrap resampling whole CVs), ME, DAR."),
    code("A.table4(trials, N_BOOT)"),
    md("## Table 5: by seniority (document length)\nPaper: defended ASR 15% (Junior) vs 33% (Senior), which the authors attribute to the 'lost in the middle' effect."),
    code("A.table5(trials, N_BOOT)"),
    md("## Table 6: by achievement density\nPaper: high-AD CVs show less bias (ΔSI 1.2 vs 2.6) but are harder to fully mitigate (ME 71% vs 86%)."),
    code("A.table6(trials, N_BOOT)"),
    md("## Table 7: by injection type\n`aware_bypassed` is the share of trials where the model raised an alert *and* the attack still succeeded (paper: p.Pos-Med detected 85.5% of the time, yet 32.2% ASR)."),
    code("A.table7(trials, N_BOOT)"),
    md("## False alarms on clean CVs\nHow often each defense raises an alert when there is no injection."),
    code("false_alarm_rate(cand)"),
    md("## Regression\nLogistic regression of success on model, seniority, AD class, attack type and defense, with "
       "cluster-robust SEs by CV, plus a Bayesian mixed-effects logit with a random intercept per CV. "
       "A defense with 0% ASR causes perfect separation; read its raw rate, not its odds ratio."),
    code("reg = A.regression(trials)\n"
         "print(reg['formula'])\n"
         "reg.get('odds_ratios', reg.get('note') or reg.get('logit_error'))"),
    code("reg['mixed'].summary() if 'mixed' in reg else reg.get('mixed_error')"),
    md("## Charts"),
    code("A.fig_asr_bars(trials, n_boot=1000);"),
    code("for d in sorted(trials['defense'].unique()):\n    A.fig_heatmap(trials, d)"),
    code("A.fig_me_vs_dar(trials) if (trials['defense'] != 'none').any() else None;"),
    md("## Lexicon SI vs LLM-as-judge (validity check)\nRun `python metrics.py --judge-model <key>` first; "
       "this cell only reads the cached judge scores if they exist."),
    code("from metrics import judge_scores, load_results\n"
         "from llm_clients import load_clients\n"
         "JUDGE = None   # e.g. 'llama3'; scores are cached in results/judge_cache.jsonl\n"
         "if JUDGE:\n"
         "    c = judge_scores(cand, load_results(RESULTS), load_clients('config.yaml', [JUDGE])[JUDGE],\n"
         "                     RESULTS.parent / 'judge_cache.jsonl')\n"
         "    print(validity(c))"),
    md("## Extension: which defense works best?\nCompares OPE with spotlighting/datamarking, the two-pass JSON "
       "sanitizer and the hidden-text and imperative-sentence pre-filter, for p.Pos-Med and for Senior CVs."),
    code("from IPython.display import Markdown\n"
         "report = A.defense_report(trials, cand, N_BOOT, source=str(RESULTS))\n"
         "Path('results/defense_report.md').write_text(report, encoding='utf-8')\n"
         "Markdown(report)"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "ipi-replication", "display_name": "Python (ipi-replication)",
                                                               "language": "python"}})
nbf.write(nb, "analysis.ipynb")
print("wrote analysis.ipynb")
