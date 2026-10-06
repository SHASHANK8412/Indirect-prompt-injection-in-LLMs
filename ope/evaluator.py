"""Baseline vs defended evaluation of the OPE over the paper's attack fixtures.

    python -m ope.evaluator --model llama3 --cvs J-H-1 M-M-2 --trials 1
    python -m ope.evaluator --model mock --protocol single multi --out results/ope_eval.jsonl

Conditions compared:
  none      User prompt + document -> LLM                                  (baseline)
  ope       Isolation + Meta-instruction -> LLM (model audits itself)       (paper's u1.X / u2.X)
  ope_full  Isolation + Meta-instruction -> LLM -> Audit -> Remediation     (full architecture)

Reports ASR, DAR, ΔSI, ME per condition, plus how often the OPE pipeline ended
safe / alerted / remediated / remediation_failed (aware-but-bypassed).
Trials are run by runner.py, so they are cached and resumable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_clients import load_clients, load_config
from metrics import score_file, summarize
from runner import DocStore, execute, plan_trials

CONDITIONS = ["none", "ope", "ope_full"]


def status_breakdown(results_path: Path) -> pd.DataFrame:
    rows = []
    for line in open(results_path, encoding="utf-8"):
        r = json.loads(line)
        if r["defense"] == "ope_full":
            st = (r.get("defense_meta") or {}).get("ope", {})
            rows.append({"model": r["model"], "attack": r["attack"], "final_status": st.get("final_status"),
                         "llm_calls": st.get("llm_calls")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    tab = pd.crosstab([df["model"], df["attack"]], df["final_status"], normalize="index").round(2)
    tab["mean_llm_calls"] = df.groupby(["model", "attack"])["llm_calls"].mean().round(2)
    return tab


def evaluate(results_path: Path, cv_dir="data/cvs", threshold=1.0) -> dict:
    cand, trials = score_file(results_path, cv_dir, threshold)
    table = summarize(trials, ["model"])
    return {"summary": table, "by_attack": summarize(trials, ["model", "attack"]),
            "status": status_breakdown(results_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--protocol", nargs="+", choices=["single", "multi"], default=["single"])
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--cvs", nargs="+")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out", default="results/ope_eval.jsonl")
    ap.add_argument("--score-only", action="store_true", help="skip running, only score the file")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ex = cfg["experiment"]
    out = Path(a.out)
    if not a.score_only:
        store = DocStore(ex["cv_dir"], ex["attacked_dir"], ex["placement"], ex["obfuscation"], ex["input_format"])
        plan = plan_trials(store, a.model, a.protocol, a.conditions, a.trials, ex["group_by"],
                           set(a.cvs) if a.cvs else None)
        print(f"{len(plan)} trials planned")
        stats = execute(plan, load_clients(a.config, a.model), out, a.temperature, ex["seed"])
        print(f"run: {stats}")
    res = evaluate(out, ex["cv_dir"], a.threshold if a.threshold is not None else cfg["metrics"]["asr_threshold"])
    pd.set_option("display.width", 180)
    print("\n== Baseline vs defended ==")
    print(res["summary"].round(3).to_string(index=False))
    print("\n== By attack ==")
    print(res["by_attack"].round(3).to_string(index=False))
    if len(res["status"]):
        print("\n== OPE pipeline outcomes (share of ope_full trials) ==")
        print(res["status"].to_string())


if __name__ == "__main__":
    main()
