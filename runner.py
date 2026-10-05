"""Run the experiment protocol against one or more models.

Single-doc : each CV x {plain, +p.Pos, +p.Neg} x defenses           (prompt u1 / u1.X)
Multi-doc  : per group, one all-plain baseline session, then for each CV in
             the group inject p.Pos-Neg or p.Pos-Med into that CV only (u2 / u2.X)
Every combination is repeated N trials, each in a fresh conversation.

    python runner.py --models mock --dry-run
    python runner.py --models llama3 --protocol single --trials 3
    python runner.py --models llama3 gpt --defenses none ope spotlight sanitize prefilter

Results are appended to a JSONL file; finished trials are skipped on re-run,
so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dataset import load_manifest
from defenses import DEFENSES, USER_PROMPTS, Document
from llm_clients import load_clients, load_config

SINGLE_ATTACKS = ("plain", "pos", "neg")
MULTI_ATTACKS = ("posneg", "posmed")
PROMPT_IDS = {"single": "u1", "multi": "u2"}


@dataclass
class Trial:
    model: str
    protocol: str
    group_id: str
    group_cvs: list[str]
    target: str | None        # attacked CV (None for the multi-doc baseline)
    attack: str
    defense: str
    trial: int
    docs: list[Document] = field(repr=False, default_factory=list)
    variant: str = ""
    input_format: str = "extracted"

    @property
    def key(self) -> str:
        raw = "|".join(map(str, (self.model, self.protocol, self.group_id, self.target, self.attack,
                                 self.defense, self.trial, self.variant, self.input_format)))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    @property
    def prompt_id(self) -> str:
        base = PROMPT_IDS[self.protocol]
        return base if self.defense == "none" else f"{base}.X" if self.defense == "ope" else f"{base}+{self.defense}"


class DocStore:
    """Loads plain/attacked document text from the inject.py output."""

    def __init__(self, cv_dir, attacked_dir, placement, obfuscation, input_format):
        self.cv_dir, self.attacked = Path(cv_dir), Path(attacked_dir)
        self.variant = f"{placement}_{obfuscation}"
        self.input_format = input_format
        self.manifest = {r["id"]: r for r in load_manifest(cv_dir)}
        inj = self.attacked / "injections.csv"
        if not inj.exists():
            raise SystemExit(f"{inj} not found - run inject.py first")
        self.rows, self._cache = {}, {}
        with open(inj, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["attack"] == "plain" or f"{r['placement']}_{r['obfuscation']}" == self.variant:
                    self.rows[(r["cv_id"], r["attack"])] = r
        missing = [cv for cv in self.manifest for a in ("plain", "pos") if (cv, a) not in self.rows]
        if missing:
            raise SystemExit(f"variant {self.variant} missing for {sorted(set(missing))[:5]}... - "
                             f"run inject.py --placement/--obfuscation accordingly")

    def doc(self, cv_id: str, attack: str) -> Document:
        if (cv_id, attack) in self._cache:
            return self._cache[(cv_id, attack)]
        r = self.rows[(cv_id, attack)]
        path = r["extracted_path"] if self.input_format == "extracted" else r["txt_path"]
        d = Document(cv_id, self.manifest[cv_id]["name"], (self.attacked / path).read_text(encoding="utf-8"),
                     str(self.attacked / r["pdf_path"]), from_pdf=self.input_format == "extracted")
        self._cache[(cv_id, attack)] = d
        return d


def make_groups(manifest: dict, group_by: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for cv_id, r in manifest.items():
        key = {"cell": f"{r['seniority']}-{r['ad_class']}", "seniority": r["seniority"], "all": "all"}[group_by]
        groups[key].append(cv_id)
    return dict(groups)


def plan_trials(store: DocStore, models, protocols, defenses, trials, group_by, cv_filter=None) -> list[Trial]:
    plan: list[Trial] = []
    cvs = [c for c in store.manifest if not cv_filter or c in cv_filter]
    common = dict(variant=store.variant, input_format=store.input_format)
    for model in models:
        if "single" in protocols:
            for cv in cvs:
                for attack in SINGLE_ATTACKS:
                    for d in defenses:
                        for t in range(trials):
                            plan.append(Trial(model, "single", cv, [cv], cv, attack, d, t,
                                              [store.doc(cv, attack)], **common))
        if "multi" in protocols:
            for gid, members in make_groups(store.manifest, group_by).items():
                members = [m for m in members if not cv_filter or m in cv_filter]
                if len(members) < 2:
                    continue
                for d in defenses:
                    for t in range(trials):
                        plan.append(Trial(model, "multi", gid, members, None, "plain", d, t,
                                          [store.doc(m, "plain") for m in members], **common))
                for target in members:
                    for attack in MULTI_ATTACKS:
                        docs = [store.doc(m, attack if m == target else "plain") for m in members]
                        for d in defenses:
                            for t in range(trials):
                                plan.append(Trial(model, "multi", gid, members, target, attack, d, t, docs, **common))
    return plan


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
def estimate(plan: list[Trial], cfg: dict) -> list[dict]:
    """Rough token/cost estimate (~4 characters per token)."""
    agg: dict[tuple, Counter] = defaultdict(Counter)
    for tr in plan:
        n_docs = len(tr.docs)
        doc_tokens = sum(len(d.text) for d in tr.docs) / 4
        if tr.defense == "sanitize":
            tin = doc_tokens + 300 * n_docs + 700 * n_docs
            tout = 700 * n_docs + (400 if n_docs == 1 else 350 * n_docs + 250)
        else:
            overhead = {"none": 40, "ope": 330, "spotlight": 160, "prefilter": 40}.get(tr.defense, 100)
            tin = doc_tokens * (1.25 if tr.defense == "spotlight" else 1) + overhead
            tout = 400 if n_docs == 1 else 350 * n_docs + 250
        c = agg[(tr.model, tr.protocol, tr.defense)]
        c["calls"] += 1 + (n_docs if tr.defense == "sanitize" else 0)
        c["trials"] += 1
        c["in"] += tin
        c["out"] += tout
    rows = []
    for (model, proto, d), c in sorted(agg.items()):
        m = cfg["models"][model]
        cost = c["in"] / 1e6 * (m.get("price_in") or 0) + c["out"] / 1e6 * (m.get("price_out") or 0)
        rows.append({"model": model, "protocol": proto, "defense": d, "trials": c["trials"], "calls": c["calls"],
                     "input_tokens": int(c["in"]), "output_tokens": int(c["out"]), "usd": round(cost, 2)})
    return rows


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_trial(tr: Trial, client, temperature: float, seed: int, store_messages: bool) -> dict:
    defense = DEFENSES[tr.defense]
    resp, meta = defense.run(client, tr.docs, tr.protocol, temperature=temperature, seed=seed + tr.trial)
    calls = meta.pop("calls")
    messages = meta.pop("messages")
    rec = {
        "key": tr.key, "model": tr.model, "model_id": client.model, "provider": client.provider,
        "protocol": tr.protocol, "group_id": tr.group_id, "group_cvs": tr.group_cvs,
        "cv_id": tr.target if tr.target else (tr.group_cvs[0] if tr.protocol == "single" else None),
        "attack": tr.attack, "defense": tr.defense, "trial": tr.trial, "variant": tr.variant,
        "input_format": tr.input_format, "temperature": temperature, "seed": seed + tr.trial,
        "prompt_id": tr.prompt_id, "prompt": USER_PROMPTS[tr.protocol],
        "candidates": {d.cv_id: d.name for d in tr.docs},
        "messages_sha1": hashlib.sha1(json.dumps(messages).encode()).hexdigest(),
        "response": resp.text,
        "latency": round(sum(c["latency"] for c in calls), 3),
        "tokens": {"input": sum(c["input_tokens"] for c in calls), "output": sum(c["output_tokens"] for c in calls),
                   "calls": len(calls)},
        "defense_alert": meta.pop("defense_alert", None),
        "defense_meta": meta,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if store_messages:
        rec["messages"] = messages
    return rec


def execute(plan, clients, out_path: Path, temperature: float, seed: int, workers: int = 1,
            store_messages: bool = False, progress: bool = True) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [t for t in plan if t.key not in done]
    lock = threading.Lock()
    stats = Counter(skipped=len(plan) - len(todo))
    err_path = out_path.with_suffix(".errors.jsonl")
    t0 = time.time()

    def work(tr: Trial):
        try:
            rec = run_trial(tr, clients[tr.model], temperature, seed, store_messages)
        except Exception as e:  # keep going; failed trials are retried on the next run
            with lock, open(err_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": tr.key, "model": tr.model, "error": repr(e)[:1000],
                                    "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
                stats["errors"] += 1
            return
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["done"] += 1
            n = stats["done"] + stats["errors"]
            if progress and (n % 10 == 0 or n == len(todo)):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"  {n}/{len(todo)}  errors={stats['errors']}  "
                      f"eta={(len(todo) - n) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    # models run in parallel; within a model, `workers` threads share its rate limiter
    by_model = defaultdict(list)
    for t in todo:
        by_model[t.model].append(t)
    with ThreadPoolExecutor(max_workers=max(1, workers) * max(1, len(by_model))) as ex:
        futures = [ex.submit(work, t) for ts in by_model.values() for t in ts]
        for _ in as_completed(futures):
            pass
    return dict(stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--protocol", nargs="+", choices=["single", "multi"], default=["single", "multi"])
    ap.add_argument("--defenses", nargs="+", choices=list(DEFENSES))
    ap.add_argument("--trials", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--placement")
    ap.add_argument("--obfuscation")
    ap.add_argument("--input-format", choices=["extracted", "txt"])
    ap.add_argument("--group-by", choices=["cell", "seniority", "all"])
    ap.add_argument("--cvs", nargs="+", help="restrict to these CV ids")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests per model")
    ap.add_argument("--store-messages", action="store_true", help="store full prompts in the JSONL")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and cost estimate, call nothing")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    ex = cfg["experiment"]
    pick = lambda name, key: getattr(a, name) if getattr(a, name) is not None else ex[key]  # noqa: E731
    store = DocStore(ex["cv_dir"], ex["attacked_dir"], pick("placement", "placement"),
                     pick("obfuscation", "obfuscation"), pick("input_format", "input_format"))
    plan = plan_trials(store, a.models, a.protocol, pick("defenses", "defenses"), pick("trials", "trials"),
                       pick("group_by", "group_by"), set(a.cvs) if a.cvs else None)
    out = Path(pick("out", "results"))
    done = load_done(out)
    remaining = [t for t in plan if t.key not in done]

    est = estimate(remaining, cfg)
    print(f"plan: {len(plan)} trials ({len(plan) - len(remaining)} already done, {len(remaining)} to run)")
    print(f"{'model':10} {'protocol':8} {'defense':10} {'trials':>7} {'calls':>7} {'in_tok':>11} {'out_tok':>10} {'usd':>8}")
    for r in est:
        print(f"{r['model']:10} {r['protocol']:8} {r['defense']:10} {r['trials']:7} {r['calls']:7} "
              f"{r['input_tokens']:11,} {r['output_tokens']:10,} {r['usd']:8.2f}")
    print(f"estimated total: ${sum(r['usd'] for r in est):.2f} (approximate; see prices in config.yaml)")
    if a.dry_run:
        return
    if not remaining:
        print("nothing to do")
        return
    clients = load_clients(a.config, a.models)
    stats = execute(remaining, clients, out, pick("temperature", "temperature"), pick("seed", "seed"),
                    a.workers, a.store_messages)
    print(f"finished: {stats}; results in {out}")
    if stats.get("errors"):
        print(f"see {out.with_suffix('.errors.jsonl')}; re-run the same command to retry failed trials")
        sys.exit(2)


if __name__ == "__main__":
    main()
