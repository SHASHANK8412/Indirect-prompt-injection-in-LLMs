"""Local web UI for the replication: results dashboard, trial explorer, dataset browser, run control.

    .venv\\Scripts\\python app\\server.py            # http://127.0.0.1:8765

Binds to localhost only. All numbers come from metrics.py / analysis.py, so they
match the CLI exactly.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import analysis as A  # noqa: E402
from dataset import load_manifest  # noqa: E402
from defenses import DEFENSES  # noqa: E402
from llm_clients import load_config  # noqa: E402
from metrics import (alert_spans, evaluative_spans, false_alarm_rate, load_results, score_file,  # noqa: E402
                     split_by_candidate, sentiment_index)
from pdfrender import OBFUSCATIONS, PLACEMENTS  # noqa: E402

RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
STATIC = Path(__file__).parent / "static"
PY = sys.executable

app = FastAPI(title="IPI replication")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cfg() -> dict:
    return load_config(ROOT / "config.yaml")


def clean(v):
    """JSON-safe values (NaN -> None, numpy -> python)."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict(orient="records"))


def results_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.jsonl", name or ""):
        raise HTTPException(400, "invalid results file name")
    p = RESULTS_DIR / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found")
    return p


_score_cache: dict[tuple, tuple] = {}
_lock = threading.Lock()


def scored(name: str, threshold: float, baseline: str):
    p = results_path(name)
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size, threshold, baseline)
    with _lock:
        if key not in _score_cache:
            c = cfg()
            cand, trials = score_file(p, c["experiment"]["cv_dir"], threshold, baseline)
            _score_cache.clear()
            _score_cache[key] = (cand, trials, load_results(p))
        return _score_cache[key]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@app.get("/api/results")
def list_results():
    out = []
    for p in sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.name.endswith(".errors.jsonl") or p.name == "judge_cache.jsonl":
            continue
        with open(p, encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        out.append({"name": p.name, "records": n, "modified": p.stat().st_mtime,
                    "mock": p.name.startswith("mock")})
    return out


@app.get("/api/config")
def get_config():
    c = cfg()
    return {"metrics": c["metrics"], "experiment": c["experiment"], "defenses": list(DEFENSES),
            "placements": PLACEMENTS, "obfuscations": OBFUSCATIONS}


@app.get("/api/summary")
def summary(file: str, threshold: float = 1.0, baseline: str = "matched"):
    cand, trials, recs = scored(file, threshold, baseline)
    if trials.empty:
        return {"empty": True, "records": len(recs)}
    nb = 400
    by_def = A.group_stats(trials, [], nb)
    dims = {}
    for dim in ("model", "seniority", "ad_class", "attack", "protocol"):
        if dim in trials and trials[dim].notna().any():
            g = A.group_stats(trials, [dim], nb)
            g["defense"] = g["defense"].astype(str)
            dims[dim] = records(g)
    heat = (trials.groupby(["defense", "model", "attack"])["success"].mean().reset_index())
    defended = by_def[by_def["defense"] != "none"]
    best = defended.sort_values(["asr", "me"], ascending=[True, False]).head(1)
    by_def["defense"] = by_def["defense"].astype(str)
    fa = false_alarm_rate(cand)
    return clean({
        "empty": False,
        "file": file,
        "records": len(recs),
        "attacked_trials": len(trials),
        "models": sorted(trials["model"].unique()),
        "defenses": [d for d in A.DEFENSE_ORDER if d in set(trials["defense"])],
        "mock": set(trials["model"]) <= {"mock"},
        "incomplete": int(trials["delta"].isna().sum()),
        "by_defense": records(by_def),
        "dims": dims,
        "heat": records(heat),
        "best": records(best.assign(defense=best["defense"].astype(str)))[0] if len(best) else None,
        "false_alarms": records(fa),
    })


TRIAL_COLS = ["key", "model", "protocol", "cv_id", "seniority", "ad_class", "attack", "defense", "trial",
              "si", "si_base", "delta", "success", "me", "alert", "rank_success", "others_delta"]


@app.get("/api/trials")
def trials_list(file: str, threshold: float = 1.0, baseline: str = "matched"):
    cand, trials, recs = scored(file, threshold, baseline)
    att = {r["key"]: r for r in records(trials[[c for c in TRIAL_COLS if c in trials]])} if len(trials) else {}
    manifest = {r["id"]: r for r in load_manifest(cfg()["experiment"]["cv_dir"])}
    si_by_key = cand.groupby("key")["si"].mean().to_dict() if len(cand) else {}
    out = []
    for r in recs:
        row = att.get(r["key"])
        if row is None:  # clean baseline run
            cv = r.get("cv_id")
            m = manifest.get(cv or "", {})
            row = {"key": r["key"], "model": r["model"], "protocol": r["protocol"], "cv_id": cv,
                   "group_id": r["group_id"], "seniority": m.get("seniority"), "ad_class": m.get("ad_class"),
                   "attack": r["attack"], "defense": r["defense"], "trial": r["trial"],
                   "si": clean(si_by_key.get(r["key"])), "alert": bool(r.get("defense_alert"))}
        row["group_id"] = r["group_id"]
        row["latency"] = r.get("latency")
        out.append(row)
    return out


def _injection_span(doc_text: str, injection: str) -> dict | None:
    words = re.findall(r"\S+", injection)
    if not words:
        return None
    m = re.search(r"\s+".join(re.escape(w) for w in words), doc_text)
    return {"start": m.start(), "end": m.end()} if m else None


@lru_cache(maxsize=8)
def _injection_rows(attacked_dir: str) -> dict:
    p = Path(attacked_dir) / "injections.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {(r["cv_id"], r["attack"], f"{r['placement']}_{r['obfuscation']}" if r["placement"] else ""): r
                for r in csv.DictReader(f)}


@app.get("/api/trial")
def trial_detail(file: str, key: str, threshold: float = 1.0, baseline: str = "matched"):
    cand, trials, recs = scored(file, threshold, baseline)
    rec = next((r for r in recs if r["key"] == key), None)
    if rec is None:
        raise HTTPException(404, "trial not found")
    c = cfg()
    att_dir = Path(c["experiment"]["attacked_dir"])
    rows = _injection_rows(str(att_dir.resolve()))
    docs = []
    for cv in rec["group_cvs"]:
        attack = rec["attack"] if (cv == rec["cv_id"] and rec["attack"] != "plain") else "plain"
        r = rows.get((cv, attack, rec["variant"] if attack != "plain" else ""))
        text, span, pdf = "", None, None
        if r:
            path = att_dir / (r["extracted_path"] if rec.get("input_format", "extracted") == "extracted" else r["txt_path"])
            if path.exists():
                text = path.read_text(encoding="utf-8")
            if r.get("injection_text"):
                span = _injection_span(text, r["injection_text"])
            pdf = r["pdf_path"]
        docs.append({"cv_id": cv, "name": rec["candidates"].get(cv, cv), "attack": attack, "text": text,
                     "injection": span, "pdf": pdf,
                     "obfuscation": r.get("obfuscation") if r else None,
                     "placement": r.get("placement") if r else None})
    resp = rec["response"]
    segs = split_by_candidate(resp, rec["candidates"])
    per_cand = []
    crow = cand[cand["key"] == key].set_index("cand_cv") if len(cand) else pd.DataFrame()
    for cv, seg in segs.items():
        si, n = sentiment_index(seg)
        per_cand.append({"cv_id": cv, "name": rec["candidates"][cv], "si": si, "n_terms": n,
                         "rank": clean(crow.loc[cv, "rank"]) if "rank" in crow and cv in crow.index else None})
    metrics_row = None
    if len(trials):
        t = trials[trials["key"] == key]
        if len(t):
            metrics_row = records(t[[c for c in TRIAL_COLS if c in t]])[0]
    return clean({
        "record": {k: v for k, v in rec.items() if k not in ("messages",)},
        "docs": docs,
        "response_spans": evaluative_spans(resp),
        "alert_spans": alert_spans(resp),
        "per_candidate": per_cand,
        "metrics": metrics_row,
    })


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
@app.get("/api/dataset")
def dataset():
    c = cfg()
    cv_dir = Path(c["experiment"]["cv_dir"])
    if not (cv_dir / "manifest.csv").exists():
        return {"cvs": [], "variants": []}
    rows = load_manifest(cv_dir)
    inj = _injection_rows(str(Path(c["experiment"]["attacked_dir"]).resolve()))
    variants = sorted({k[2] for k in inj if k[2]})
    return {"cvs": rows, "variants": variants}


@app.get("/api/cv/{cv_id}")
def cv_detail(cv_id: str, attack: str = "plain", variant: str = ""):
    c = cfg()
    if not re.fullmatch(r"[A-Za-z0-9\-]+", cv_id):
        raise HTTPException(400, "bad id")
    att_dir = Path(c["experiment"]["attacked_dir"])
    inj = _injection_rows(str(att_dir.resolve()))
    r = inj.get((cv_id, attack, variant if attack != "plain" else ""))
    if r is None:
        txt = Path(c["experiment"]["cv_dir"]) / f"{cv_id}.txt"
        if not txt.exists():
            raise HTTPException(404, "CV not found")
        return {"text": txt.read_text(encoding="utf-8"), "extracted": None, "pdf": None, "injection": None}
    extracted = (att_dir / r["extracted_path"]).read_text(encoding="utf-8")
    return {"text": (att_dir / r["txt_path"]).read_text(encoding="utf-8"), "extracted": extracted,
            "pdf": r["pdf_path"], "injection_text": r.get("injection_text") or None,
            "injection": _injection_span(extracted, r["injection_text"]) if r.get("injection_text") else None,
            "verified": r.get("verified"), "max_contrast": r.get("max_contrast"),
            "min_font_size": r.get("min_font_size")}


@app.get("/files/attacked/{path:path}")
def attacked_file(path: str):
    base = Path(cfg()["experiment"]["attacked_dir"]).resolve()
    p = (base / path).resolve()
    if base not in p.parents or p.suffix != ".pdf" or not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="application/pdf")


# ---------------------------------------------------------------------------
# models & jobs
# ---------------------------------------------------------------------------
@app.get("/api/models")
def models():
    c = cfg()
    tags, ollama_up = set(), False
    base = next((m.get("base_url") for m in c["models"].values() if m["provider"] == "ollama" and m.get("base_url")),
                "http://localhost:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=1.5)
        ollama_up = r.ok
        tags = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        pass
    out = []
    for name, m in c["models"].items():
        if m["provider"] == "mock":
            ready, note = True, "simulator - synthetic output"
        elif m["provider"] == "ollama":
            model = m.get("model", "")
            present = model in tags or f"{model}:latest" in tags
            ready = ollama_up and present
            note = ("ready" if ready else "Ollama not running" if not ollama_up else f"run: ollama pull {model}")
        else:
            env = m.get("api_key_env", "")
            ready = bool(os.environ.get(env))
            note = "API key found" if ready else f"set {env}"
        out.append({"key": name, "provider": m["provider"], "model": m.get("model"), "ready": ready, "note": note,
                    "price_in": m.get("price_in"), "price_out": m.get("price_out")})
    return {"ollama_up": ollama_up, "models": out}


class ExperimentParams(BaseModel):
    models: list[str] = Field(min_length=1)
    protocol: list[str] = ["single", "multi"]
    defenses: list[str] = ["none", "ope"]
    trials: int = Field(10, ge=1, le=100)
    temperature: float = Field(0.0, ge=0, le=2)
    cvs: list[str] = []
    placement: str = "bottom"
    obfuscation: str = "white"
    group_by: str = "cell"
    workers: int = Field(1, ge=1, le=16)
    out: str = "raw"


class PrepareParams(BaseModel):
    step: str
    placement: str = "bottom"
    obfuscation: str = "white"
    all_variants: bool = False


def experiment_args(p: ExperimentParams) -> list[str]:
    c = cfg()
    bad = [m for m in p.models if m not in c["models"]]
    if bad:
        raise HTTPException(400, f"unknown models {bad}")
    if not set(p.protocol) <= {"single", "multi"} or not p.protocol:
        raise HTTPException(400, "bad protocol")
    if not set(p.defenses) <= set(DEFENSES) or not p.defenses:
        raise HTTPException(400, "bad defenses")
    if p.placement not in PLACEMENTS or p.obfuscation not in OBFUSCATIONS or p.group_by not in ("cell", "seniority", "all"):
        raise HTTPException(400, "bad variant")
    if any(not re.fullmatch(r"[A-Za-z0-9\-]+", cv) for cv in p.cvs):
        raise HTTPException(400, "bad CV id")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,60}", p.out):
        raise HTTPException(400, "output name: letters, digits, _ and - only")
    args = ["runner.py", "--models", *p.models, "--protocol", *p.protocol, "--defenses", *p.defenses,
            "--trials", str(p.trials), "--temperature", str(p.temperature), "--placement", p.placement,
            "--obfuscation", p.obfuscation, "--group-by", p.group_by, "--workers", str(p.workers),
            "--out", f"results/{p.out}.jsonl"]
    if p.cvs:
        args += ["--cvs", *p.cvs]
    return args


def _parse_plan(stdout: str) -> dict:
    rows = []
    m = re.search(r"plan: (\d+) trials \((\d+) already done, (\d+) to run\)", stdout)
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[3].isdigit():
            rows.append({"model": parts[0], "protocol": parts[1], "defense": parts[2], "trials": int(parts[3]),
                         "calls": int(parts[4]), "input_tokens": int(parts[5].replace(",", "")),
                         "output_tokens": int(parts[6].replace(",", "")), "usd": float(parts[7])})
    tot = re.search(r"estimated total: \$([\d.]+)", stdout)
    return {"total": int(m.group(1)) if m else None, "done": int(m.group(2)) if m else None,
            "to_run": int(m.group(3)) if m else None, "rows": rows, "usd": float(tot.group(1)) if tot else None}


@app.post("/api/dry-run")
def dry_run(p: ExperimentParams):
    args = experiment_args(p) + ["--dry-run"]
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise HTTPException(400, (r.stderr or r.stdout)[-1500:])
    return _parse_plan(r.stdout)


class Job:
    def __init__(self, kind: str, label: str, args: list[str], out: str | None = None):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.label, self.args, self.out = kind, label, args, out
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{self.id}.log"
        self.started = time.time()
        self.ended: float | None = None
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen([PY, *args], cwd=ROOT, stdout=self._log, stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.stopped = False

    def status(self) -> dict:
        code = self.proc.poll()
        if code is not None and self.ended is None:
            self.ended = time.time()
            self._log.close()
        log = self.log_path.read_text(encoding="utf-8", errors="replace") if self.log_path.exists() else ""
        plan = _parse_plan(log)
        prog = re.findall(r"^\s+(\d+)/(\d+)\s+errors=(\d+)\s+eta=([\d.]+) min", log, re.M)
        done, total, errors, eta = (int(prog[-1][0]), int(prog[-1][1]), int(prog[-1][2]), float(prog[-1][3])) \
            if prog else (0, plan.get("to_run") or 0, 0, None)
        state = "running" if code is None else "stopped" if self.stopped else "finished" if code == 0 else "failed"
        return {"id": self.id, "kind": self.kind, "label": self.label, "state": state, "exit_code": code,
                "started": self.started, "ended": self.ended, "done": done, "total": total, "errors": errors,
                "eta_min": eta, "out": self.out, "log_tail": log[-4000:], "command": "python " + " ".join(self.args)}

    def stop(self):
        if self.proc.poll() is None:
            self.stopped = True
            self.proc.terminate()


JOBS: dict[str, Job] = {}


def _busy() -> bool:
    return any(j.proc.poll() is None for j in JOBS.values())


@app.post("/api/runs")
def start_run(p: ExperimentParams):
    if _busy():
        raise HTTPException(409, "a job is already running - stop it or wait for it to finish")
    args = experiment_args(p)
    label = f"{', '.join(p.models)} · {'/'.join(p.protocol)} · {', '.join(p.defenses)} · {p.trials} trial(s)"
    job = Job("experiment", label, args, f"{p.out}.jsonl")
    JOBS[job.id] = job
    return job.status()


@app.post("/api/prepare")
def prepare(p: PrepareParams):
    if _busy():
        raise HTTPException(409, "a job is already running")
    if p.step == "dataset":
        args, label = ["dataset.py", "generate"], "Generate 27 CVs"
    elif p.step == "inject":
        if p.placement not in PLACEMENTS or p.obfuscation not in OBFUSCATIONS:
            raise HTTPException(400, "bad variant")
        args = ["inject.py", "--placement", p.placement, "--obfuscation", p.obfuscation]
        label = f"Build attacked variants ({p.placement}, {p.obfuscation})"
        if p.all_variants:
            args.append("--all-variants")
            label = "Build all attacked variants (4 placements × 3 obfuscations)"
    else:
        raise HTTPException(400, "unknown step")
    _injection_rows.cache_clear()
    job = Job(p.step, label, args)
    JOBS[job.id] = job
    return job.status()


@app.get("/api/runs")
def list_runs():
    if any(j.kind == "inject" and j.proc.poll() is not None for j in JOBS.values()):
        _injection_rows.cache_clear()
    return sorted((j.status() for j in JOBS.values()), key=lambda s: s["started"], reverse=True)


@app.post("/api/runs/{job_id}/stop")
def stop_run(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job.stop()
    return job.status()


# ---------------------------------------------------------------------------
# OPE: the paper's defense architecture as a service
# ---------------------------------------------------------------------------
class OPEDoc(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = ""
    content: str = Field("", max_length=400_000)


class OPEAnalyzeParams(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=8000)
    documents: list[OPEDoc] = Field(min_length=1, max_length=30)
    model: str = "mock"
    mode: str = "ope"                  # ope | baseline | compare
    temperature: float = Field(0.0, ge=0, le=2)
    include_evidence: bool = False     # researcher view: audit signals, flagged statements


class OPELoadParams(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    data_base64: str = Field(min_length=1, max_length=14_000_000)


@app.post("/api/ope/load")
def ope_load(p: OPELoadParams):
    import base64
    from ope.loaders import load_bytes
    try:
        data = base64.b64decode(p.data_base64, validate=True)
        doc = load_bytes(Path(p.filename).stem, p.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # corrupt PDF / DOCX
        raise HTTPException(400, f"could not read {p.filename}: {e.__class__.__name__}")
    return {"name": p.filename, "content": doc.content, "source": doc.source}


@app.post("/api/ope/analyze")
def ope_analyze(p: OPEAnalyzeParams):
    from llm_clients import LLMError, load_clients
    from ope import Document as ODoc, OPEPipeline
    if p.model not in cfg()["models"]:
        raise HTTPException(400, f"unknown model {p.model}")
    if p.mode not in ("ope", "baseline", "compare"):
        raise HTTPException(400, "mode must be ope, baseline or compare")
    ids = [d.id for d in p.documents]
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "document ids must be unique")
    client = load_clients(str(ROOT / "config.yaml"), [p.model])[p.model]
    docs = [ODoc(d.id, d.content, d.name or d.id) for d in p.documents]
    pipe = OPEPipeline(client, log_path=RESULTS_DIR / "ope_security_log.jsonl")
    out = {}
    try:
        for mode in (["baseline", "ope"] if p.mode == "compare" else [p.mode]):
            r = pipe.analyze(p.user_prompt, docs, mode=mode, temperature=p.temperature)
            res = r.public(include_evidence=p.include_evidence)
            res.update(calls=r.calls, latency=round(r.latency, 1), tokens=r.input_tokens + r.output_tokens,
                       sentiment=[{"id": d.id, "name": d.name, "si": round(si, 2), "terms": n}
                                  for d, (si, n) in ((d, sentiment_index(seg)) for d in docs
                                                     for cv, seg in split_by_candidate(
                                                         r.response, {x.id: x.name for x in docs}).items()
                                                     if cv == d.id)])
            out[mode] = res
    except LLMError as e:
        raise HTTPException(502, str(e))
    return clean(out)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8765))
    print(f"IPI replication UI: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
