"use strict";

/* ====================================================================
   State & helpers
   ==================================================================== */
const S = {
  view: "results",
  files: [], file: null, threshold: 1, baseline: "matched",
  summary: null, trials: null, trialsFor: null,
  dim: "model", metric: "asr", heatDefense: null,
  filters: { model: "", defense: "", attack: "", protocol: "", outcome: "", q: "" },
  listLimit: 200, selKey: null, detail: null, docTab: 0,
  dataset: null, selCv: null, cvVariant: "plain", cvInfo: null,
  models: null, config: null, runs: [], plan: null,
};
const DEF_ORDER = ["none", "ope", "spotlight", "sanitize", "prefilter"];
const DEF_COLOR = { none: "var(--c1)", ope: "var(--c2)", spotlight: "var(--c3)", sanitize: "var(--c4)", prefilter: "var(--c5)" };
const DEF_NAME = { none: "No defense", ope: "OPE (paper)", spotlight: "Spotlighting", sanitize: "Two-pass sanitizer", prefilter: "Pre-filter" };
const ATTACK = { plain: "Clean", pos: "p.Pos", neg: "p.Neg", posneg: "p.Pos-Neg", posmed: "p.Pos-Med" };
const ATTACK_DESC = { pos: "inflate one CV", neg: "deflate one CV", posneg: "inflate attacker, run down others", posmed: "inflate attacker, make others look ordinary" };
const DIM_LABEL = { model: "Model", seniority: "Seniority", ad_class: "Achievement density", attack: "Attack type", protocol: "Protocol" };
const ORDER = { Junior: 0, Mid: 1, Senior: 2, High: 0, Medium: 1, Low: 2, pos: 0, neg: 1, posneg: 2, posmed: 3, single: 0, multi: 1 };
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = (v) => typeof v === "number" && !Number.isNaN(v);
const pct = (v, d = 0) => (isNum(v) ? `${(v * 100).toFixed(d)}%` : "–");
const num = (v, d = 2) => (isNum(v) ? v.toFixed(d) : "–");
const signed = (v, d = 2) => (isNum(v) ? (v > 0 ? "+" : "") + v.toFixed(d) : "–");
const byOrder = (a, b) => (ORDER[a] ?? 99) - (ORDER[b] ?? 99) || String(a).localeCompare(String(b));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return r.json();
}
function qs(extra = {}) {
  return new URLSearchParams({ file: S.file, threshold: S.threshold, baseline: S.baseline, ...extra }).toString();
}
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (err ? " err" : ""); t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), err ? 6000 : 3000);
}
function loading(el, msg = "Loading") { el.innerHTML = `<div class="loading"><span class="spin"></span> ${esc(msg)}…</div>`; }
function heatBg(v, max = 1) {
  if (!isNum(v)) return "background:var(--surface-2);color:var(--text-3)";
  const p = Math.max(0, Math.min(1, v / max));
  const mix = Math.round(8 + p * 82);
  return `background:color-mix(in srgb, var(--accent) ${mix}%, var(--surface));color:${p > 0.5 ? "#fff" : "var(--text)"}`;
}
function defChip(d) { return `<span class="chip"><span class="dot" style="background:${DEF_COLOR[d] || "var(--text-3)"}"></span>${esc(DEF_NAME[d] || d)}</span>`; }
function emptyState(title, body, action = "") {
  return `<div class="card empty"><h3>${esc(title)}</h3><p>${body}</p>${action}</div>`;
}

/* tooltip for any [data-tip] */
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest("[data-tip]"); const tip = $("#tooltip");
  if (!el) { tip.hidden = true; return; }
  tip.innerHTML = el.dataset.tip; tip.hidden = false;
});
document.addEventListener("mousemove", (e) => {
  const tip = $("#tooltip"); if (tip.hidden) return;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = e.clientY + 16 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16;
  tip.style.left = x + "px"; tip.style.top = y + "px";
});

/* theme */
(function initTheme() {
  let t = null; try { t = localStorage.getItem("theme"); } catch {}
  if (t) document.documentElement.dataset.theme = t;
  $("#themeToggle").onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {}
  };
})();

/* ====================================================================
   Routing & top bar
   ==================================================================== */
const TITLES = { analyze: "OPE Analyze", results: "Results", explorer: "Trial explorer", dataset: "Dataset", runs: "Runs" };
function route() {
  const v = (location.hash || "#results").slice(1).split("/")[0];
  S.view = TITLES[v] ? v : "results";
  $$("nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === S.view));
  $$(".view").forEach((s) => (s.hidden = s.id !== `view-${S.view}`));
  $("#viewTitle").textContent = TITLES[S.view];
  $("#resultControls").hidden = !["results", "explorer"].includes(S.view);
  render();
}
window.addEventListener("hashchange", route);

function render() {
  ({ analyze: renderAnalyze, results: renderResults, explorer: renderExplorer, dataset: renderDataset, runs: renderRuns })[S.view]();
}

async function loadFiles(keep = true) {
  S.files = await api("/api/results");
  const sel = $("#fileSelect");
  if (!S.files.length) { sel.innerHTML = `<option value="">no results yet</option>`; S.file = null; return; }
  if (!keep || !S.files.some((f) => f.name === S.file)) {
    let saved = null; try { saved = localStorage.getItem("file"); } catch {}
    S.file = S.files.some((f) => f.name === saved) ? saved : (S.files.find((f) => !f.mock) || S.files[0]).name;
  }
  sel.innerHTML = S.files.map((f) =>
    `<option value="${esc(f.name)}" ${f.name === S.file ? "selected" : ""}>${esc(f.name)} · ${f.records} runs${f.mock ? " · mock" : ""}</option>`).join("");
}
function invalidate() { S.summary = null; S.trials = null; S.detail = null; }
$("#fileSelect").onchange = (e) => { S.file = e.target.value; try { localStorage.setItem("file", S.file); } catch {} S.selKey = null; invalidate(); render(); };
$("#thresholdSelect").onchange = (e) => { S.threshold = +e.target.value; invalidate(); render(); };
$("#baselineSelect").onchange = (e) => { S.baseline = e.target.value; invalidate(); render(); };

/* ====================================================================
   Results view
   ==================================================================== */
async function renderResults() {
  const el = $("#view-results");
  if (!S.file) {
    el.innerHTML = emptyState("No results yet", "Run an experiment first - or try the mock model to see how everything works.",
      `<a class="btn primary" href="#runs">Go to Runs</a>`);
    return;
  }
  if (!S.summary || S.summary.file !== S.file) {
    loading(el, "Scoring results");
    try { S.summary = await api(`/api/summary?${qs()}`); S.summary.file = S.file; }
    catch (e) { el.innerHTML = emptyState("Could not score this file", esc(e.message)); return; }
  }
  const s = S.summary;
  if (s.empty) {
    el.innerHTML = emptyState("No attacked trials yet", `This file has ${s.records} run(s) but no attacked trial with a matching clean baseline. Let the run continue, then refresh.`);
    return;
  }
  const none = s.by_defense.find((r) => r.defense === "none");
  const ope = s.by_defense.find((r) => r.defense === "ope");
  const banners = [];
  if (s.mock) banners.push(`<div class="banner warn"><b>Mock data.</b>&nbsp;These numbers come from the simulator and only show that the pipeline works. Never report them.</div>`);
  if (s.incomplete) banners.push(`<div class="banner info">${s.incomplete} attacked trial(s) have no clean baseline yet, so they are left out of ASR.</div>`);
  const small = s.attacked_trials < 40;
  if (small && !s.mock) banners.push(`<div class="banner info">Only ${s.attacked_trials} attacked trials: treat these numbers as a pilot, and read the confidence intervals.</div>`);

  el.innerHTML = `
    ${banners.join("")}
    <div class="kpis">
      ${kpi("Attacked trials", s.attacked_trials.toLocaleString(), `${s.models.join(", ")} · ${s.records.toLocaleString()} runs incl. baselines`)}
      ${kpi("ASR, no defense", none ? pct(none.asr) : "–", none ? `95% CI ${pct(none.asr_lo)}–${pct(none.asr_hi)} · paper: 92–98%` : "no undefended trials",
            none ? `Share of attacked trials where SI moved ≥ ${S.threshold} toward the attacker's goal` : "")}
      ${kpi("Best defense", s.best ? esc(DEF_NAME[s.best.defense] || s.best.defense) : "–",
            s.best ? `ASR ${pct(s.best.asr)} · ME ${pct(s.best.me)} · DAR ${pct(s.best.dar)}` : "run with a defense to compare")}
      ${kpi("OPE: alerted yet bypassed", ope ? pct(ope.aware_bypassed) : "–",
            ope ? `of OPE trials raised an alert and still succeeded · DAR ${pct(ope.dar)}` : "no OPE trials",
            "The paper's 'aware-but-bypassed' effect: detection without resistance")}
    </div>
    <div class="grid cols-2">
      <div class="card card-pad">
        <h2>Attack success by defense</h2>
        <p class="sub">Share of attacked trials that succeeded. Whiskers show 95% CIs from resampling whole CVs.</p>
        ${barChart(s.by_defense)}
      </div>
      <div class="card card-pad">
        <h2>Defense scorecard</h2>
        <p class="sub">ME = mitigation efficiency (100% = output back at baseline). DAR = detection alert rate. False alarms = alerts on clean CVs.</p>
        ${scorecard(s)}
      </div>
    </div>
    <div class="card card-pad" style="margin-top:16px">
      <div class="row" style="margin-bottom:12px">
        <div><h2>Breakdown</h2><p class="sub" style="margin:0">Reproduces the paper's Tables 4–7. Hover a cell for n and its CI.</p></div>
        <div class="spacer"></div>
        <div class="seg" id="dimSeg">${Object.keys(s.dims).map((d) => `<button data-v="${d}" class="${d === S.dim ? "on" : ""}">${DIM_LABEL[d]}</button>`).join("")}</div>
        <div class="seg" id="metricSeg">${[["asr", "ASR"], ["me", "ME"], ["dar", "DAR"], ["delta_si", "ΔSI"], ["aware_bypassed", "Alerted & bypassed"]]
          .map(([k, l]) => `<button data-v="${k}" class="${k === S.metric ? "on" : ""}">${l}</button>`).join("")}</div>
      </div>
      <div class="table-wrap">${breakdown(s)}</div>
    </div>
    <div class="card card-pad" style="margin-top:16px">
      <div class="row" style="margin-bottom:12px">
        <div><h2>Model × attack type</h2><p class="sub" style="margin:0">ASR for each model and injection template, under one defense.</p></div>
        <div class="spacer"></div>
        <div class="seg" id="heatSeg">${s.defenses.map((d) => `<button data-v="${d}" class="${d === (S.heatDefense || s.defenses[0]) ? "on" : ""}">${esc(DEF_NAME[d] || d)}</button>`).join("")}</div>
      </div>
      <div class="table-wrap">${modelAttackHeat(s)}</div>
    </div>`;
  $$("#dimSeg button").forEach((b) => (b.onclick = () => { S.dim = b.dataset.v; renderResults(); }));
  $$("#metricSeg button").forEach((b) => (b.onclick = () => { S.metric = b.dataset.v; renderResults(); }));
  $$("#heatSeg button").forEach((b) => (b.onclick = () => { S.heatDefense = b.dataset.v; renderResults(); }));
}
function kpi(label, value, hint, tip = "") {
  return `<div class="card kpi" ${tip ? `data-tip="${esc(tip)}"` : ""}><div class="label">${label}</div><div class="value">${value}</div><div class="hint">${hint}</div></div>`;
}
function barChart(rows) {
  const sorted = [...rows].sort((a, b) => DEF_ORDER.indexOf(a.defense) - DEF_ORDER.indexOf(b.defense));
  return `<div class="bars">${sorted.map((r) => {
    const w = (r.asr || 0) * 100, lo = (r.asr_lo ?? r.asr) * 100, hi = (r.asr_hi ?? r.asr) * 100;
    return `<div class="bar-row" data-tip="<b>${esc(DEF_NAME[r.defense] || r.defense)}</b><br>ASR ${pct(r.asr, 1)} (95% CI ${pct(r.asr_lo, 1)}–${pct(r.asr_hi, 1)})<br>n = ${r.n} trials">
      <div class="bar-label"><span class="dot" style="background:${DEF_COLOR[r.defense]}"></span>${esc(r.defense)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${DEF_COLOR[r.defense]}"></div>
        ${hi > lo ? `<div class="bar-ci" style="left:${lo}%;width:${hi - lo}%"></div>` : ""}</div>
      <div class="bar-val">${pct(r.asr)} <small>n=${r.n}</small></div></div>`;
  }).join("")}
  <div class="axis"><div></div><div><span>0%</span><span>50%</span><span>100%</span></div><div></div></div></div>`;
}
function scorecard(s) {
  const fa = Object.fromEntries((s.false_alarms || []).map((r) => [r.defense, r.false_alarm_rate]));
  const rows = s.by_defense.filter((r) => r.defense !== "none").sort((a, b) => (a.asr - b.asr) || (b.me - a.me));
  if (!rows.length) return `<p class="muted">Only undefended trials in this file. Add defenses in a run to compare them.</p>`;
  return `<div class="table-wrap"><table><thead><tr><th>Defense</th><th class="num">ASR</th><th class="num">ME</th><th class="num">DAR</th><th class="num" data-tip="Raised an alert AND the attack still succeeded">Alerted<br>& bypassed</th><th class="num">False<br>alarms</th></tr></thead><tbody>
    ${rows.map((r, i) => `<tr><td>${defChip(r.defense)} ${i === 0 ? `<span class="chip good">best</span>` : ""}</td>
      <td class="num">${pct(r.asr, 1)}</td><td class="num">${pct(r.me, 1)}</td><td class="num">${pct(r.dar, 1)}</td>
      <td class="num">${pct(r.aware_bypassed, 1)}</td><td class="num">${pct(fa[r.defense], 1)}</td></tr>`).join("")}
  </tbody></table></div>`;
}
function breakdown(s) {
  const rows = s.dims[S.dim] || [];
  const groups = [...new Set(rows.map((r) => r[S.dim]))].sort(byOrder);
  const defs = s.defenses.filter((d) => !(["me", "dar", "aware_bypassed"].includes(S.metric) && d === "none"));
  const isPct = S.metric !== "delta_si";
  const label = (g) => (S.dim === "attack" ? `${ATTACK[g] || g} <span class="muted small">${esc(ATTACK_DESC[g] || "")}</span>` : esc(g));
  return `<table class="heat"><thead><tr><th>${DIM_LABEL[S.dim]}</th>${defs.map((d) => `<th>${defChip(d)}</th>`).join("")}</tr></thead><tbody>
    ${groups.map((g) => `<tr><td>${label(g)}</td>${defs.map((d) => {
      const r = rows.find((x) => x[S.dim] === g && x.defense === d);
      if (!r) return `<td class="cell" style="${heatBg(null)}">–</td>`;
      const v = r[S.metric];
      const lo = r[S.metric + "_lo"], hi = r[S.metric + "_hi"];
      const tip = `<b>${esc(g)} · ${esc(DEF_NAME[d] || d)}</b><br>n = ${r.n}<br>ASR ${pct(r.asr, 1)}${isNum(r.asr_lo) ? ` [${pct(r.asr_lo)}–${pct(r.asr_hi)}]` : ""}` +
        (d !== "none" ? `<br>ME ${pct(r.me, 1)} · DAR ${pct(r.dar, 1)}` : "") + `<br>mean ΔSI ${signed(r.delta_si)}`;
      const style = isPct ? heatBg(v) : heatBg(Math.abs(v ?? NaN), 2);
      return `<td class="cell" style="${style}" data-tip="${esc(tip)}">${isPct ? pct(v) : signed(v)}${isNum(lo) && isPct ? `<small>${pct(lo)}–${pct(hi)}</small>` : ""}</td>`;
    }).join("")}</tr>`).join("")}
  </tbody></table>`;
}
function modelAttackHeat(s) {
  const d = S.heatDefense && s.defenses.includes(S.heatDefense) ? S.heatDefense : s.defenses[0];
  const cells = s.heat.filter((r) => r.defense === d);
  const models = [...new Set(cells.map((r) => r.model))].sort();
  const attacks = [...new Set(s.heat.map((r) => r.attack))].sort(byOrder);
  return `<table class="heat"><thead><tr><th>Model</th>${attacks.map((a) => `<th data-tip="${esc(ATTACK_DESC[a] || "")}">${ATTACK[a] || a}</th>`).join("")}</tr></thead><tbody>
    ${models.map((m) => `<tr><td>${esc(m)}</td>${attacks.map((a) => {
      const r = cells.find((x) => x.model === m && x.attack === a);
      return `<td class="cell" style="${heatBg(r?.success)}">${r ? pct(r.success) : "–"}</td>`;
    }).join("")}</tr>`).join("")}
  </tbody></table>`;
}

/* ====================================================================
   Explorer view
   ==================================================================== */
async function renderExplorer() {
  const el = $("#view-explorer");
  if (!S.file) { el.innerHTML = emptyState("No results yet", "Run an experiment first.", `<a class="btn primary" href="#runs">Go to Runs</a>`); return; }
  if (!S.trials || S.trialsFor !== S.file) {
    loading(el, "Loading trials");
    try { S.trials = await api(`/api/trials?${qs()}`); S.trialsFor = S.file; }
    catch (e) { el.innerHTML = emptyState("Could not load trials", esc(e.message)); return; }
  }
  const T = S.trials;
  const opts = (k) => [...new Set(T.map((t) => t[k]).filter(Boolean))].sort(byOrder);
  const f = S.filters;
  const sel = (k, label, values, names = {}) => `<label class="field">${label}<select data-f="${k}"><option value="">All</option>
    ${values.map((v) => `<option value="${esc(v)}" ${f[k] === v ? "selected" : ""}>${esc(names[v] || v)}</option>`).join("")}</select></label>`;
  el.innerHTML = `
    <div class="explorer">
      <div class="card">
        <div class="filters">
          ${sel("model", "Model", opts("model"))}
          ${sel("defense", "Defense", opts("defense").sort((a, b) => DEF_ORDER.indexOf(a) - DEF_ORDER.indexOf(b)), DEF_NAME)}
          ${sel("attack", "Attack", opts("attack"), ATTACK)}
          ${sel("protocol", "Protocol", opts("protocol"))}
          ${sel("outcome", "Outcome", ["success", "failed", "alerted", "bypassed", "baseline"],
            { success: "Attack succeeded", failed: "Attack failed", alerted: "Alert raised", bypassed: "Alerted & succeeded", baseline: "Clean baselines" })}
          <label class="field">Search<input type="search" data-f="q" value="${esc(f.q)}" placeholder="CV id, seniority…"></label>
        </div>
        <div class="trial-list" id="trialList"></div>
        <div class="list-foot" id="listFoot"></div>
      </div>
      <div class="card detail" id="detail">${S.selKey ? "" : emptyState("Pick a trial", "Select a trial on the left to see the CV the model received, with the hidden injection highlighted, next to the model's response.")}</div>
    </div>`;
  $$("[data-f]", el).forEach((inp) => {
    inp.addEventListener(inp.tagName === "INPUT" ? "input" : "change", () => { f[inp.dataset.f] = inp.value.trim(); S.listLimit = 200; renderList(); });
  });
  renderList();
  if (S.selKey) loadDetail(S.selKey, false);
}
function filteredTrials() {
  const f = S.filters, q = f.q.toLowerCase();
  return S.trials.filter((t) =>
    (!f.model || t.model === f.model) && (!f.defense || t.defense === f.defense) && (!f.attack || t.attack === f.attack) &&
    (!f.protocol || t.protocol === f.protocol) &&
    (!f.outcome || (f.outcome === "success" && t.success === 1) || (f.outcome === "failed" && t.success === 0) ||
      (f.outcome === "alerted" && t.alert) || (f.outcome === "bypassed" && t.alert && t.success === 1) ||
      (f.outcome === "baseline" && t.attack === "plain")) &&
    (!q || [t.cv_id, t.group_id, t.seniority, t.ad_class, t.model].some((x) => String(x || "").toLowerCase().includes(q))));
}
function renderList() {
  const rows = filteredTrials();
  const list = $("#trialList");
  list.innerHTML = rows.slice(0, S.listLimit).map((t) => {
    const outcome = t.attack === "plain" ? `<span class="chip">baseline</span>`
      : t.success === 1 ? `<span class="chip bad">succeeded</span>` : t.success === 0 ? `<span class="chip good">blocked</span>` : `<span class="chip">no baseline</span>`;
    return `<div class="trial-item ${t.key === S.selKey ? "sel" : ""}" data-key="${t.key}">
      <div class="t1">${esc(t.protocol === "multi" && t.attack === "plain" ? t.group_id : t.cv_id)} <span class="chip ${t.attack === "plain" ? "" : "accent"}">${ATTACK[t.attack] || t.attack}</span>${t.alert ? `<span class="chip warn" data-tip="The response or the defense raised an injection alert">alert</span>` : ""}</div>
      <div class="t2"><span class="dot" style="background:${DEF_COLOR[t.defense]}"></span> ${esc(t.defense)} · ${esc(t.model)} · ${esc(t.protocol)} · trial ${t.trial + 1}</div>
      <div class="t3">${outcome}<span class="small muted">${t.attack === "plain" ? `SI ${num(t.si)}` : `ΔSI ${signed(t.delta)}`}</span></div></div>`;
  }).join("") || `<div class="empty">No trials match these filters.</div>`;
  $("#listFoot").innerHTML = `<span>${Math.min(rows.length, S.listLimit)} of ${rows.length} shown</span>` +
    (rows.length > S.listLimit ? `<button class="btn sm" id="moreBtn">Show more</button>` : "");
  $("#moreBtn")?.addEventListener("click", () => { S.listLimit += 300; renderList(); });
  $$(".trial-item", list).forEach((it) => (it.onclick = () => {
    $$(".trial-item.sel", list).forEach((x) => x.classList.remove("sel"));
    it.classList.add("sel"); loadDetail(it.dataset.key);
  }));
}
async function loadDetail(key, fresh = true) {
  S.selKey = key;
  const el = $("#detail");
  if (fresh || !S.detail || S.detail.record.key !== key) {
    loading(el, "Loading trial");
    try { S.detail = await api(`/api/trial?${qs({ key })}`); S.docTab = Math.max(0, S.detail.docs.findIndex((d) => d.attack !== "plain")); }
    catch (e) { el.innerHTML = emptyState("Could not load trial", esc(e.message)); return; }
  }
  renderDetail();
}
function renderDetail() {
  const { record: r, docs, metrics: m, per_candidate: pc } = S.detail;
  const el = $("#detail");
  const attacked = r.attack !== "plain";
  const status = !attacked ? `<span class="chip">clean baseline</span>`
    : m?.success === 1 ? `<span class="chip bad">attack succeeded</span>` : m?.success === 0 ? `<span class="chip good">attack blocked</span>` : `<span class="chip">no baseline yet</span>`;
  const alert = r.defense_alert || S.detail.alert_spans.length;
  const doc = docs[S.docTab] || docs[0];
  const meta = r.defense_meta || {};
  el.innerHTML = `
    <div class="detail-head">
      <h2>${esc(attacked ? `${r.cv_id} · ${ATTACK[r.attack]}` : `Clean run · ${r.protocol === "multi" ? r.group_id : r.cv_id}`)}</h2>
      <div class="row">${status}${alert ? `<span class="chip warn">⚠ injection alert</span>` : ""}${defChip(r.defense)}
        <span class="chip">${esc(r.model)} (${esc(r.model_id)})</span><span class="chip">${esc(r.protocol)}</span><span class="chip">trial ${r.trial + 1}</span><span class="chip mono">${esc(r.prompt_id)}</span></div>
      ${attacked && m ? `<div class="metrics">
        ${metric("SI with injection", num(m.si))}${metric("SI clean baseline", num(m.si_base))}
        ${metric("ΔSI", signed(m.delta))}${r.defense !== "none" ? metric("Mitigation (ME)", pct(m.me)) : ""}
        ${isNum(m.others_delta) ? metric("Others' ΔSI", signed(m.others_delta)) : ""}
        ${metric("Latency", `${num(r.latency, 1)} s`)}${metric("Tokens", `${(r.tokens?.input || 0).toLocaleString()} → ${(r.tokens?.output || 0).toLocaleString()}`)}
      </div>` : `<div class="metrics">${metric("Latency", `${num(r.latency, 1)} s`)}${metric("Tokens", `${(r.tokens?.input || 0).toLocaleString()} → ${(r.tokens?.output || 0).toLocaleString()}`)}</div>`}
    </div>
    <div class="panes">
      <div class="pane">
        <h3>What the model received</h3>
        ${docs.length > 1 ? `<div class="doc-tabs">${docs.map((d, i) => `<button class="btn sm ${i === S.docTab ? "primary" : ""}" data-tab="${i}">${esc(d.cv_id)}${d.attack !== "plain" ? " ⚠" : ""}</button>`).join("")}</div>` : ""}
        <div class="row small muted" style="margin-bottom:6px">
          <span>${esc(doc.name)} · ${doc.attack === "plain" ? "clean CV" : `<b style="color:var(--inj-ink)">injected</b> (${esc(doc.placement)}, ${esc(obfLabel(doc.obfuscation))})`}</span>
          <span class="spacer"></span>${doc.pdf ? `<a href="/files/attacked/${esc(doc.pdf)}" target="_blank" rel="noopener">Open PDF ↗</a>` : ""}
        </div>
        <div class="textbox doc">${doc.text ? markInjection(doc.text, doc.injection) : `<span class="muted">Document text not found - was data/attacked regenerated?</span>`}</div>
        ${doc.injection ? `<p class="small muted">Highlighted: the injected text. In the PDF it is ${esc(obfLabel(doc.obfuscation))}, so a human reviewer doesn't see it, but text extraction keeps it.</p>` : ""}
        ${defenseMeta(r.defense, meta, r.defense_alert)}
      </div>
      <div class="pane">
        <h3>Model response</h3>
        <div class="textbox">${highlightResponse(r.response, S.detail.response_spans, S.detail.alert_spans)}</div>
        <div class="legend">
          ${[5, 4, 3, 2, 1].map((w) => `<span><i style="background:var(--s${w}-bg);outline:1px solid var(--s${w})"></i>${w} ${["", "critical", "negative", "neutral", "positive", "superlative"][w]}</span>`).join("")}
          <span><i style="background:var(--warn-soft);outline:1px dashed var(--warn)"></i>alert (not scored)</span>
          <span><s>struck</s> = negated</span>
        </div>
        ${pc.length > 1 ? `<h3 style="margin-top:16px">Per-candidate sentiment</h3><table><thead><tr><th>Candidate</th><th class="num">SI</th><th class="num">Terms</th><th class="num">Rank</th></tr></thead><tbody>
          ${pc.map((c) => `<tr><td>${esc(c.name)} <span class="muted small">${esc(c.cv_id)}</span>${c.cv_id === r.cv_id && attacked ? ` <span class="chip accent">attacker</span>` : ""}</td><td class="num">${num(c.si)}</td><td class="num">${c.n_terms}</td><td class="num">${c.rank ?? "–"}</td></tr>`).join("")}
        </tbody></table>` : `<p class="small muted">SI ${num(pc[0]?.si)} from ${pc[0]?.n_terms ?? 0} evaluative term(s). ${pc[0]?.n_terms < 4 ? "Few terms, so one word can swing the score." : ""}</p>`}
      </div>
    </div>`;
  $$("[data-tab]", el).forEach((b) => (b.onclick = () => { S.docTab = +b.dataset.tab; renderDetail(); }));
}
function metric(k, v) { return `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`; }
function obfLabel(o) { return { white: "white text on white", tiny: "1pt text", visible: "visible text" }[o] || o || ""; }
function markInjection(text, span) {
  if (!span) return esc(text);
  return esc(text.slice(0, span.start)) + `<mark class="inj" data-tip="Injected instruction">${esc(text.slice(span.start, span.end))}</mark>` + esc(text.slice(span.end));
}
function markTerms(text, s, e, terms) {
  let out = "", pos = s;
  for (const t of terms) {
    if (t.end <= s || t.start >= e) continue;
    const a = Math.max(t.start, s), b = Math.min(t.end, e);
    out += esc(text.slice(pos, a));
    out += `<mark class="term w${t.weight}${t.negated ? " neg" : ""}" data-tip="<b>${esc(t.term)}</b> → ${t.weight}${t.negated ? " (negated)" : ""}">${esc(text.slice(a, b))}</mark>`;
    pos = b;
  }
  return out + esc(text.slice(pos, e));
}
function highlightResponse(text, terms, alerts) {
  let out = "", pos = 0;
  for (const a of alerts) {
    out += markTerms(text, pos, a.start, terms);
    out += `<span class="alert-s" data-tip="Alert sentence - removed before scoring">${markTerms(text, a.start, a.end, terms)}</span>`;
    pos = a.end;
  }
  return out + markTerms(text, pos, text.length, terms);
}
function defenseMeta(defense, meta, alert) {
  if (defense === "prefilter" && meta.prefilter) {
    const items = Object.entries(meta.prefilter);
    return `<h3 style="margin-top:14px">Pre-filter report</h3>${items.map(([cv, v]) =>
      `<div class="small"><b>${esc(cv)}</b>: ${v.hidden_text_removed ? `<span class="chip warn">hidden text removed</span>` : `<span class="chip">no hidden text</span>`}
       ${v.flagged_sentences.length ? `${v.flagged_sentences.length} sentence(s) removed:<ul>${v.flagged_sentences.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` : ""}</div>`).join("")}
       <p class="small muted">The text above is what was extracted. The model received it with the hidden and flagged parts removed.</p>`;
  }
  if (defense === "sanitize") {
    return `<h3 style="margin-top:14px">Sanitizer</h3><p class="small">${meta.flagged_docs?.length ? `Pass 1 flagged suspicious content in <b>${meta.flagged_docs.join(", ")}</b>.` : "Pass 1 did not flag anything."} The model only saw the JSON extracted in pass 1.</p>`;
  }
  if (defense === "spotlight") return `<p class="small muted" style="margin-top:12px">Spotlighting: the model received this text datamarked (every space replaced with ˆ) inside &lt;&lt;DOC&gt;&gt; delimiters.</p>`;
  if (defense === "ope") return `<p class="small muted" style="margin-top:12px">OPE: the text was wrapped in UNTRUSTED markers, with the paper's three-layer instructions.</p>`;
  return "";
}

/* ====================================================================
   Dataset view
   ==================================================================== */
async function renderDataset() {
  const el = $("#view-dataset");
  if (!S.dataset) { loading(el, "Loading dataset"); S.dataset = await api("/api/dataset"); }
  const { cvs, variants } = S.dataset;
  if (!cvs.length) {
    el.innerHTML = emptyState("No dataset yet", "Generate the 27 synthetic CVs and their attacked variants from the Runs page.", `<a class="btn primary" href="#runs">Go to Runs</a>`);
    return;
  }
  const sen = ["Junior", "Mid", "Senior"], ad = ["High", "Medium", "Low"];
  const words = { Junior: "~500 words", Mid: "~1,200 words", Senior: "~2,500 words" };
  el.innerHTML = `
    <div class="grid split split-ds" id="dsGrid">
      <div class="card card-pad">
        <h2>27 synthetic CVs</h2>
        <p class="sub">Seniority (a proxy for length) × achievement density (measurable achievements per 100 words). Every person and organisation is fictional.</p>
        <div class="cv-grid">
          <div></div>${ad.map((a) => `<div class="hd" data-tip="${a === "High" ? "AD ≥ 1.5" : a === "Low" ? "AD ≤ 0.5" : "0.5 < AD < 1.5"}">AD: ${a}</div>`).join("")}
          ${sen.map((s) => `<div class="rowhd">${s}<small>${words[s]}</small></div>${ad.map((a) => `<div class="cell-box">
            ${cvs.filter((c) => c.seniority === s && c.ad_class === a).map((c) => `<button class="cv-card ${c.id === S.selCv ? "sel" : ""}" data-cv="${c.id}">
              <div class="n">${esc(c.name)}</div><div class="r"><span>${esc(c.role)}</span><span class="mono">${esc(c.id)}</span></div>
              <div class="r"><span>${c.word_count} words</span><span>AD ${(+c.ad_score).toFixed(2)}</span></div></button>`).join("")}
          </div>`).join("")}`).join("")}
        </div>
        <p class="small muted" style="margin-top:12px">Attacked variants built: ${variants.length ? variants.map((v) => `<span class="chip">${esc(v.replace("_", " · "))}</span>`).join(" ") : "none yet"}</p>
      </div>
      <div class="card" id="cvPanel">${S.selCv ? "" : emptyState("Pick a CV", "See the PDF as a recruiter would, next to the text an LLM actually reads.")}</div>
    </div>`;
  $$(".cv-card", el).forEach((b) => (b.onclick = () => { S.selCv = b.dataset.cv; S.cvVariant = "plain"; renderDataset(); }));
  if (S.selCv) renderCvPanel();
}
async function renderCvPanel() {
  const el = $("#cvPanel");
  const cv = S.dataset.cvs.find((c) => c.id === S.selCv);
  const opts = [["plain", "Clean"]];
  for (const v of S.dataset.variants) for (const a of ["pos", "neg", "posneg", "posmed"]) opts.push([`${a}|${v}`, `${ATTACK[a]} · ${v.replace("_", ", ")}`]);
  const [attack, variant] = S.cvVariant === "plain" ? ["plain", ""] : S.cvVariant.split("|");
  loading(el, "Loading CV");
  let info;
  try { info = await api(`/api/cv/${encodeURIComponent(S.selCv)}?attack=${attack}&variant=${encodeURIComponent(variant)}`); }
  catch (e) { el.innerHTML = emptyState("Could not load CV", esc(e.message)); return; }
  const text = info.extracted ?? info.text;
  el.innerHTML = `
    <div class="detail-head">
      <h2>${esc(cv.name)} <span class="muted small mono">${esc(cv.id)}</span></h2>
      <div class="row"><span class="chip">${esc(cv.role)}</span><span class="chip">${esc(cv.seniority)}</span><span class="chip">AD ${esc(cv.ad_class)} (${(+cv.ad_score).toFixed(2)})</span>
        <span class="chip">${cv.word_count} words · ${cv.n_achievements} achievements</span><span class="spacer"></span>
        <label class="field" style="min-width:220px">Variant<select id="variantSel">${opts.map(([k, l]) => `<option value="${k}" ${k === S.cvVariant ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></label></div>
      ${info.injection_text ? `<div class="banner warn" style="margin:12px 0 0">Hidden instruction: “${esc(info.injection_text)}”&nbsp;·&nbsp;check ${info.verified === "True" ? "passed" : "failed"} (contrast ${esc(info.max_contrast || "–")}, smallest font ${esc(info.min_font_size || "–")} pt)</div>` : ""}
    </div>
    <div class="panes">
      <div class="pane"><h3>PDF: what a recruiter sees ${info.pdf ? `<a class="small" style="margin-left:auto;font-weight:450" href="/files/attacked/${esc(info.pdf)}" target="_blank" rel="noopener">open ↗</a>` : ""}</h3>
        ${info.pdf ? `<iframe title="CV PDF" src="/files/attacked/${esc(info.pdf)}#toolbar=0&view=FitH" style="width:100%;height:620px;border:1px solid var(--border);border-radius:8px;background:#fff"></iframe>` : `<p class="muted small">No PDF found. Build the attacked variants first.</p>`}</div>
      <div class="pane"><h3>Extracted text: what the LLM reads</h3><div class="textbox doc" style="max-height:620px">${markInjection(text, info.injection)}</div></div>
    </div>`;
  $("#variantSel").onchange = (e) => { S.cvVariant = e.target.value; renderCvPanel(); };
}

/* ====================================================================
   Runs view
   ==================================================================== */
const RUN_FORM = { models: [], protocol: ["single", "multi"], defenses: ["none", "ope"], trials: 10, temperature: 0, cvs: "", placement: "bottom", obfuscation: "white", group_by: "cell", workers: 1, out: "raw" };
async function renderRuns() {
  const el = $("#view-runs");
  if (!S.models || !S.config) {
    loading(el, "Checking models");
    [S.models, S.config, S.dataset] = await Promise.all([api("/api/models"), api("/api/config"), api("/api/dataset")]);
    if (!RUN_FORM.models.length) RUN_FORM.models = [S.models.models.find((m) => m.ready && m.provider !== "mock")?.key || "mock"];
  }
  const F = RUN_FORM;
  const ds = S.dataset;
  const checks = (name, values, labels = {}, disabled = {}) => `<div class="checks">${values.map((v) =>
    `<label class="check ${disabled[v] ? "disabled" : ""}" ${disabled[v] ? `data-tip="${esc(disabled[v])}"` : ""}><input type="checkbox" name="${name}" value="${esc(v)}" ${F[name].includes(v) ? "checked" : ""}>${esc(labels[v] || v)}</label>`).join("")}</div>`;
  const notReady = Object.fromEntries(S.models.models.filter((m) => !m.ready).map((m) => [m.key, m.note]));
  el.innerHTML = `
    <div class="grid split split-runs" id="runsGrid">
      <div class="stack">
        <div class="card card-pad">
          <h2>New experiment</h2>
          <p class="sub">Runs <span class="mono">runner.py</span> in the background. Results go to <span class="mono">results/&lt;name&gt;.jsonl</span>. Reusing a name resumes that run and skips trials already done.</p>
          <div class="form-grid">
            <div class="field wide">Models ${checks("models", S.models.models.map((m) => m.key), Object.fromEntries(S.models.models.map((m) => [m.key, `${m.key} · ${m.model}`])), notReady)}</div>
            <div class="field">Protocol ${checks("protocol", ["single", "multi"], { single: "Single-doc (p.Pos, p.Neg)", multi: "Multi-doc (p.Pos-Neg, p.Pos-Med)" })}</div>
            <div class="field wide">Defenses ${checks("defenses", S.config.defenses, DEF_NAME)}</div>
            <label class="field">Trials per condition<input type="number" name="trials" min="1" max="100" value="${F.trials}"></label>
            <label class="field">Temperature<input type="number" name="temperature" min="0" max="2" step="0.1" value="${F.temperature}"></label>
            <label class="field">Placement<select name="placement">${S.config.placements.map((p) => `<option ${p === F.placement ? "selected" : ""}>${p}</option>`).join("")}</select></label>
            <label class="field">Obfuscation<select name="obfuscation">${S.config.obfuscations.map((p) => `<option value="${p}" ${p === F.obfuscation ? "selected" : ""}>${obfLabel(p)}</option>`).join("")}</select></label>
            <label class="field">Multi-doc session size<select name="group_by">${[["cell", "3 CVs (one grid cell)"], ["seniority", "9 CVs (one seniority)"], ["all", "all 27 CVs"]].map(([v, l]) => `<option value="${v}" ${v === F.group_by ? "selected" : ""}>${l}</option>`).join("")}</select></label>
            <label class="field">Parallel requests per model<input type="number" name="workers" min="1" max="16" value="${F.workers}"></label>
            <label class="field wide">Only these CVs <span class="muted">(optional, space-separated, e.g. J-H-1 M-M-2 S-L-3)</span><input type="text" name="cvs" value="${esc(F.cvs)}" placeholder="all 27"></label>
            <label class="field">Output name<input type="text" name="out" value="${esc(F.out)}" pattern="[A-Za-z0-9_\\-]+"></label>
          </div>
          <div class="row" style="margin-top:16px">
            <button class="btn" id="estBtn">Estimate size &amp; cost</button>
            <button class="btn primary" id="startBtn">Start run</button>
            <span class="small muted" id="formNote"></span>
          </div>
          <div id="planBox"></div>
        </div>
        <div class="card"><div class="card-pad" style="padding-bottom:6px"><h2>Jobs</h2><p class="sub">Started from this window while the server is running. Stopping is safe: a run resumes from where it stopped.</p></div><div id="jobs"></div></div>
      </div>
      <div class="stack">
        <div class="card card-pad">
          <h2>Models</h2>
          <p class="sub">${S.models.ollama_up ? `<span class="chip good">Ollama running</span>` : `<span class="chip bad">Ollama not running</span>`} Configure models in <span class="mono">config.yaml</span>.</p>
          ${S.models.models.map((m) => `<div class="model-row"><span class="dot" style="background:${m.ready ? "var(--good)" : "var(--border-strong)"}"></span>
            <b>${esc(m.key)}</b><span class="muted small">${esc(m.provider)}</span><span class="spacer"></span><span class="small ${m.ready ? "muted" : ""}">${esc(m.note)}</span></div>`).join("")}
          <button class="btn sm" style="margin-top:10px" id="refreshModels">Recheck</button>
        </div>
        <div class="card card-pad">
          <h2>Data</h2>
          <p class="sub">${ds.cvs.length ? `${ds.cvs.length} CVs · variants: ${ds.variants.length ? ds.variants.join(", ") : "none"}` : "No dataset yet."}</p>
          <div class="stack" style="gap:10px">
            <button class="btn" data-prep="dataset">${ds.cvs.length ? "Regenerate" : "Generate"} 27 CVs</button>
            <div class="row"><select id="prepPlacement">${S.config.placements.map((p) => `<option ${p === "bottom" ? "selected" : ""}>${p}</option>`).join("")}</select>
              <select id="prepObf">${S.config.obfuscations.map((p) => `<option value="${p}" ${p === "white" ? "selected" : ""}>${obfLabel(p)}</option>`).join("")}</select></div>
            <button class="btn" data-prep="inject">Build attacked variants</button>
            <button class="btn sm" data-prep="inject-all" data-tip="1,296 PDFs. Takes about 15 minutes.">Build all 12 variants</button>
          </div>
        </div>
      </div>
    </div>`;
  bindRunForm(el);
  renderJobs();
}
function readForm(el) {
  const F = RUN_FORM;
  for (const k of ["models", "protocol", "defenses"]) F[k] = $$(`input[name=${k}]:checked`, el).map((i) => i.value);
  for (const k of ["trials", "workers"]) F[k] = parseInt($(`[name=${k}]`, el).value, 10) || 1;
  F.temperature = parseFloat($("[name=temperature]", el).value) || 0;
  for (const k of ["placement", "obfuscation", "group_by", "cvs", "out"]) F[k] = $(`[name=${k}]`, el).value.trim();
  return { ...F, cvs: F.cvs ? F.cvs.split(/[\s,]+/).filter(Boolean) : [] };
}
function bindRunForm(el) {
  const note = $("#formNote");
  $("#estBtn").onclick = async () => {
    const p = readForm(el);
    $("#planBox").innerHTML = `<div class="loading"><span class="spin"></span> Planning…</div>`;
    try { S.plan = await api("/api/dry-run", { method: "POST", body: JSON.stringify(p) }); renderPlan(); }
    catch (e) { $("#planBox").innerHTML = `<div class="banner warn" style="margin-top:14px">${esc(e.message)}</div>`; }
  };
  $("#startBtn").onclick = async () => {
    const p = readForm(el);
    const notReady = S.models.models.filter((m) => p.models.includes(m.key) && !m.ready);
    if (notReady.length && !confirm(`${notReady.map((m) => `${m.key}: ${m.note}`).join("\n")}\n\nStart anyway?`)) return;
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify(p) });
      toast("Run started"); note.textContent = ""; pollRuns(true);
    } catch (e) { toast(e.message, true); }
  };
  $("#refreshModels").onclick = async () => { S.models = await api("/api/models"); renderRuns(); };
  $$("[data-prep]", el).forEach((b) => (b.onclick = async () => {
    const step = b.dataset.prep;
    const body = step === "dataset" ? { step } : { step: "inject", placement: $("#prepPlacement").value, obfuscation: $("#prepObf").value, all_variants: step === "inject-all" };
    if (step === "dataset" && S.dataset.cvs.length && !confirm("Regenerate the CVs? Same seed, so they come out identical unless you've edited the generator.")) return;
    try { await api("/api/prepare", { method: "POST", body: JSON.stringify(body) }); toast("Job started"); pollRuns(true); }
    catch (e) { toast(e.message, true); }
  }));
}
function renderPlan() {
  const p = S.plan;
  const hours = (n) => (n * 60) / 3600;
  $("#planBox").innerHTML = `
    <div style="margin-top:16px">
      <div class="row"><b>${(p.total ?? 0).toLocaleString()} trials</b><span class="muted">${(p.done ?? 0).toLocaleString()} already done · ${(p.to_run ?? 0).toLocaleString()} to run</span>
        <span class="spacer"></span><span class="chip ${p.usd > 0 ? "warn" : "good"}">≈ $${num(p.usd, 2)}</span></div>
      <div class="table-wrap" style="margin-top:8px"><table><thead><tr><th>Model</th><th>Protocol</th><th>Defense</th><th class="num">Trials</th><th class="num">Calls</th><th class="num">Input tok</th><th class="num">Output tok</th><th class="num">USD</th></tr></thead><tbody>
        ${p.rows.map((r) => `<tr><td>${esc(r.model)}</td><td>${esc(r.protocol)}</td><td>${defChip(r.defense)}</td><td class="num">${r.trials.toLocaleString()}</td><td class="num">${r.calls.toLocaleString()}</td><td class="num">${r.input_tokens.toLocaleString()}</td><td class="num">${r.output_tokens.toLocaleString()}</td><td class="num">${num(r.usd)}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Token counts are rough (≈4 characters per token) and prices in config.yaml are approximate. A local 7B model on CPU takes about 1 min per call, so this would be ~${num(hours(p.rows.reduce((a, r) => a + r.calls, 0)), 0)} h with one worker.</p>
    </div>`;
}
function renderJobs() {
  const el = $("#jobs"); if (!el) return;
  if (!S.runs.length) { el.innerHTML = `<div class="empty small">No jobs started yet.</div>`; return; }
  el.innerHTML = S.runs.map((j) => {
    const p = j.total ? Math.min(100, (j.done / j.total) * 100) : j.state === "finished" ? 100 : 0;
    const chip = { running: "accent", finished: "good", failed: "bad", stopped: "warn" }[j.state];
    const dur = ((j.ended || Date.now() / 1000) - j.started) / 60;
    return `<div class="job ${j.state}">
      <div class="row"><span class="chip ${chip}">${j.state === "running" ? `<span class="spin" style="width:10px;height:10px;border-width:1.5px"></span>` : ""} ${j.state}</span>
        <b>${esc(j.label)}</b><span class="spacer"></span>
        ${j.state === "running" ? `<button class="btn sm danger" data-stop="${j.id}">Stop</button>` : ""}
        ${j.kind === "experiment" && j.out ? `<button class="btn sm" data-open="${esc(j.out)}">View results</button>` : ""}</div>
      ${j.kind === "experiment" ? `<div class="progress"><div style="width:${p}%"></div></div>
        <div class="small muted">${j.done.toLocaleString()} / ${(j.total || 0).toLocaleString()} trials · ${j.errors} error(s) · ${num(dur, 1)} min elapsed${j.state === "running" && isNum(j.eta_min) ? ` · ~${num(j.eta_min, 0)} min left` : ""}</div>`
        : `<div class="small muted" style="margin-top:4px">${num(dur, 1)} min</div>`}
      <details ${j.state === "failed" ? "open" : ""}><summary>Log · <span class="mono">${esc(j.command)}</span></summary><pre class="log">${esc(j.log_tail || "(no output yet)")}</pre></details>
    </div>`;
  }).join("");
  $$("[data-stop]", el).forEach((b) => (b.onclick = async () => { await api(`/api/runs/${b.dataset.stop}/stop`, { method: "POST" }); pollRuns(true); }));
  $$("[data-open]", el).forEach((b) => (b.onclick = async () => {
    await loadFiles(false); S.file = b.dataset.open; invalidate();
    $("#fileSelect").value = S.file; location.hash = "#results";
  }));
}
let _pollTimer = null, _wasRunning = false;
async function pollRuns(now = false) {
  clearTimeout(_pollTimer);
  try { S.runs = await api("/api/runs"); } catch { /* server restarting */ }
  const running = S.runs.some((j) => j.state === "running");
  $("#runBadge").hidden = !running;
  if (S.view === "runs") renderJobs();
  if (_wasRunning && !running) {               // a job just ended: refresh what may have changed
    S.dataset = null; await loadFiles(); invalidate();
    if (S.view !== "runs") render(); else { S.dataset = await api("/api/dataset"); }
  }
  _wasRunning = running;
  _pollTimer = setTimeout(pollRuns, running ? 2000 : 8000);
}

/* ====================================================================
   Boot
   ==================================================================== */
(async function boot() {
  try { await loadFiles(); }
  catch (e) { toast("Cannot reach the server: " + e.message, true); }
  route();
  pollRuns();
})();
