"use strict";
/* ====================================================================
   OPE Analyze: run the paper's defense architecture on your own documents
   Isolation -> Meta-instruction -> LLM -> Audit -> Remediation -> Secure output
   ==================================================================== */
const OPE = {
  task: "Give me a summary evaluation of the previous CV in three short paragraphs",
  docs: [], model: null, mode: "compare", evidence: false, busy: false, result: null, nextId: 1,
};
const OPE_PROMPTS = {
  single: "Give me a summary evaluation of the previous CV in three short paragraphs",
  multi: "Give me a summary evaluation of the previous CVs in three short paragraphs for each CV and a summary comparison table of all the CVs",
};
const OPE_STATUS = {
  safe: { label: "SAFE", cls: "good", icon: "✓" },
  alerted: { label: "INJECTION DETECTED", cls: "warn", icon: "!" },
  remediated: { label: "REMEDIATED", cls: "accent", icon: "↻" },
  remediation_failed: { label: "REMEDIATION FAILED", cls: "bad", icon: "✕" },
  baseline: { label: "UNPROTECTED (baseline)", cls: "", icon: "–" },
};

async function renderAnalyze() {
  const el = $("#view-analyze");
  if (!S.models) { loading(el, "Checking models"); S.models = await api("/api/models"); }
  if (!S.dataset) S.dataset = await api("/api/dataset");
  if (!OPE.model) OPE.model = S.models.models.find((m) => m.ready && m.provider !== "mock")?.key || "mock";
  const ds = S.dataset;
  el.innerHTML = `
    <div class="banner info"><div>
      Implements the paper's <b>Outer Prompt Extension</b> (Section 4.2): an isolation layer, then meta-instructions, the LLM, an audit, and remediation if needed. Add documents, which may be untrusted, and a task, then compare the unprotected answer with the protected one.
    </div></div>
    <div class="ope-arch card card-pad" aria-label="OPE architecture">
      ${["User task", "1 · Isolation", "2 · Meta-instruction", "LLM", "3 · Audit", "Remediation", "Secure output"]
        .map((s, i) => `<span class="ope-node ${i === 0 || i === 6 ? "end" : ""}">${s}</span>`).join(`<span class="ope-arrow">→</span>`)}
    </div>
    <div class="grid split split-ope" id="opeGrid">
      <div class="card card-pad stack" style="gap:14px">
        <div>
          <div class="row"><h2>User task</h2><span class="spacer"></span>
            <button class="btn sm" data-preset="single">Single-CV prompt</button><button class="btn sm" data-preset="multi">Multi-CV prompt</button></div>
          <textarea id="opeTask" rows="3" class="ta">${esc(OPE.task)}</textarea>
        </div>
        <div>
          <div class="row"><h2>Documents <span class="muted small">(untrusted content)</span></h2></div>
          <div id="opeDocs" class="stack" style="gap:10px;margin-top:8px"></div>
          <div class="row" style="margin-top:10px">
            <button class="btn sm" id="addText">+ Paste text</button>
            <label class="btn sm">Upload PDF / DOCX / TXT<input type="file" id="opeFile" accept=".pdf,.docx,.txt,.md" multiple hidden></label>
          </div>
          ${ds.cvs.length ? `<div class="row" style="margin-top:8px">
            <select id="dsCv">${ds.cvs.map((c) => `<option value="${c.id}">${esc(c.id)} · ${esc(c.name)} (${esc(c.role)})</option>`).join("")}</select>
            <select id="dsAttack"><option value="plain">clean</option>${ds.variants.length ? ["pos", "neg", "posneg", "posmed"].map((a) => `<option value="${a}">${ATTACK[a]} (injected)</option>`).join("") : ""}</select>
            <button class="btn sm" id="addDs">+ Add from dataset</button></div>` : ""}
        </div>
        <div class="form-grid" style="grid-template-columns:repeat(auto-fit,minmax(160px,1fr))">
          <label class="field">Model<select id="opeModel">${S.models.models.map((m) => `<option value="${m.key}" ${m.key === OPE.model ? "selected" : ""} ${m.ready ? "" : "disabled"}>${esc(m.key)} · ${esc(m.model)}${m.ready ? "" : " (not ready)"}</option>`).join("")}</select></label>
          <div class="field">Mode<div class="seg" id="opeMode">${[["compare", "Compare both"], ["ope", "OPE only"], ["baseline", "Baseline only"]].map(([k, l]) => `<button data-v="${k}" class="${k === OPE.mode ? "on" : ""}">${l}</button>`).join("")}</div></div>
        </div>
        <label class="check" style="align-self:flex-start"><input type="checkbox" id="opeEvidence" ${OPE.evidence ? "checked" : ""}> Researcher view (show audit evidence, including the detected statements)</label>
        <div class="row"><button class="btn primary" id="opeRun" ${OPE.busy ? "disabled" : ""}>${OPE.busy ? `<span class="spin" style="width:14px;height:14px;border-width:2px"></span> Analysing…` : "Analyse"}</button>
          <span class="small muted">${OPE.model && OPE.model !== "mock" ? "Local models take 1–4 minutes per call; OPE may use up to 4 calls." : ""}</span></div>
      </div>
      <div id="opeResult">${OPE.result ? "" : emptyState("No analysis yet", "Add at least one document and press <b>Analyse</b>. Try a CV from the dataset with an injected variant to see the defense work.")}</div>
    </div>`;
  renderOpeDocs();
  if (OPE.result) renderOpeResult();
  bindAnalyze(el);
}

function renderOpeDocs() {
  const box = $("#opeDocs");
  if (!OPE.docs.length) { box.innerHTML = `<div class="empty small" style="padding:18px">No documents yet.</div>`; return; }
  box.innerHTML = OPE.docs.map((d, i) => `
    <div class="doc-card">
      <div class="row"><input type="text" value="${esc(d.id)}" data-i="${i}" data-k="id" style="width:120px" aria-label="Document id">
        <input type="text" value="${esc(d.name)}" data-i="${i}" data-k="name" placeholder="name" style="flex:1" aria-label="Document name">
        <span class="chip">${esc(d.source)}</span>${d.tag ? `<span class="chip warn">${esc(d.tag)}</span>` : ""}
        <button class="btn sm" data-rm="${i}" aria-label="Remove document">✕</button></div>
      <textarea rows="6" class="ta mono" data-i="${i}" data-k="content">${esc(d.content)}</textarea>
      <div class="small muted">${d.content.split(/\s+/).filter(Boolean).length} words</div>
    </div>`).join("");
  $$("[data-k]", box).forEach((inp) => inp.addEventListener("input", () => { OPE.docs[+inp.dataset.i][inp.dataset.k] = inp.value; }));
  $$("[data-rm]", box).forEach((b) => (b.onclick = () => { OPE.docs.splice(+b.dataset.rm, 1); renderOpeDocs(); }));
}

function addDoc(d) {
  let id = d.id || `doc${OPE.nextId++}`;
  while (OPE.docs.some((x) => x.id === id)) id = `${id}-${OPE.nextId++}`;
  OPE.docs.push({ ...d, id });
  renderOpeDocs();
}

function bindAnalyze(el) {
  $("#opeTask").addEventListener("input", (e) => (OPE.task = e.target.value));
  $$("[data-preset]", el).forEach((b) => (b.onclick = () => { OPE.task = OPE_PROMPTS[b.dataset.preset]; $("#opeTask").value = OPE.task; }));
  $("#addText").onclick = () => addDoc({ name: "", content: "", source: "text" });
  $("#opeFile").onchange = async (e) => {
    for (const f of e.target.files) {
      if (f.size > 10e6) { toast(`${f.name} is larger than 10 MB`, true); continue; }
      const b64 = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result.split(",")[1]); r.onerror = rej; r.readAsDataURL(f); });
      try {
        const d = await api("/api/ope/load", { method: "POST", body: JSON.stringify({ filename: f.name, data_base64: b64 }) });
        addDoc({ id: f.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_\-]/g, "_").slice(0, 40), name: f.name, content: d.content, source: d.source });
      } catch (err) { toast(err.message, true); }
    }
    e.target.value = "";
  };
  $("#addDs")?.addEventListener("click", async () => {
    const cv = $("#dsCv").value, attack = $("#dsAttack").value;
    const variant = attack === "plain" ? "" : S.dataset.variants[0];
    try {
      const d = await api(`/api/cv/${encodeURIComponent(cv)}?attack=${attack}&variant=${encodeURIComponent(variant)}`);
      const c = S.dataset.cvs.find((x) => x.id === cv);
      addDoc({ id: cv, name: c.name, content: d.extracted ?? d.text, source: "pdf (dataset)", tag: attack === "plain" ? "" : `${ATTACK[attack]} injected` });
    } catch (err) { toast(err.message, true); }
  });
  $("#opeModel").onchange = (e) => (OPE.model = e.target.value);
  $$("#opeMode button").forEach((b) => (b.onclick = () => { OPE.mode = b.dataset.v; $$("#opeMode button").forEach((x) => x.classList.toggle("on", x === b)); }));
  $("#opeEvidence").onchange = (e) => (OPE.evidence = e.target.checked);
  $("#opeRun").onclick = runOpe;
}

async function runOpe() {
  const docs = OPE.docs.filter((d) => d.content.trim());
  if (!OPE.task.trim()) return toast("Write a task first", true);
  if (!docs.length) return toast("Add at least one document with text", true);
  if (new Set(docs.map((d) => d.id)).size !== docs.length) return toast("Document ids must be unique", true);
  OPE.busy = true; renderAnalyze();
  $("#opeResult").innerHTML = `<div class="card loading"><span class="spin"></span> Running ${OPE.mode === "compare" ? "baseline and OPE" : OPE.mode}… this can take a few minutes on local models.</div>`;
  try {
    OPE.result = await api("/api/ope/analyze", { method: "POST", body: JSON.stringify({
      user_prompt: OPE.task, model: OPE.model, mode: OPE.mode, include_evidence: OPE.evidence,
      documents: docs.map((d) => ({ id: d.id, name: d.name || d.id, content: d.content })) }) });
    OPE.result._evidence = OPE.evidence;
  } catch (e) { toast(e.message, true); OPE.result = null; }
  OPE.busy = false; renderAnalyze();
}

function renderOpeResult() {
  const R = OPE.result;
  const modes = ["baseline", "ope"].filter((m) => R[m]);
  $("#opeResult").innerHTML = `<div class="${modes.length > 1 ? "ope-compare" : ""}">${modes.map((m) => opeCard(m, R[m])).join("")}</div>`;
}

function opeCard(mode, r) {
  const s = r.security, st = OPE_STATUS[s.final_status] || OPE_STATUS.baseline;
  const ope = mode === "ope";
  const steps = ope ? [
    ["Isolation", "done"], ["Meta-instruction", "done"], ["LLM", "done"],
    ["Audit", s.injection_detected ? "warn" : "done", s.injection_detected ? "injection detected" : "clear"],
    ["Remediation", !s.remediation_triggered ? "skip" : s.final_status === "remediated" ? "done" : "bad",
      !s.remediation_triggered ? "not needed" : s.final_status === "remediated" ? "output regenerated" : "still biased"],
  ] : [["LLM", "done"]];
  return `<div class="card">
    <div class="detail-head">
      <div class="row"><h2 style="margin:0">${ope ? "Protected by OPE" : "Unprotected baseline"}</h2><span class="spacer"></span>
        <span class="status-pill ${st.cls}">${st.icon} ${st.label}</span></div>
      <p class="small muted" style="margin:8px 0 0">${esc(s.message)}</p>
      <div class="ope-steps">${steps.map(([n, cls, note]) => `<span class="ope-step ${cls}">${n}${note ? `<small>${esc(note)}</small>` : ""}</span>`).join(`<span class="ope-arrow">→</span>`)}</div>
      <div class="row small muted" style="margin-top:8px">${r.calls} LLM call(s) · ${num(r.latency, 1)} s · ${r.tokens.toLocaleString()} tokens
        ${r.sentiment.map((x) => `<span class="chip" data-tip="Sentiment Index for ${esc(x.name)}, 1 = critical … 5 = glowing (${x.terms} scored terms)">SI ${esc(x.id)}: <b>${num(x.si)}</b></span>`).join("")}</div>
    </div>
    <div class="pane"><h3>Answer</h3><div class="textbox">${esc(r.response)}</div>
      ${ope && r.audits && OPE.result._evidence ? opeEvidence(r.audits) : ""}</div>
  </div>`;
}

function opeEvidence(audits) {
  return `<details open style="margin-top:12px"><summary><b>Audit evidence</b> (researcher view)</summary>
    ${audits.map((a, i) => `<div class="evidence">
      <div class="row small"><b>${i === 0 ? "First draft" : `Remediation attempt ${i}`}</b>
        <span class="chip ${a.final_action === "remediate" ? "bad" : a.final_action === "alert" ? "warn" : "good"}">${a.final_action}</span>
        <span class="muted">confidence ${num(a.confidence)}</span></div>
      <div class="small" style="margin-top:4px">${esc(a.reason)}</div>
      <table class="small" style="margin-top:6px"><tbody>
        <tr><td>Model raised its own alert</td><td>${a.model_alert ? "yes" : "no"}</td></tr>
        <tr><td>Verifier: draft follows injection</td><td>${a.verifier_verdict == null ? "–" : a.verifier_verdict ? "yes" : "no"}</td></tr>
        <tr><td>Anchored evaluative terms</td><td>${a.anchored_terms?.length ? a.anchored_terms.map(esc).join(", ") : "none"}</td></tr>
      </tbody></table>
      ${Object.keys(a.flagged_statements || {}).length ? `<div class="small muted" style="margin-top:6px">Detected statements (shown for research only):</div>
        ${Object.entries(a.flagged_statements).map(([d, ss]) => ss.map((s) => `<div class="quarantine"><b>${esc(d)}</b>: ${esc(s)}</div>`).join("")).join("")}` : ""}
    </div>`).join("")}</details>`;
}
