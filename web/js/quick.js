// Quick access: the knobs people actually turn, as three strips at the top of a
// model's editor - Load, Thinking, Sampling. Which knobs sit in which strip,
// their labels and the few value lists --help only spells out in prose are data
// (QUICK, below), so a new build or a fork extends the strips without touching
// the renderer: a key this build's --help doesn't print is simply left out.
//
// Load and Sampling are part of the editor's form: their fields carry data-k
// like the knob grid's, so Save + Reload, presets and Refine see them, and the
// grid leaves those keys out (two inputs for one key would race on Save).
// Thinking applies at once, like the switches it grew from, so it carries no
// data-k and is redrawn from the server's state on every poll.
import { esc } from "./core.js";

export const QUICK = {
  load: {
    title: "Load",
    hint: "Applies on the next load. Saving unloads the model.",
    context: "ctx-size",
    placement: {fit: "fit", moe: "n-cpu-moe", dense: "gpu-layers"},
    fields: [
      {key: "cache-type-k", label: "KV cache K"},
      {key: "cache-type-v", label: "KV cache V"},
      {key: "kv-tail-tokens", label: "Exact tail"},
      {key: "parallel", label: "Slots"},
      {key: "flash-attn", label: "Flash attn", ui: "seg"},
      {key: "batch-size", label: "Batch"},
      {key: "ubatch-size", label: "Micro-batch"},   // not "µbatch": uppercase turns µ into a Greek M
    ],
  },
  thinking: {
    title: "Thinking",
    hint: "Applies at once.",
    mode: {key: "reasoning", options: [["", "Auto"], ["on", "On"], ["off", "Off"]]},
    budget: {key: "reasoning-budget",
             options: [["", "unlimited"], ["512", "512 tokens"], ["2048", "2k tokens"], ["8192", "8k tokens"]]},
    // --help lists these in prose, so the parser reads them as free strings
    fields: [
      {key: "reasoning-format", label: "Format", options: ["auto", "none", "deepseek", "deepseek-legacy"]},
      {key: "reasoning-effort", label: "Effort", options: ["default", "minimal", "low", "medium", "high", "xhigh"]},
    ],
  },
  sampling: {
    title: "Sampling",
    hint: "Defaults. A client's request overrides them.",
    shown: [
      {key: "temp", label: "Temp"}, {key: "top-p", label: "Top-p"}, {key: "top-k", label: "Top-k"},
      {key: "min-p", label: "Min-p"}, {key: "repeat-penalty", label: "Repeat penalty"},
    ],
    more: [
      {key: "presence-penalty", label: "Presence"}, {key: "frequency-penalty", label: "Frequency"},
      {key: "repeat-last-n", label: "Repeat window"}, {key: "seed", label: "Seed"},
      {key: "dry-multiplier", label: "DRY"}, {key: "xtc-probability", label: "XTC"},
    ],
  },
};

const CTX_MIN = 2048, CTX_STEP = 1024, CTX_FALLBACK = 32768;

/* ---------- reading the schema, the GGUF and the section ---------- */

// name -> knob, by key and by every alias (--gpu-layers is also --n-gpu-layers)
export function knobFinder(schema) {
  const byName = new Map();
  for (const g of (schema && schema.groups) || [])
    for (const k of g.knobs)
      for (const n of [k.key, ...(k.aliases || [])]) if (!byName.has(n)) byName.set(n, k);
  return name => byName.get(name) || null;
}

// The spelling to write: the one the section already uses, so a save never
// leaves a second spelling of the same option beside the first.
export function spelling(settings, knob) {
  for (const a of knob.aliases || []) if ((settings || {})[a] != null) return a;
  return knob.key;
}

export function valueOf(settings, knob) {
  if (!knob) return "";
  for (const a of [knob.key, ...(knob.aliases || [])]) {
    const v = (settings || {})[a];
    if (v != null) return String(v);
  }
  return "";
}

const num = v => { const n = Number(v); return Number.isFinite(n) && n > 0 ? Math.floor(n) : null; };

export const isMoe = meta => (num((meta || {}).expert_count) || 0) > 0;

// Slider ends from the GGUF header; null where the header doesn't say.
export function limits(meta) {
  const m = meta || {}, layers = num(m.block_count);
  return {ctx: num(m.context_length), moe: layers, ngl: layers ? layers + 1 : null};
}

// Manual once layers are pinned by hand or fit is turned off; Auto leaves both to fit.
export function placementMode(settings, find) {
  const P = QUICK.load.placement;
  const fit = valueOf(settings, find(P.fit));
  const pinned = valueOf(settings, find(P.moe)) !== "" || valueOf(settings, find(P.dense)) !== "";
  return pinned || fit === "off" ? "manual" : "auto";
}

export const contextMode = (settings, find) =>
  valueOf(settings, find(QUICK.load.context)) !== "" ? "set" : "auto";

// The schema keys the strips own; the knob grid leaves them out.
export function ownedKeys(schema) {
  const find = knobFinder(schema), out = new Set();
  const L = QUICK.load, T = QUICK.thinking, S = QUICK.sampling;
  const names = [L.context, L.placement.fit, L.placement.moe, L.placement.dense,
    ...L.fields.map(f => f.key), T.mode.key, T.budget.key, ...T.fields.map(f => f.key),
    ...S.shown.map(f => f.key), ...S.more.map(f => f.key)];
  for (const n of names) { const k = find(n); if (k) out.add(k.key); }
  return out;
}

// Keys the Thinking strip writes on its own (they never pass through the form).
export const instantKeys = () =>
  [QUICK.thinking.mode.key, QUICK.thinking.budget.key, ...QUICK.thinking.fields.map(f => f.key)];

/* ---------- what changed ---------- */

const group = v => (/^\d{5,}$/.test(v) ? v.replace(/\B(?=(\d{3})+(?!\d))/g, " ") : v);
const shown = v => (v === "" ? "inherit" : group(v));

// [{label, from, to}] -> "No changes" / "2 changes: context 32 768 → 48 128, temp inherit → 0.6"
export function describeChanges(list) {
  if (!list.length) return "No changes";
  const parts = list.map(c => `${c.label} ${shown(c.from)} → ${shown(c.to)}`);
  return `${list.length} change${list.length === 1 ? "" : "s"}: ${parts.join(", ")}`;
}

// "fit cut the context" - only when ctx-size is left to fit and the last load
// got less than the model was trained for.
export function fitNote(m, meta, glob) {
  const key = QUICK.load.context;   // ctx-size has no other long spelling
  const sect = (m && m.settings) || {}, rt = m && m.runtime, max = limits(meta).ctx;
  if (sect[key] != null || (glob || {})[key] != null) return "";
  if (!rt || !max || rt.n_ctx_slot >= max) return "";
  return `On the last load fit set the context to ${group(String(rt.n_ctx_slot))} `
    + `(the model takes up to ${group(String(max))}). Pick Set to choose it yourself; fit then moves layers instead.`;
}

/* ---------- widgets ---------- */

const lab = (label, key) => `<span class="qlab">${esc(label)}<span class="qkey">${esc(key)}</span></span>`;
const descOf = (label, knob) => esc(`${knob.key} ${label} ${knob.desc || ""}`.toLowerCase());

// one form field: select for an enum, a text box otherwise (numbers keep the
// author's inputmode, and stay text so values like "all" or "-1" pass)
function field(m, knob, f) {
  const s = m.settings || {}, v = valueOf(s, knob), k = spelling(s, knob);
  const set = v !== "" ? " set" : "";
  const attrs = `data-k="${esc(k)}" data-orig="${esc(v)}" data-label="${esc(f.label.toLowerCase())}"`;
  if (f.ui === "seg" && knob.type === "enum") {
    // "auto" (or, with no such option, a Default button) writes nothing: the
    // build's own default applies, as Auto does on the Thinking switch
    const opts = (knob.options || []).includes("auto") ? knob.options : ["", ...(knob.options || [])];
    const pressed = val => val === "" ? (v === "" || v === "auto") : val === v;
    const btn = (val, text) => `<button type="button" class="qbtn" data-qseg aria-pressed="${pressed(val)}" data-v="${esc(val)}">${esc(text)}</button>`;
    const segs = opts.map(o => btn(o === "auto" ? "" : o,
      o === "" ? "Default" : o.charAt(0).toUpperCase() + o.slice(1))).join("");
    return `<div class="qf${set}" data-desc="${descOf(f.label, knob)}">${lab(f.label, knob.key)}
      <span class="tseg" role="group" aria-label="${esc(f.label)}">${segs}</span>
      <input type="hidden" ${attrs} value="${esc(v)}"></div>`;
  }
  let ctrl;
  if (knob.type === "enum") {
    const opts = [""].concat(knob.options || []);
    if (v && !opts.includes(v)) opts.push(v);
    ctrl = `<select ${attrs}>${opts.map(o => `<option value="${esc(o)}"${o === v ? " selected" : ""}>${o === "" ? "(inherit)" : esc(o)}</option>`).join("")}</select>`;
  } else {
    // the build's default as the placeholder: a strip box is too narrow for "inherit (...)"
    const ph = knob.default || "inherit";
    const im = knob.type === "int" || knob.type === "float" ? ' inputmode="decimal"' : "";
    ctrl = `<input ${attrs} value="${esc(v)}" placeholder="${esc(ph)}"${im}>`;
  }
  const tip = `${knob.desc || ""}${knob.default ? ` (left empty: ${knob.default})` : ""}`.trim();
  return `<label class="qf${set}" data-desc="${descOf(f.label, knob)}" title="${esc(tip)}">${lab(f.label, knob.key)}${ctrl}</label>`;
}

// a slider mirrored onto the text box that carries data-k
const slider = (name, min, max, step, v, label, off = false) => {
  const n = Number(v), val = Number.isFinite(n) && v !== "" ? Math.min(Math.max(n, min), max) : (v === "all" ? max : min);
  return `<input type="range" class="qrange" data-qmirror="${name}" min="${min}" max="${max}" step="${step}" value="${val}" aria-label="${esc(label)}"${off ? " disabled" : ""}>`;
};

const seg2 = (name, mode, a, b, label) =>
  `<span class="tseg" role="group" aria-label="${esc(label)}">`
  + `<button type="button" class="qbtn" data-qmode="${name}" data-v="${a[0]}" aria-pressed="${mode === a[0]}">${a[1]}</button>`
  + `<button type="button" class="qbtn" data-qmode="${name}" data-v="${b[0]}" aria-pressed="${mode === b[0]}">${b[1]}</button></span>`;

function contextRow(m, meta, find, glob) {
  const knob = find(QUICK.load.context);
  if (!knob) return "";
  const s = m.settings || {}, v = valueOf(s, knob), mode = contextMode(s, find);
  const max = limits(meta).ctx, rt = m.runtime;
  const inherited = (glob || {})["ctx-size"];
  const fill = inherited || (rt && rt.n_ctx_slot) || (max ? Math.min(max, CTX_FALLBACK) : CTX_FALLBACK);
  const auto = mode === "auto";
  const ph = auto ? (inherited ? `${group(String(inherited))} from [*]` : "fit decides") : "";
  const per = rt ? `per slot ${group(String(rt.n_ctx_slot))}${rt.n_slots > 1 ? ` · ${rt.n_slots} slots` : ""} · last load` : "per slot: after a load";
  const note = fitNote(m, meta, glob);
  return `<div class="qrow qctx${v !== "" ? " set" : ""}" data-qctx data-desc="${descOf("context", knob)}">
    <div class="qf">${lab("Context", knob.key)}${seg2("ctx", mode, ["auto", "Auto"], ["set", "Set"], "context")}</div>
    ${max ? `<div class="qslide">${slider("ctx", CTX_MIN, max, CTX_STEP, auto ? (rt ? rt.n_ctx_slot : fill) : v, "context size", auto)}
      <span class="qticks"><span>${group(String(CTX_MIN))}</span><span>model max ${group(String(max))}</span></span></div>` : `<div class="qslide"></div>`}
    <label class="qf qnum"><span class="qlab qsub">${esc(per)}</span>
      <input data-k="${esc(spelling(s, knob))}" data-orig="${esc(v)}" data-label="context" data-qlink="ctx" data-fill="${esc(String(fill))}"
        value="${esc(auto ? "" : v)}" placeholder="${esc(ph)}" inputmode="numeric"${auto ? " disabled" : ""}></label>
  </div>${note ? `<div class="slotnote qwarn">${esc(note)}</div>` : ""}`;
}

function placementRow(m, meta, find) {
  const P = QUICK.load.placement, s = m.settings || {};
  const fitK = find(P.fit), moe = isMoe(meta) || valueOf(s, find(P.moe)) !== "";
  const layerK = find(moe ? P.moe : P.dense);
  if (!fitK && !layerK) return "";
  const mode = placementMode(s, find), manual = mode === "manual";
  const lim = limits(meta), max = moe ? lim.moe : lim.ngl;
  const lv = layerK ? valueOf(s, layerK) : "", fitV = fitK ? valueOf(s, fitK) : "";
  const cpuMoe = valueOf(s, find("cpu-moe"));
  // Manual writes fit = off: fit gives up on its own once layers are set by hand,
  // and says so on every load. A section whose layers were pinned by hand while
  // fit stayed on opens with fit = off already pending, counted as a change and
  // explained under the row, so the next Save writes it in plain sight.
  const fitInput = fitK ? `<input type="hidden" data-k="${esc(spelling(s, fitK))}" data-orig="${esc(fitV)}" data-label="placement"
      data-qfit value="${esc(manual ? "off" : (fitV === "off" ? "" : fitV))}" data-qfit-auto="${esc(fitV === "off" ? "" : fitV)}">` : "";
  const what = moe ? "MoE layers on CPU" : "GPU layers";
  const ends = moe ? ["0 · all on GPU", "all layers"] : ["0 · CPU only", "all on GPU"];
  const trade = moe ? "more: less VRAM, slower" : "more: faster, more VRAM";
  const layer = layerK ? `
    <div class="qslide" data-qmanual${manual ? "" : " hidden"}>${lab(what, layerK.key)}${max ? `${slider("layers", 0, max, 1, lv, what)}
      <span class="qticks"><span>${ends[0]}</span><span>${ends[1]}${max ? ` · ${max}` : ""}</span></span>` : ""}</div>
    <label class="qf qnum" data-qmanual${manual ? "" : " hidden"}><span class="qlab qsub">${esc(trade)}</span>
      <input data-k="${esc(spelling(s, layerK))}" data-orig="${esc(lv)}" data-label="${esc(what.toLowerCase())}" data-qlink="layers"
        value="${esc(manual ? lv : "")}" placeholder="${esc(moe ? "0" : "all")}" inputmode="numeric"${manual ? "" : " disabled"}></label>` : "";
  const legacy = manual && fitK && fitV !== "off"
    ? `<div class="slotnote dim">Layers are set by hand, so Save also writes ${esc(fitK.key)} = off: fit gives up on its own here and warns on every load.</div>` : "";
  const cpu = moe && cpuMoe && cpuMoe !== "false"
    ? `<div class="slotnote dim">cpu-moe is set: every expert stays on the CPU, whatever this says.</div>` : "";
  return `<div class="qrow qplace${manual ? " set" : ""}" data-qplace data-desc="${esc(`fit placement ${layerK ? layerK.key : ""} layers`)}">
    <div class="qf">${lab("Placement", fitK ? fitK.key : "fit")}${seg2("place", mode, ["auto", "Auto"], ["manual", "Manual"], "placement")}${fitInput}</div>
    ${layer}
    <span class="qmuted" data-qauto${manual ? " hidden" : ""}>fit decides at load</span>
  </div>${legacy}${cpu}`;
}

/* ---------- the three strips ---------- */

const strip = (id, title, hint, body) => `<section class="qstrip" data-qstrip="${id}">
  <div class="qhead"><span class="qtitle">${esc(title)}</span><span class="qhint">${esc(hint)}</span></div>
  <div class="qbody">${body}</div></section>`;

export function loadStrip(m, meta, schema, glob) {
  const find = knobFinder(schema), L = QUICK.load;
  const fields = L.fields.map(f => { const k = find(f.key); return k ? field(m, k, f) : ""; }).join("");
  const body = contextRow(m, meta, find, glob) + placementRow(m, meta, find)
    + (fields ? `<div class="qfields">${fields}</div>` : "");
  return body.trim() ? strip("load", L.title, L.hint, body) : "";
}

let moreOpen = false;            // "6 more" sampling row, kept across editor rebuilds
export const setMoreOpen = v => { moreOpen = !!v; };

export function samplingStrip(m, schema) {
  const find = knobFinder(schema), S = QUICK.sampling;
  const row = list => list.map(f => { const k = find(f.key); return k ? field(m, k, f) : ""; }).join("");
  const shownF = row(S.shown), moreF = row(S.more);
  if (!shownF && !moreF) return "";
  const s = m.settings || {};
  const moreSet = S.more.some(f => valueOf(s, find(f.key)) !== "");
  const open = moreOpen || moreSet, n = S.more.filter(f => find(f.key)).length;
  const toggle = moreF ? `<button type="button" class="qlink" data-qmore aria-expanded="${open}">${open ? "Fewer" : `${n} more`}</button>` : "";
  return strip("sampling", S.title, S.hint,
    `<div class="qfields">${shownF}${toggle}</div>${moreF ? `<div class="qfields" data-qmore-row${open ? "" : " hidden"}>${moreF}</div>` : ""}`);
}

// Thinking + Vision: switches that write at once (see models.js applyToggle).
export function thinkingStrip(m, meta, schema) {
  if (!m.in_ini) return "";
  const find = knobFinder(schema), T = QUICK.thinking, s = m.settings || {};
  const modeK = find(T.mode.key), budgetK = find(T.budget.key);
  const mode = valueOf(s, modeK), budget = valueOf(s, budgetK);
  const kind = meta && meta.thinking;
  const extras = T.fields.map(f => ({f, k: find(f.key)})).filter(x => x.k);
  const anySet = mode || budget || extras.some(x => valueOf(s, x.k));
  // a template with no reasoning markers has nothing to switch; still shown if
  // something was set, so it can be put back
  const think = !!modeK && !(meta && Object.keys(meta).length && !kind && !anySet);
  const vOn = !!s.mmproj, vision = vOn || !!m.vision_off;
  if (!think && !vision) return "";
  const id = esc(m.id), parts = [];
  if (think) {
    const segs = T.mode.options.map(([v, label]) =>
      `<button type="button" class="qbtn" data-think-mode="${v}" data-think="${id}" aria-pressed="${mode === v}">${label}</button>`).join("");
    parts.push(`<div class="qf">${lab("Mode", modeK.key)}<span class="tseg" role="group" aria-label="thinking">${segs}</span></div>`);
    if (budgetK) {
      const opts = T.budget.options.slice();
      const odd = {"0": "0 · no thinking", "-1": "unlimited (-1)"};   // llama.cpp's two special values
      if (budget && !opts.some(([v]) => v === budget)) opts.push([budget, odd[budget] || `${budget} tokens`]);
      parts.push(`<label class="qf">${lab("Budget", budgetK.key)}<select data-think-budget="${id}" title="most tokens it may think before answering"${mode === "off" ? " disabled" : ""}>`
        + opts.map(([v, label]) => `<option value="${esc(v)}"${v === budget ? " selected" : ""}>${esc(label)}</option>`).join("") + `</select></label>`);
    }
    for (const {f, k} of extras) {
      const v = valueOf(s, k), def = k.default || f.options[0];
      const opts = f.options.filter(o => o !== def);
      if (v && !opts.includes(v) && v !== def) opts.push(v);
      parts.push(`<label class="qf">${lab(f.label, k.key)}<select data-think-key="${esc(spelling(s, k))}" data-think="${id}" title="${esc(k.desc || "")}">`
        + `<option value=""${v === "" || v === def ? " selected" : ""}>${esc(def)}</option>`
        + opts.map(o => `<option value="${esc(o)}"${o === v ? " selected" : ""}>${esc(o)}</option>`).join("") + `</select></label>`);
    }
  }
  if (vision) {
    const seg = (v, label) => `<button type="button" class="qbtn" data-vision="${id}" data-vision-on="${v}" aria-pressed="${vOn === v}">${label}</button>`;
    parts.push(`${think ? `<span class="qsep" aria-hidden="true"></span>` : ""}<div class="qf" title="Loads the model's vision projector (mmproj). Off keeps the file on disk and frees its VRAM.">${lab("Vision", "mmproj")}<span class="tseg" role="group" aria-label="vision">${seg(true, "On")}${seg(false, "Off")}</span></div>`);
  }
  const hint = !think ? "" : kind === "switch" ? "Its chat template has a thinking switch, so Off really turns it off."
    : kind === "tags" ? "Its template shows thinking but has no switch: Off may not stop a model trained to always think." : "";
  const off = vision && !vOn ? "Vision off: text only. The projector file is kept; turn it back on any time." : "";
  const notes = [hint, off].filter(Boolean).map(t => `<div class="slotnote dim">${esc(t)}</div>`).join("");
  return strip("thinking", T.title, T.hint, `<div class="qfields">${parts.join("")}</div>${notes}`);
}

/* ---------- DOM helpers for the handlers in models.js ---------- */

// keep a slider and its text box in step
export function mirror(el) {
  const row = el.closest("[data-qctx],[data-qplace]");
  if (!row) return;
  if (el.dataset.qmirror) {
    const box = row.querySelector(`[data-qlink="${el.dataset.qmirror}"]`);
    if (box && !box.disabled) box.value = el.value;
  } else if (el.dataset.qlink) {
    const r = row.querySelector(`[data-qmirror="${el.dataset.qlink}"]`);
    const n = Number(el.value);
    if (r) r.value = el.value === "all" ? r.max : (Number.isFinite(n) ? String(n) : r.min);
  }
}

// Auto / Set (context), Auto / Manual (placement)
export function setMode(btn) {
  const name = btn.dataset.qmode, v = btn.dataset.v;
  btn.parentElement.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
  if (name === "ctx") {
    const row = btn.closest("[data-qctx]"), box = row.querySelector("[data-qlink=ctx]");
    const auto = v === "auto";
    box.disabled = auto;
    box.value = auto ? "" : (box.dataset.orig || box.dataset.fill || "");
    box.placeholder = auto ? "fit decides" : "";
    const r = row.querySelector("[data-qmirror=ctx]");
    if (r) { r.disabled = auto; if (!auto) r.value = box.value; }
    if (!auto) box.focus();
  } else if (name === "place") {
    const row = btn.closest("[data-qplace]"), manual = v === "manual";
    row.querySelectorAll("[data-qmanual]").forEach(x => { x.hidden = !manual; });
    row.querySelectorAll("[data-qauto]").forEach(x => { x.hidden = manual; });
    const box = row.querySelector("[data-qlink=layers]");
    if (box) { box.disabled = !manual; box.value = manual ? box.dataset.orig : ""; }
    const fit = row.querySelector("[data-qfit]");
    if (fit) fit.value = manual ? "off" : fit.dataset.qfitAuto;
    if (box && manual) { mirror(box); box.focus(); }
  }
}

// a value set from outside (Refine's winner): a strip box left to fit is
// switched to Set / Manual first, so what Save sends matches what is shown
export function adopt(el, value) {
  const row = el.closest("[data-qctx],[data-qplace]");
  if (row && el.disabled) {
    const btn = row.querySelector(`[data-qmode][data-v="${row.matches("[data-qctx]") ? "set" : "manual"}"]`);
    if (btn) setMode(btn);
  }
  el.value = value;
  if (row) mirror(el);
}

// a segmented field (flash-attn) writes its hidden data-k input
export function setSeg(btn) {
  const fld = btn.closest(".qf");
  fld.querySelectorAll("[data-qseg]").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
  const inp = fld.querySelector("input[data-k]");
  if (inp) inp.value = btn.dataset.v;
}

export function toggleMore(btn) {
  const strip = btn.closest("[data-qstrip]"), row = strip && strip.querySelector("[data-qmore-row]");
  if (!row) return;
  row.hidden = !row.hidden;
  setMoreOpen(!row.hidden);
  btn.setAttribute("aria-expanded", String(!row.hidden));
  btn.textContent = row.hidden ? `${row.querySelectorAll("[data-k]").length} more` : "Fewer";
}

// every form field whose value differs from the server's, in DOM order
export function changes(root) {
  const out = [];
  root.querySelectorAll("[data-k][data-orig]").forEach(el => {
    const to = el.value.trim(), from = el.dataset.orig;
    const fld = el.closest(".qf,.fld,.qrow");
    if (fld) fld.classList.toggle("dirty", to !== from);
    if (to !== from) out.push({label: el.dataset.label || el.dataset.k, from, to});
  });
  return out;
}
