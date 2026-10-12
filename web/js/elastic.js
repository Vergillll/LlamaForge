// Elastic VRAM: what the dashboard chip and the Setup card show. Pure: reads
// /api/state's `elastic` entry, returns markup with every value through esc().
import { esc } from "./core.js";

const gb = mib => (mib / 1024).toFixed(1);

function resized(st, id) {
  const e = st && st.elastic;
  return (e && (e.models || []).find(m => m.id === id && m.at > 0)) || null;
}

const swapping = (st, id) => Boolean(st && st.elastic && st.elastic.swapping === id);

// "40/64 on GPU", or "48/48 on GPU, experts of 20 on CPU" for an expert step:
// the label is "<gpu>/<layers>" plus what else moved.
function where(m, suffix) {
  const frac = `${m.gpu_layers}/${m.layers}`;
  const rest = typeof m.label === "string" && m.label.startsWith(frac) ? m.label.slice(frac.length) : "";
  return `${frac}${suffix}${rest}`;
}

/** The Setup card's "now" line. `grow`: the elastic_grow setting (default on);
 *  with it off a model only grows back when the user clicks its chip. */
export function statusText(e, {grow = true} = {}) {
  if (!e) return "off";
  if (!e.enabled) {
    const w = e.waiting || [];
    if (!w.length) return "off";
    return `off · ${w.join(", ")} ` + (w.length === 1
      ? "goes back to full size once it fits" : "go back to full size once they fit");
  }
  if (e.standby) return "standing by while Game mode is active";
  if (e.swapping) return `resizing ${e.swapping}...`;
  const s = (e.models || []).filter(m => m.at > 0);
  if (!s.length) return "on · no model resized";
  const others = Object.values(e.others_mib || {}).reduce((a, b) => a + b, 0);
  return s.map(m => `${m.id}: ${where(m, " layers on GPU")}`).join(" · ")
    + ` (other apps use ${gb(others)} GB). `
    + (grow ? "Slower until it grows back."
            : "Slower until you grow it back by clicking its dashboard chip (automatic grow-back is off).");
}

/** Whether the "now" line deserves the amber warn colour. */
export function statusWarn(e) {
  return Boolean(e && (e.swapping || (e.models || []).some(m => m.at > 0)));
}

/** Why Elastic can't do anything here, or "" when it can (or we can't tell yet). */
export function gpuNote(st) {
  if (!st) return "";
  if (st.platform === "macos")
    return "Needs an NVIDIA GPU. On a Mac the GPU shares system memory, so there is nothing to move layers to; this setting does nothing here.";
  const g = st.gpus;
  if (Array.isArray(g) && !g.some(x => x && "index" in x && "total" in x))
    return "Needs an NVIDIA GPU. nvidia-smi didn't report one on this machine, so this setting does nothing here.";
  return "";
}

/** The dashboard chip; clicking it grows the model back now. */
export function chip(st, m) {
  if (swapping(st, m.id))
    return `<button type="button" class="tag elastic" disabled title="Elastic VRAM is reloading this model at a new size" aria-label="Resizing">resizing…</button>`;
  const r = resized(st, m.id);
  if (!r) return "";
  const tip = "Elastic VRAM moved layers to the CPU while other apps needed the GPU, so it's slower. Click to grow it back now, if it fits.";
  return `<button type="button" class="tag elastic" data-elastic-grow="${esc(m.id)}" title="${esc(tip)}" aria-label="Grow back now">Elastic · ${esc(where(r, " on GPU"))}</button>`;
}

/** Everything chip() reads, for the row's change signature. */
export function chipSig(st, m) {
  if (swapping(st, m.id)) return ["resizing"];
  const r = resized(st, m.id);
  return r ? [r.gpu_layers, r.layers, r.label] : "";
}

export function eventsMarkup(e) {
  const ev = (e && e.events) || [];
  if (!ev.length) return "";
  const at = t => new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  const kept = k => k === true ? " · conversation kept" : k === false ? " · conversation not kept" : "";
  return `<div class="el-events" style="max-height:180px;overflow:auto;margin:4px 0 10px;font-family:var(--mono);font-size:12px">${ev.map(x =>
    `<div class="${x.ok ? "" : "bad"}" style="padding:2px 0">${esc(at(x.t))} ${esc(x.model)} ${esc(x.from)} &rarr; ${esc(x.to)} · ${esc(x.reason)} · ${esc(x.secs)} s${kept(x.kept)}${x.note ? " · " + esc(x.note) : ""}</div>`).join("")}</div>`;
}
