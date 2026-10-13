import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const source = (await readFile(new URL("../web/js/quick.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const q = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

// a schema shaped like argspec.build_schema() gives it: key = first long flag
const knob = (key, type, extra = {}) => ({key, type, aliases: [key], default: "", desc: "", options: null, ...extra});
const SCHEMA = {count: 20, groups: [
  {name: "common", knobs: [
    knob("ctx-size", "int", {default: "0"}),
    knob("gpu-layers", "int", {default: "auto", aliases: ["gpu-layers", "n-gpu-layers"]}),
    knob("n-cpu-moe", "int"), knob("cpu-moe", "bool"),
    knob("fit", "enum", {options: ["on", "off"]}),
    knob("cache-type-k", "enum", {options: ["f16", "q8_0"], default: "f16"}),
    knob("cache-type-v", "enum", {options: ["f16", "q8_0"], default: "f16"}),
    knob("parallel", "int", {default: "-1"}),
    knob("flash-attn", "enum", {options: ["on", "off", "auto"]}),
    knob("batch-size", "int", {default: "2048"}), knob("ubatch-size", "int", {default: "512"}),
    knob("threads", "int"),
  ]},
  {name: "server", knobs: [
    knob("reasoning", "enum", {options: ["on", "off", "auto"]}),
    knob("reasoning-budget", "int", {default: "-1"}),
    knob("reasoning-format", "str", {default: "auto"}),
  ]},
  {name: "sampling", knobs: ["temp", "top-p", "top-k", "min-p", "repeat-penalty", "presence-penalty",
    "frequency-penalty", "repeat-last-n", "seed", "dry-multiplier", "xtc-probability"].map(k => knob(k, "float"))},
]};
const MOE = {expert_count: 128, block_count: 30, context_length: 262144, thinking: "switch"};
const DENSE = {block_count: 64, context_length: 131072, thinking: "switch"};
const model = (settings = {}, extra = {}) => ({id: "m", in_ini: true, status: "unloaded", settings, ...extra});
const dataK = html => [...html.matchAll(/data-k="([^"]+)"/g)].map(x => x[1]);

test("knobs are found by any spelling, and written in the one the section already uses", () => {
  const find = q.knobFinder(SCHEMA);
  assert.equal(find("n-gpu-layers").key, "gpu-layers");
  assert.equal(find("kv-tail-tokens"), null);
  assert.equal(q.spelling({"n-gpu-layers": "20"}, find("gpu-layers")), "n-gpu-layers");
  assert.equal(q.spelling({}, find("gpu-layers")), "gpu-layers");
  assert.equal(q.valueOf({"n-gpu-layers": 20}, find("gpu-layers")), "20");
});

test("the grid gives up exactly the keys the strips draw, by schema key", () => {
  const owned = q.ownedKeys(SCHEMA);
  for (const k of ["ctx-size", "gpu-layers", "n-cpu-moe", "fit", "flash-attn", "reasoning", "reasoning-format", "temp", "xtc-probability"])
    assert.ok(owned.has(k), k);
  assert.ok(!owned.has("threads"));
  assert.ok(!owned.has("n-gpu-layers"));             // the alias never names a field
  assert.ok(!owned.has("kv-tail-tokens"));           // not in this build: nothing to own
});

test("slider ends come from the GGUF header, or not at all", () => {
  assert.deepEqual(q.limits(MOE), {ctx: 262144, moe: 30, ngl: 31});
  assert.deepEqual(q.limits({}), {ctx: null, moe: null, ngl: null});
  assert.equal(q.isMoe(MOE), true);
  assert.equal(q.isMoe(DENSE), false);
});

test("placement is Manual once layers are pinned or fit is off; context is Set once ctx-size is", () => {
  const find = q.knobFinder(SCHEMA);
  assert.equal(q.placementMode({}, find), "auto");
  assert.equal(q.placementMode({"n-cpu-moe": "14"}, find), "manual");
  assert.equal(q.placementMode({"n-gpu-layers": "40"}, find), "manual");
  assert.equal(q.placementMode({fit: "off"}, find), "manual");
  assert.equal(q.placementMode({fit: "on"}, find), "auto");
  assert.equal(q.contextMode({}, find), "auto");
  assert.equal(q.contextMode({"ctx-size": "32000"}, find), "set");
});

test("a MoE model gets the expert slider, a dense one the GPU-layers slider", () => {
  const moe = q.loadStrip(model({"n-cpu-moe": "14", fit: "off"}), MOE, SCHEMA, {});
  assert.ok(dataK(moe).includes("n-cpu-moe"));
  assert.ok(!dataK(moe).includes("gpu-layers"));
  assert.match(moe, /data-qmirror="layers" min="0" max="30"[^>]*value="14"/);
  assert.match(moe, /data-qmirror="ctx" min="2048" max="262144"/);
  const dense = q.loadStrip(model({"n-gpu-layers": "40", fit: "off"}), DENSE, SCHEMA, {});
  assert.ok(dataK(dense).includes("n-gpu-layers"), "keeps the section's spelling");
  assert.ok(!dataK(dense).includes("n-cpu-moe"));
  assert.match(dense, /data-qmirror="layers" min="0" max="65"/);
});

test("one form field per key, and the knob the build lacks is simply absent", () => {
  const html = q.loadStrip(model(), MOE, SCHEMA, {}) + q.samplingStrip(model(), SCHEMA);
  const keys = dataK(html);
  assert.equal(new Set(keys).size, keys.length, keys.join(","));
  assert.ok(!keys.includes("kv-tail-tokens"));
  assert.doesNotMatch(html, /Exact tail/);
});

test("Auto placement says fit decides, with no numbers; Manual writes fit = off", () => {
  const auto = q.loadStrip(model(), MOE, SCHEMA, {});
  assert.match(auto, /<span class="qmuted" data-qauto>fit decides at load<\/span>/);
  assert.match(auto, /data-qfit value=""/);
  const manual = q.loadStrip(model({"n-cpu-moe": "14", fit: "off"}), MOE, SCHEMA, {});
  assert.match(manual, /data-orig="off"[^>]*data-qfit value="off"/);
  assert.doesNotMatch(manual, /Save also writes/);
});

test("layers pinned while fit stayed on: fit = off is pending, in plain sight", () => {
  const html = q.loadStrip(model({"n-cpu-moe": "14"}), MOE, SCHEMA, {});
  assert.match(html, /data-orig=""[^>]*data-qfit value="off"/);
  assert.match(html, /Save also writes fit = off/);
});

test("context Auto leaves the box empty for fit; Set shows the value; per slot comes from the last load", () => {
  const auto = q.loadStrip(model({}, {runtime: {n_ctx_slot: 15360, n_slots: 1}}), DENSE, SCHEMA, {});
  assert.match(auto, /data-qlink="ctx" data-fill="15360"\s+value="" placeholder="fit decides" inputmode="numeric" disabled/);
  assert.match(auto, /per slot 15 360 · last load/);
  assert.match(auto, /fit set the context to 15 360 \(the model takes up to 131 072\)/);
  const set = q.loadStrip(model({"ctx-size": "32000"}, {runtime: {n_ctx_slot: 32000, n_slots: 1}}), DENSE, SCHEMA, {});
  assert.match(set, /value="32000" placeholder="" inputmode="numeric">/);
  assert.doesNotMatch(set, /fit set the context/);
  assert.match(q.loadStrip(model(), DENSE, SCHEMA, {}), /per slot: after a load/);
});

test("context inherited from [*] is named, and fit is not blamed for it", () => {
  const html = q.loadStrip(model({}, {runtime: {n_ctx_slot: 8192, n_slots: 1}}), DENSE, SCHEMA, {"ctx-size": "8192"});
  assert.match(html, /placeholder="8192 from \[\*\]"/);
  assert.equal(q.fitNote(model({}, {runtime: {n_ctx_slot: 8192, n_slots: 1}}), DENSE, {"ctx-size": "8192"}), "");
});

test("flash-attn is a segment whose Auto writes nothing", () => {
  const on = q.loadStrip(model({"flash-attn": "on"}), DENSE, SCHEMA, {});
  assert.match(on, /aria-pressed="true" data-v="on">On</);
  const auto = q.loadStrip(model({"flash-attn": "auto"}), DENSE, SCHEMA, {});
  assert.match(auto, /aria-pressed="true" data-v="">Auto</);
});

test("values and ids are escaped", () => {
  const html = q.loadStrip(model({"cache-type-k": '"><img src=x>'}), DENSE, SCHEMA, {})
    + q.thinkingStrip({...model({mmproj: "p"}), id: 'x"><b>'}, DENSE, SCHEMA);
  assert.doesNotMatch(html, /<img src=x>|<b>/);
  assert.match(html, /&lt;img src=x&gt;/);
});

test("Thinking applies at once: no form field, the build's default is the first choice", () => {
  const html = q.thinkingStrip(model({mmproj: "p.gguf"}), MOE, SCHEMA);
  assert.deepEqual(dataK(html), []);
  assert.match(html, /Applies at once\./);
  assert.match(html, /data-think-key="reasoning-format"[^>]*><option value="" selected>auto<\/option><option value="none">/);
  assert.doesNotMatch(html, /reasoning-effort/);        // not in this build
  assert.match(html, /data-vision="m" data-vision-on="true" aria-pressed="true"/);
});

test("no thinking markers and nothing set: only Vision, or nothing", () => {
  const plain = {block_count: 32, context_length: 8192};
  assert.equal(q.thinkingStrip(model(), plain, SCHEMA), "");
  const vis = q.thinkingStrip(model({}, {vision_off: true}), plain, SCHEMA);
  assert.match(vis, /Vision/);
  assert.doesNotMatch(vis, /data-think-mode/);
  assert.match(q.thinkingStrip(model({reasoning: "off"}), plain, SCHEMA), /data-think-mode/);
});

test("sampling shows five and folds the rest, unless one of the rest is set", () => {
  const folded = q.samplingStrip(model(), SCHEMA);
  assert.equal(dataK(folded).length, 11);
  assert.match(folded, /aria-expanded="false">6 more</);
  assert.match(folded, /data-qmore-row hidden/);
  const open = q.samplingStrip(model({seed: "42"}), SCHEMA);
  assert.match(open, /aria-expanded="true">Fewer</);
  assert.doesNotMatch(open, /data-qmore-row hidden/);
});

test("the change summary reads like the editor talks", () => {
  assert.equal(q.describeChanges([]), "No changes");
  assert.equal(q.describeChanges([{label: "context", from: "32768", to: "49152"}]),
    "1 change: context 32 768 → 49 152");
  assert.equal(q.describeChanges([{label: "temp", from: "", to: "0.6"}, {label: "placement", from: "", to: "off"}]),
    "2 changes: temp inherit → 0.6, placement inherit → off");
});

const PRED = {predicted: {total_mib: 14336, weights_mib: 11264, kv_mib: 342, exact_mib: 248,
                          mmproj_mib: 1089, compute_mib: 1393, ram_mib: 7354, ctx: 64000, ctx_slot: 64000},
              saved: {total_mib: 13926}, confident: true, notes: [], fit_on: false,
              gpu: {name: "RTX <5070>", total_mib: 16303, others_mib: 842, budget_mib: 15461},
              fits: true, limits: {ctx: 190464, layers: 11}, measured_mib: null};

test("the VRAM block: total, the delta from unsaved edits, what is left free", () => {
  const html = q.vramBlock(PRED);
  assert.match(html, /≈ 14\.0 GiB <span class="qv-delta">\(\+0\.4 from unsaved changes\)<\/span> · 1\.1 GiB free/);
  assert.match(html, /Weights 11\.0.*KV 0\.3.*Exact tail 0\.2.*Vision 1\.1.*Compute 1\.4.*Other apps 0\.8/s);
  assert.match(html, /RTX &lt;5070&gt;/);
  assert.doesNotMatch(html, /RTX <5070>/);
  assert.match(html, /· predicted/);
  assert.doesNotMatch(html, /qv-cap/);
});

test("over the budget with everything pinned: won't fit, with the line where it ends", () => {
  const html = q.vramBlock({...PRED, predicted: {...PRED.predicted, total_mib: 16384}});
  assert.match(html, /class="qv-sum over"/);
  assert.match(html, /over by 0\.9 GiB - won&#39;t fit/);     // esc() takes the apostrophe too
  assert.match(html, /class="qv-cap"/);
});

test("over the budget with fit still in charge: no alarm, fit cuts back", () => {
  const r = {...PRED, fit_on: true, predicted: {...PRED.predicted, total_mib: 30000},
             notes: ["context left to fit: counted at the model's full trained length; fit shortens it at load if it doesn't fit"]};
  const html = q.vramBlock(r);
  assert.match(html, /class="qv-sum soft"/);
  assert.match(html, /fit cuts back at load/);
  assert.match(html, /estimate uncertain/);
  assert.doesNotMatch(html, /won&#39;t fit|qv-cap/);
});

test("no delta under 51 MiB, a measured load when there is one, and an error stays readable", () => {
  assert.doesNotMatch(q.vramBlock({...PRED, saved: {total_mib: 14300}}), /unsaved changes/);
  assert.match(q.vramBlock({...PRED, measured_mib: 14745}),
    /Saved settings: predicted 13\.6 GiB \/ measured 14\.4 GiB on the last load/);
  assert.match(q.vramBlock(PRED), /Saved settings: predicted 13\.6 GiB \/ not measured yet/);
  assert.match(q.vramBlock({error: "model is <gone>"}), /unavailable: model is &lt;gone&gt;/);
  assert.match(q.vramBlock(null), /unavailable/);
});

test("zones: where a slider stops fitting", () => {
  assert.equal(q.zoneSplit(190464, 2048, 262144, true), 72.4);
  assert.equal(q.zoneSplit(11, 0, 30, false), 36.7);
  assert.equal(q.zoneSplit(null, 0, 30, true), 0);      // nothing fits: all red
  assert.equal(q.zoneSplit(null, 0, 30, false), 100);
  assert.equal(q.zoneSplit(999999, 2048, 262144, true), 100);
});

test("the Load strip leaves room for the estimate and hides empty zones", () => {
  const html = q.loadStrip(model({"n-cpu-moe": "12", fit: "off", "ctx-size": "64000"}), MOE, SCHEMA, {});
  assert.match(html, /data-qvram aria-live="polite"/);
  assert.equal((html.match(/data-qzone="(ctx|layers)" aria-hidden="true" hidden/g) || []).length, 2);
});
