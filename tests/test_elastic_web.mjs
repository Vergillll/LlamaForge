import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const core = await readFile(new URL("../web/js/core.js", import.meta.url), "utf8");
const coreUrl = `data:text/javascript;base64,${Buffer.from(core).toString("base64")}`;
const source = (await readFile(new URL("../web/js/elastic.js", import.meta.url), "utf8"))
  .replace('"./core.js"', `"${coreUrl}"`);
const el = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const shrunk = {enabled: true, standby: false, swapping: null, cooldown_s: 120, others_mib: {"0": 6246},
  waiting: [],
  models: [{id: "qwen<27b>", at: 1, gpu_layers: 40, layers: 64, label: "40/64", steps: 4, full: "64/64"},
           {id: "small", at: 0, gpu_layers: 32, layers: 32, label: "32/32", steps: 2, full: "32/32"}],
  events: [{t: 1760000000, model: "qwen<27b>", from: "64/64", to: "40/64", reason: "other apps use 6.1 GB",
            secs: 19.6, ok: true, kept: true, note: ""}]};

test("status text: off, standing by, resizing, and what is shrunk", () => {
  assert.equal(el.statusText(null), "off");
  assert.equal(el.statusText({enabled: false}), "off");
  assert.match(el.statusText({enabled: true, standby: true}), /Game mode/);
  assert.match(el.statusText({enabled: true, swapping: "x", models: []}), /resizing x/);
  assert.equal(el.statusText({enabled: true, models: [], others_mib: {}}), "on · no model resized");
  assert.equal(el.statusText(shrunk),
    "qwen<27b>: 40/64 layers on GPU (other apps use 6.1 GB). Slower until it grows back.");
});

test("status text: turned off while shrunk waits for room, it doesn't promise an instant restore", () => {
  const off = {...shrunk, enabled: false, waiting: ["qwen<27b>"]};
  const t = el.statusText(off);
  assert.match(t, /^off/);
  assert.match(t, /qwen<27b>/);
  assert.match(t, /once it fits/);
  assert.doesNotMatch(t, /now|instant/i);
});

test("an expert-offload step says the experts moved, not just the layer count", () => {
  const moe = {...shrunk, models: [{id: "moe", at: 1, gpu_layers: 48, layers: 48,
    label: "48/48, experts of 20 on CPU", steps: 5, full: "48/48"}]};
  assert.match(el.statusText(moe), /moe: 48\/48 layers on GPU, experts of 20 on CPU/);
  assert.match(el.chip({elastic: moe}, {id: "moe"}), /Elastic · 48\/48 on GPU, experts of 20 on CPU/);
  assert.deepEqual(el.chipSig({elastic: moe}, {id: "moe"}), [48, 48, "48/48, experts of 20 on CPU"]);
});

test("the chip shows only for a resized model, escaped", () => {
  assert.equal(el.chip({elastic: shrunk}, {id: "small"}), "");
  assert.equal(el.chip(null, {id: "small"}), "");
  const html = el.chip({elastic: shrunk}, {id: "qwen<27b>"});
  assert.match(html, /Elastic · 40\/64 on GPU/);
  assert.match(html, /data-elastic-grow="qwen&lt;27b&gt;"/);
  assert.doesNotMatch(html, /qwen<27b>/);
  assert.deepEqual(el.chipSig({elastic: shrunk}, {id: "qwen<27b>"}), [40, 64, "40/64"]);
  assert.equal(el.chipSig({elastic: shrunk}, {id: "small"}), "");
});

test("the chip still shows for a model waiting to grow back after Elastic was turned off", () => {
  const off = {...shrunk, enabled: false, waiting: ["qwen<27b>"]};
  assert.match(el.chip({elastic: off}, {id: "qwen<27b>"}), /Elastic · 40\/64 on GPU/);
});

test("the event log is escaped and says whether the conversation was kept", () => {
  const html = el.eventsMarkup(shrunk);
  assert.match(html, /qwen&lt;27b&gt;/);
  assert.match(html, /64\/64 &rarr; 40\/64/);
  assert.match(html, /conversation kept/);
  assert.equal(el.eventsMarkup({events: []}), "");
  assert.equal(el.eventsMarkup(null), "");
});

test("a failed attempt that didn't move shows its note, escaped", () => {
  const html = el.eventsMarkup({events: [{t: 1760000000, model: "m", from: "40/64", to: "40/64",
    reason: "you asked", secs: 0, ok: false, kept: null, note: "other apps <still> use the VRAM it needs"}]});
  assert.match(html, /class="bad"/);
  assert.match(html, /other apps &lt;still&gt; use/);
  assert.doesNotMatch(html, /conversation/);
});

test("needs an NVIDIA GPU: an honest note on a Mac or without nvidia-smi", () => {
  assert.match(el.gpuNote({platform: "macos", gpus: []}), /NVIDIA/);
  assert.match(el.gpuNote({platform: "linux", gpus: [{error: "nvidia-smi not found"}]}), /NVIDIA/);
  assert.match(el.gpuNote({platform: "windows", gpus: []}), /NVIDIA/);
  assert.equal(el.gpuNote({platform: "windows", gpus: [{index: 0, used: 1, total: 16000}]}), "");
  assert.equal(el.gpuNote(null), "");            // state not loaded yet: say nothing
});

test("plural: one waiting model fits, several fit", () => {
  assert.match(el.statusText({...shrunk, enabled: false, waiting: ["a"]}), /a goes back to full size once it fits$/);
  assert.match(el.statusText({...shrunk, enabled: false, waiting: ["a", "b"]}), /a, b go back to full size once they fit$/);
});

test("with auto-grow off the status points at the dashboard chip, not an automatic grow", () => {
  const t = el.statusText(shrunk, {grow: false});
  assert.match(t, /dashboard chip/);
  assert.doesNotMatch(t, /until it grows back/);
  assert.match(el.statusText(shrunk, {grow: true}), /until it grows back/);
  assert.match(el.statusText(shrunk), /until it grows back/);
});

test("statusWarn: something is resized or a swap is running", () => {
  assert.equal(el.statusWarn(null), false);
  assert.equal(el.statusWarn({models: []}), false);
  assert.equal(el.statusWarn(shrunk), true);
  assert.equal(el.statusWarn({swapping: "x", models: []}), true);
});

test("a model mid-swap gets a disabled 'resizing' chip, and the signature changes", () => {
  const busy = {...shrunk, swapping: "qwen<27b>"};
  const html = el.chip({elastic: busy}, {id: "qwen<27b>"});
  assert.match(html, /disabled/);
  assert.match(html, /resizing…/);
  assert.doesNotMatch(html, /qwen<27b>/);
  assert.notDeepEqual(el.chipSig({elastic: busy}, {id: "qwen<27b>"}),
                      el.chipSig({elastic: shrunk}, {id: "qwen<27b>"}));
  // a full-size model starting to shrink shows it too
  const first = {...shrunk, swapping: "small"};
  assert.match(el.chip({elastic: first}, {id: "small"}), /resizing…/);
  assert.notEqual(el.chipSig({elastic: first}, {id: "small"}), "");
});

test("a label without the gpu/layers prefix falls back to the layer count", () => {
  const odd = {...shrunk, models: [{id: "m", at: 1, gpu_layers: 30, layers: 60, label: "odd<label>", steps: 3, full: "60/60"}]};
  assert.match(el.chip({elastic: odd}, {id: "m"}), /Elastic · 30\/60 on GPU</);
  assert.doesNotMatch(el.chip({elastic: odd}, {id: "m"}), /odd/);
  assert.match(el.statusText(odd), /^m: 30\/60 layers on GPU \(/);
  const none = {...odd, models: [{...odd.models[0], label: undefined}]};
  assert.match(el.chip({elastic: none}, {id: "m"}), /Elastic · 30\/60 on GPU</);
});
