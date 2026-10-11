// Embers tab: the user's embers, each one's morning brief, its wiki, Ask,
// recent runs and settings, plus "new ember" from a bundled template.
//
// Brief, page and index HTML comes from embers/view.py on the server, which
// escapes every piece of model or source text before it builds markup (the
// same contract as the Help tab's docs renderer). Everything this file
// interpolates itself goes through esc().
import { $, $$, api, esc, setHTML, toast, askYes } from "./core.js";
import { models, config as cfgOf } from "./state.js";
import { showModal } from "./models.js";
import { forgeReset, forgeSend, initForge, renderForge } from "./forge.js";

const SLOT_HINT = {
  folder: "A folder of notes, e.g. D:\\Notes or ~/notes (Markdown or text, read only)",
  ics: "A .ics calendar file, or an https:// link to one",
  git: "A local git repository folder (reads the commit log)",
  rss: "An RSS or Atom feed address",
  llamacpp: "Read from LlamaForge itself: nothing to set",
  machine: "Read from LlamaForge itself: nothing to set",
};
const AUTO = ["llamacpp", "machine"];      // templates.AUTO_SLOTS
const JOB_LABEL = { ingest: "Read sources", brief: "Write brief", lint: "Check wiki" };
const SUBS = [["brief", "Brief"], ["wiki", "Wiki"], ["ask", "Ask"], ["activity", "Activity"], ["settings", "Settings"]];

const E = {
  data: null,          // last /api/embers payload
  sel: null,           // selected ember id
  sub: "brief",        // shown section
  sig: "",             // the selected ember's run signature, to refresh on change
  page: null,          // wiki page shown (null = index)
  forge: false,        // the Forge conversation is showing instead of an ember
};
try { E.sel = localStorage.getItem("lf_ember"); } catch (e) {}

const card = id => ((E.data && E.data.embers) || []).find(c => c.id === id);
// Changes when a run finishes or a brief lands: the cue to reload what's on screen.
const sigOf = c => c ? JSON.stringify([c.last, c.brief && c.brief.date, c.counts]) : "";

/* ---------- formatting ---------- */
function parseTs(s) {
  if (!s) return null;
  const d = new Date(String(s).replace(" ", "T"));
  return isNaN(d) ? null : d;
}
function ago(s) {
  const d = parseTs(s);
  if (!d) return "";
  const secs = (Date.now() - d) / 1000;
  return secs < 60 ? "just now" : secs < 3600 ? Math.floor(secs / 60) + "m ago"
       : secs < 86400 ? Math.floor(secs / 3600) + "h ago" : Math.floor(secs / 86400) + "d ago";
}
function when(s) {
  const d = parseTs(s);
  if (!d) return "";
  const t = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  const day = new Date(d); day.setHours(0, 0, 0, 0);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((day - today) / 86400000);
  return diff === 0 ? `today ${t}` : diff === 1 ? `tomorrow ${t}`
       : `${d.toLocaleDateString("en-GB", { weekday: "short" })} ${t}`;
}
function dayName(date) {
  const d = parseTs(date + "T12:00:00");
  return d ? d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }) : date;
}
function took(a, b) {
  const s = (parseTs(b) - parseTs(a)) / 1000;
  return isFinite(s) && s >= 0 ? (s < 60 ? Math.round(s) + "s" : Math.round(s / 60) + "m") : "";
}

/* ---------- one flame colour per ember ----------
   Derived from the id, so an ember keeps its colour across reloads and
   machines. The hues are a curated set that stays legible on both Hearth
   surfaces; colour is identity only, never status (the LED carries that).
   Only the Hearth skin draws the flame - Classic hides .ef-flame. */
const HUES = [24, 6, 42, 172, 196, 262, 328, 138, 52, 292, 350, 214];
function emberHue(id) {
  let h = 0;
  for (const ch of String(id)) h = (h * 31 + ch.codePointAt(0)) >>> 0;
  return HUES[h % HUES.length];
}
const FLAME = `<svg class="ef-flame" viewBox="0 0 24 24" aria-hidden="true">`
  + `<path class="o" d="M12 2.5c.8 3.6 5.5 5.3 5.5 10.8a5.5 5.5 0 0 1-11 0c0-3.2 2.1-4.6 2.3-7.4 1.1 1.2 2 2.2 3.2 2.4.3-1.9.3-3.8 0-5.8z"/>`
  + `<path class="i" d="M12 11.5c.4 1.7 2.6 2.5 2.6 5a2.6 2.6 0 0 1-5.2 0c0-1.4.9-2.1 1.1-3.3.5.5.9.9 1.5 1 .1-.9.1-1.7 0-2.7z"/></svg>`;

/* ---------- state of one ember, as one LED + one line ---------- */
function state(c) {
  if (c.error) return ["err", "can't be read: " + c.error];
  if (c.running) return ["run", `${JOB_LABEL[c.running] || c.running}…`];
  const waiting = Object.entries(c.waiting || {});
  if (waiting.length) return ["wait", `waiting to ${(JOB_LABEL[waiting[0][0]] || waiting[0][0]).toLowerCase()}: ${waiting[0][1]}`];
  if ((c.queued || []).length) return ["wait", "queued: " + c.queued.map(j => JOB_LABEL[j] || j).join(", ")];
  if (!c.enabled) return ["off", "paused"];
  const last = Object.values(c.last || {}).sort((a, b) => String(b.started).localeCompare(String(a.started)))[0];
  if (last && last.status === "error") return ["err", "last run failed"];
  if (last && last.status === "partial") return ["warn", "last run partly failed"];
  return ["ok", c.next && c.next.brief ? "next brief " + when(c.next.brief) : "idle"];
}

/* ---------- loading ---------- */
export async function loadEmbers() {
  const v = $("#view-embers");
  if (!v.dataset.ready) {
    v.dataset.ready = "1";
    setHTML(v, `<div class="ef-wrap">
        <aside class="ef-side">
          <div class="tbl-head"><h2>Your embers</h2><span class="count" id="ef-count"></span></div>
          <div class="ef-row ef-forgerow" id="ef-forgerow" data-forge-open role="button" tabindex="0"
            title="Describe what you want; Forge builds the ember">
            <span class="ef-led forge"></span>
            <span class="ef-rowtext"><span class="ef-name">Forge</span><span class="ef-sub">builds embers by talking with you</span></span>
          </div>
          <div id="ef-list" class="ef-list"></div>
          <button id="ef-new" class="primary ef-newbtn">+ New ember</button>
          <div id="ef-sched" class="note"></div>
          <div id="ef-folder" class="ef-folder"></div>
        </aside>
        <section id="ef-main" class="ef-main"></section>
      </div>`);
    v.addEventListener("click", onClick);
    v.addEventListener("submit", onSubmit);
    v.addEventListener("keydown", e => {
      if (e.key === "Enter" && e.target.matches("[data-ember], [data-forge-open]")) e.target.click();
    });
    $("#ef-new").onclick = openNew;
    initForge(async id => {
      await refresh(true);                       // the new ember shows in the list; Forge stays open
    });
  }
  await refresh(false);
}

/** Poll while the tab is showing: list + status strip; a section reloads only when a run finished. */
export async function poll() { await refresh(true); }

async function refresh(quiet) {
  const d = await api("/api/embers").catch(() => null);
  if (!d || d.error) {
    if (!quiet) setHTML($("#ef-main"), `<div class="card"><div class="msg err">${esc(d ? d.error : "The panel didn't answer.")}</div></div>`);
    return;
  }
  E.data = d;
  if (!card(E.sel)) E.sel = d.embers.length ? d.embers[0].id : null;
  renderList();
  if (E.forge) {                                 // the conversation owns the main area
    if (!$("#ef-forge")) renderForge($("#ef-main"));
    return;
  }
  const c = card(E.sel);
  const sig = sigOf(c);
  if (!quiet || !$("#ef-detail") || $("#ef-detail").dataset.id !== (E.sel || "")) {
    E.sig = sig;
    renderDetail();
  } else {
    renderStatus(c);
    if (sig !== E.sig) {
      E.sig = sig;
      if (["brief", "activity", "wiki"].includes(E.sub)) showSub(E.sub);
    }
  }
}

function renderList() {
  const d = E.data, list = d.embers;
  $("#ef-count").textContent = list.length ? String(list.length) : "";
  setHTML($("#ef-list"), list.map(c => {
    const [led, line] = state(c);
    const sub = c.error ? line
      : c.brief ? `${dayName(c.brief.date)} · ${c.counts.open} open` : "no brief yet";
    return `<div class="ef-row${c.id === E.sel && !E.forge ? " sel" : ""}" data-ember="${esc(c.id)}" role="button" tabindex="0"
        title="${esc(line)}" style="--ember-h:${emberHue(c.id)}">
      ${FLAME}<span class="ef-led ${led}"></span>
      <span class="ef-rowtext"><span class="ef-name">${esc(c.name)}</span><span class="ef-sub">${esc(sub)}</span></span>
    </div>`;
  }).join(""));
  $("#ef-forgerow").classList.toggle("sel", E.forge);
  const sched = !d.scheduler ? "The embers scheduler isn't running in this panel, so nothing runs. It starts when you launch LlamaForge normally."
    : !d.scheduled ? "Automatic runs are off (config embers_scheduler). Run now still works."
    : "Runs on schedule while LlamaForge is open, when the GPU is free.";
  setHTML($("#ef-sched"), esc(sched) + (d.last_error ? `<div class="msg err">${esc(d.last_error)}</div>` : ""));
  const f = $("#ef-folder");                    // redraw only on change: never wipe a half-typed path
  if (!$("#ef-folderform") && f.dataset.path !== d.embers_dir) renderFolder();
}

/* ---------- where all embers live ---------- */
function renderFolder() {
  $("#ef-folder").dataset.path = E.data.embers_dir;
  setHTML($("#ef-folder"), `<div class="ef-flabel">Folder <button type="button" class="ef-flink" data-folder="edit">Change</button></div>
    <code class="ef-fpath" title="${esc(E.data.embers_dir)}">${esc(E.data.embers_dir)}</code>`);
}

function editFolder() {
  delete $("#ef-folder").dataset.path;          // Cancel / save redraws it
  setHTML($("#ef-folder"), `<form id="ef-folderform">
      <div class="ef-flabel">Folder</div>
      <input name="path" value="${esc(E.data.embers_dir)}" spellcheck="false" autocomplete="off" aria-label="Embers folder">
      <div class="note">Each ember is a subfolder of plain Markdown, so a folder inside an Obsidian vault works.
      Nothing is moved: embers in the old folder stay there until you move them.</div>
      <div class="ef-factions">
        <button class="primary" type="submit">Use folder</button>
        <button type="button" data-folder="default" title="Back to the embers folder next to LlamaForge">Default</button>
        <button type="button" data-folder="cancel">Cancel</button>
      </div>
      <div id="ef-fmsg" class="msg"></div>
    </form>`);
  const input = $("#ef-folderform input");
  input.focus(); input.select();
}

async function saveFolder(path) {
  const form = $("#ef-folderform");
  if (form) form.querySelectorAll("button").forEach(b => b.disabled = true);
  const d = await api("/api/embers/folder", { path }).catch(() => null);
  if (!d || d.error) {
    if (form) form.querySelectorAll("button").forEach(b => b.disabled = false);
    setHTML($("#ef-fmsg"), `<span class="msg err">${esc(d ? d.error : "The panel didn't answer.")}</span>`);
    return;
  }
  const n = k => k === 1 ? "1 ember" : `${k} embers`;
  toast(`Using ${d.embers_dir}. ${d.found ? `Found ${n(d.found)} there.` : "No embers there yet."}`
    + (d.left ? ` ${n(d.left)} stayed in the old folder.` : ""), "ok");
  E.sel = null;
  $("#ef-folder").textContent = "";             // no form, no path: refresh redraws it
  await refresh(false);
}

/* ---------- empty state ---------- */
function renderEmpty() {
  setHTML($("#ef-main"), `<div class="card ef-empty">
      <h3>An ember keeps a wiki for you</h3>
      <p>Point it at things you already have (a notes folder, a calendar, a repo, a feed).
      Overnight it reads them with the model you run here and keeps a small Markdown wiki
      of open loops, people and dates. Each morning it writes a short brief.</p>
      <p>Every item in the wiki cites a quote from a source snapshot, and the panel checks that the quote
      is really there. Nothing leaves this machine unless you turn on push.</p>
      <p class="note">Wiki folder: <code>${esc(E.data.embers_dir)}</code> · plain Markdown, opens in Obsidian.</p>
      <div class="actions"><button class="primary" type="button" data-forge-open>Talk to Forge</button>
        <button type="button" id="ef-new2">Start from a template</button></div>
      <p class="note">Forge asks what you want kept track of and builds the ember for you, using a model you
      already loaded. Templates are ready-made embers you point at your sources yourself.</p>
    </div>`);
  $("#ef-new2").onclick = openNew;
}

/* ---------- one ember ---------- */
function renderDetail() {
  const c = card(E.sel);
  if (!c) return renderEmpty();
  setHTML($("#ef-main"), `<div id="ef-detail" data-id="${esc(c.id)}" style="--ember-h:${emberHue(c.id)}">
      <div class="ef-head">
        <div class="ef-headtext">
          <h2 class="ef-title">${FLAME}<span class="ef-tname">${esc(c.name)}</span>${c.origin === "forge" ? ` <span class="ef-tag" title="Built by Forge">forge</span>` : ""}</h2>
          <div class="ef-mission">${esc(c.mission || "")}</div>
        </div>
        <div class="qbtns">
          <button class="qbtn load" data-run="run" title="Read the sources, then write the brief">Run now</button>
          <button class="qbtn" data-run="brief" title="Write the brief from the wiki as it is">Brief only</button>
          <button class="qbtn" data-run="lint" title="Look for stale items, missing evidence and contradictions">Check wiki</button>
        </div>
      </div>
      <div id="ef-status" class="ef-status"></div>
      <div class="ef-tabs" role="tablist">${SUBS.map(([k, label]) =>
        `<button type="button" class="ef-tab${E.sub === k ? " on" : ""}" data-sub="${k}" role="tab">${label}</button>`).join("")}</div>
      <div id="ef-sub" class="ef-subview"></div>
    </div>`);
  renderStatus(c);
  showSub(E.sub);
}

function renderStatus(c) {
  const el = $("#ef-status");
  if (!el || !c) return;
  const [led, line] = state(c);
  const parts = [`<span class="ef-led ${led}"></span><span>${esc(line)}</span>`];
  if ((c.queued || []).length) parts.push(`<button class="qbtn stop" data-cancel>Cancel queued</button>`);
  const n = c.counts || {};
  if (!c.error) parts.push(`<span class="ef-counts">${esc(n.pages || 0)} pages · ${esc(n.open || 0)} open items · ${esc(n.verified || 0)} with a checked quote</span>`);
  setHTML(el, parts.join(""));
}

async function showSub(sub) {
  E.sub = sub;
  $$(".ef-tab").forEach(b => b.classList.toggle("on", b.dataset.sub === sub));
  const el = $("#ef-sub"), c = card(E.sel);
  if (!el || !c) return;
  if (c.error) {
    setHTML(el, `<div class="card"><div class="msg err">${esc(c.error)}</div>
      <div class="note">Fix or remove <code>${esc(c.root)}/ember.json</code>.</div></div>`);
    return;
  }
  const fn = { brief: showBrief, wiki: showWiki, ask: showAsk, activity: showActivity, settings: showSettings }[sub];
  await fn(el, c);
}

function failed(el, d) {
  if (d && !d.error) return false;
  setHTML(el, `<div class="msg err">${esc(d ? d.error : "The panel didn't answer.")}</div>`);
  return true;
}

/* ---------- brief ---------- */
async function showBrief(el, c, date) {
  const d = await api(`/api/embers/brief?id=${encodeURIComponent(c.id)}` + (date ? `&date=${encodeURIComponent(date)}` : ""));
  if (failed(el, d) || E.sel !== c.id || E.sub !== "brief") return;
  if (!d.date) {
    const busy = c.running || (c.queued || []).length;
    setHTML(el, `<div class="ef-blank">${busy
      ? "The first run is on its way. Reading sources can take a few minutes on a big folder; the brief appears here when it's done."
      : `No brief yet. <button class="qbtn load" data-run="run">Run now</button> to read the sources and write one.`}</div>`);
    return;
  }
  setHTML(el, `<div class="ef-briefbar">
      <select id="ef-date" aria-label="Brief date">${d.dates.map(x =>
        `<option value="${esc(x)}"${x === d.date ? " selected" : ""}>${esc(dayName(x))}</option>`).join("")}</select>
      <span class="note">Numbers are sources: click one to read the quote in its snapshot.</span>
    </div>
    <article class="ef-doc">${d.html}</article>`);
  $("#ef-date").onchange = e => showBrief(el, c, e.target.value);
}

/* ---------- wiki ---------- */
async function showWiki(el, c) {
  const d = await api(`/api/embers/pages?id=${encodeURIComponent(c.id)}`);
  if (failed(el, d) || E.sel !== c.id || E.sub !== "wiki") return;
  const kinds = {};
  d.pages.forEach(p => (kinds[p.page.split("/")[0]] ||= []).push(p));
  setHTML(el, `<div class="ef-wiki">
      <nav class="ef-pages">
        <a href="#" class="ef-pg${E.page ? "" : " on"}" data-pg="">Index</a>
        ${Object.keys(kinds).sort().map(k => `<div class="ef-kind">${esc(k)}</div>` + kinds[k].map(p =>
          `<a href="#" class="ef-pg${E.page === p.page ? " on" : ""}" data-pg="${esc(p.page)}">${esc(p.title || p.page)}${
            p.open ? ` <span class="ef-open">${esc(p.open)}</span>` : ""}</a>`).join("")).join("")}
        ${d.pages.length ? "" : `<div class="note">No pages yet.</div>`}
        <div class="note">${esc(d.raws.length)} source snapshots</div>
      </nav>
      <article class="ef-doc" id="ef-page">${E.page ? "" : d.index_html}</article>
    </div>`);
  if (E.page) openPage(E.page);
}

async function openPage(page, item) {
  E.page = page || null;
  $$(".ef-pg").forEach(a => a.classList.toggle("on", a.dataset.pg === (page || "")));
  const el = $("#ef-page");
  if (!el) return;
  if (!page) return showWiki($("#ef-sub"), card(E.sel));
  const d = await api(`/api/embers/page?id=${encodeURIComponent(E.sel)}&page=${encodeURIComponent(page)}`);
  if (failed(el, d)) return;
  setHTML(el, d.html);
  const li = item && $(`li[data-item="${CSS.escape(item)}"]`, el);
  if (li) { li.classList.add("ef-hit"); li.scrollIntoView({ block: "center" }); }
  else el.scrollIntoView({ block: "nearest" });
}

/* ---------- ask ---------- */
function showAsk(el, c) {
  setHTML(el, `<form class="ef-askf" id="ef-askf">
      <input id="ef-q" maxlength="500" autocomplete="off" placeholder="e.g. What am I waiting on from Sam?">
      <button class="primary" type="submit">Ask</button>
    </form>
    <div class="note">Answers come only from items in this wiki whose quote checks out, using a model that's
    already loaded (${c.model ? `${esc(c.model)} if it's up, otherwise the main one` : "the main one"}).
    Ask never loads a model.</div>
    <div id="ef-ans"></div>`);
  $("#ef-q").focus();
}

async function ask(q) {
  const out = $("#ef-ans"), btn = $("#ef-askf button");
  btn.disabled = true;
  setHTML(out, `<div class="msg work">Thinking with the loaded model…</div>`);
  const d = await api("/api/embers/ask", { id: E.sel, question: q }).catch(() => null);
  btn.disabled = false;
  if (failed(out, d)) return;
  if (d.status === "empty") {
    setHTML(out, `<div class="ef-blank">Nothing in this wiki can answer that yet: no item with a checked quote
      matched. Run the ember, or point it at more sources.</div>`);
    return;
  }
  setHTML(out, `<div class="ef-answer">
      <p>${esc(d.answer || "(the model gave no answer)")}</p>
      ${d.grounded ? "" : `<div class="ef-warn">Some names, numbers or dates in this answer aren't in the items it
        cites or in your question. Check them against the sources below.</div>`}
      ${d.items.length ? `<ol class="ef-cites">${d.items.map(i => `<li>
          <a href="#" class="ef-link" data-page="${esc(i.page)}" data-item="${esc(i.id)}">${esc(i.page)}</a>
          ${i.status === "open" ? "" : `<span class="ef-tag">done</span>`} ${esc(i.text)}
          <div class="ef-citeq"><button type="button" class="ef-quote" data-raw="${esc(i.raw)}" title="${esc(i.quote)}">source</button>
          <q>${esc(i.quote)}</q></div></li>`).join("")}</ol>`
        : `<div class="note">The model cited no items.</div>`}
      <div class="note">Answered by ${esc(d.model)}.</div>
    </div>`);
}

/* ---------- activity ---------- */
async function showActivity(el, c) {
  const d = await api(`/api/embers/log?id=${encodeURIComponent(c.id)}`);
  if (failed(el, d) || E.sel !== c.id || E.sub !== "activity") return;
  const rows = d.runs.map(r => `<tr class="ef-run-${esc(r.status)}">
      <td>${esc(JOB_LABEL[r.job] || r.job)}</td><td>${esc(r.status)}</td>
      <td title="${esc(r.started)}">${esc(ago(r.started))}</td><td>${esc(took(r.started, r.finished))}</td>
      <td>${r.tokens_in ? esc(`${r.tokens_in} → ${r.tokens_out}`) : ""}</td><td class="ef-err">${esc(r.error)}</td></tr>`).join("");
  setHTML(el, `${rows ? `<table class="ef-runs"><thead><tr><th>Job</th><th>Result</th><th>Started</th><th>Took</th>
      <th>Tokens in → out</th><th>Note</th></tr></thead><tbody>${rows}</tbody></table>` : `<div class="ef-blank">No runs yet.</div>`}
    ${d.log.length ? `<details class="ef-logd"><summary>log.md (last ${esc(d.log.length)} lines)</summary>
      <div class="log">${esc(d.log.join("\n"))}</div></details>` : ""}`);
}

/* ---------- settings ---------- */
function showSettings(el, c) {
  const ids = models().map(m => m.id);
  if (c.model && !ids.includes(c.model)) ids.unshift(c.model);
  const p = c.push || { kind: "", url: "" }, pl = c.push_last;
  setHTML(el, `<form id="ef-set" class="ef-set">
      <div class="card"><h3>Ember</h3>
        <div class="formrow">
          <label class="f grow"><span class="lbl">Name</span><input name="name" maxlength="120" value="${esc(c.name)}"></label>
          <label class="f"><span class="lbl">Model</span><select name="model">
            <option value="">Whichever is loaded, else Setup's auto-load model</option>
            ${ids.map(m => `<option value="${esc(m)}"${m === c.model ? " selected" : ""}>${esc(m)}</option>`).join("")}
          </select></label>
          <label class="f ef-check"><input type="checkbox" name="enabled"${c.enabled ? " checked" : ""}> Runs on schedule</label>
        </div>
        ${c.model_hint ? `<div class="note">Template suggests: ${esc(c.model_hint)}.</div>` : ""}
        ${!c.model && !cfgOf().auto_load_model ? `<div class="slotnote warn">No model picked here and no auto-load
          model in Setup, so this ember only runs while you have a model loaded. Pick one to let it load
          one itself overnight.</div>` : ""}
      </div>
      <div class="card"><h3>Sources</h3>
        ${c.slots.map(s => slotField(s, c.bindings[s.id] || "")).join("")}
        <div class="note">Read only: an ember never writes to your sources.</div>
      </div>
      <div class="card"><h3>Schedule</h3>
        <div class="formrow">${["ingest", "brief", "lint"].map(j => `
          <label class="f"><span class="lbl">${esc(JOB_LABEL[j])}</span>
            <input name="job-${j}" value="${esc(c.schedule[j] || "off")}" placeholder="07:00, sun 03:00 or off" size="11"></label>`).join("")}
        </div>
        <div class="note">Local time. A job waits while you're using the GPU and runs when it's free.</div>
      </div>
      <div class="card"><h3>Push the brief</h3>
        <div class="formrow">
          <label class="f"><span class="lbl">Send to</span><select name="push-kind">
            <option value="">Off</option>
            <option value="ntfy"${p.kind === "ntfy" ? " selected" : ""}>ntfy (phone)</option>
            <option value="webhook"${p.kind === "webhook" ? " selected" : ""}>Webhook (Slack, Discord, …)</option>
          </select></label>
          <label class="f grow"><span class="lbl">Address</span>
            <input name="push-url" maxlength="1000" value="${esc(p.url)}" placeholder="https://ntfy.sh/a-long-random-topic"></label>
          <button type="button" id="ef-pushtest">Send test</button>
        </div>
        <div class="note">After each brief this sends its title, headline and first three bullets as plain text.
          No wiki pages, quotes or source text. An ntfy.sh topic can be read by anyone who knows its name: pick a
          long random one, or run your own ntfy server.</div>
        <div id="ef-pushmsg" class="note">${pl ? (pl.ok ? `Last push: delivered ${esc(ago(pl.at))}.`
          : `<span class="msg err">Last push failed ${esc(ago(pl.at))}: ${esc(pl.error)}</span>`) : ""}</div>
      </div>
      <div class="actions ef-actions">
        <button type="submit" class="primary">Save</button>
        <span id="ef-setmsg" class="msg"></span>
        <button type="button" id="ef-remove" class="ef-danger">Remove ember</button>
      </div>
      <div class="note">Wiki folder: <code>${esc(c.root)}</code> · change where all embers live under Folder in the list.</div>
    </form>`);
  $("#ef-pushtest").onclick = pushTest;
  $("#ef-remove").onclick = remove;
}

function slotField(s, value) {
  const auto = AUTO.includes(s.type);
  return `<div class="ef-slot">
    <label class="f grow"><span class="lbl">${esc(s.label || s.id)}${s.required ? " *" : ""}</span>
      ${auto ? `<input disabled value="automatic">`
             : `<input name="slot-${esc(s.id)}" value="${esc(value)}" placeholder="${esc(s.type === "rss" && s.default ? s.default : "")}">`}</label>
    <div class="note">${esc(SLOT_HINT[s.type] || s.type)}</div>
  </div>`;
}

function formBindings(form, slots) {
  const b = {};
  slots.forEach(s => {
    const i = form.elements["slot-" + s.id];
    if (i && i.value.trim()) b[s.id] = i.value.trim();
  });
  return b;
}

async function saveSettings(form) {
  const c = card(E.sel), f = form.elements, msg = $("#ef-setmsg");
  const kind = f["push-kind"].value, url = f["push-url"].value.trim();
  const body = {
    id: c.id, name: f.name.value, model: f.model.value, enabled: f.enabled.checked,
    bindings: formBindings(form, c.slots),
    jobs: { ingest: f["job-ingest"].value.trim(), brief: f["job-brief"].value.trim(), lint: f["job-lint"].value.trim() },
    push: kind ? { kind, url } : null,
  };
  msg.className = "msg work"; msg.textContent = "Saving…";
  const d = await api("/api/embers/update", body).catch(() => null);
  if (!d || d.error) { msg.className = "msg err"; msg.textContent = d ? d.error : "The panel didn't answer."; return false; }
  msg.className = "msg ok"; msg.textContent = "Saved.";
  await refresh(true);
  $(".ef-tname").textContent = card(E.sel).name;
  return true;
}

async function pushTest() {
  const out = $("#ef-pushmsg");
  if (!(await saveSettings($("#ef-set")))) return;
  setHTML(out, `<span class="msg work">Sending…</span>`);
  const d = await api("/api/embers/push/test", { id: E.sel }).catch(() => null);
  // A delivery failure comes back as {ok: false, error}; a bare {error} is the panel refusing.
  setHTML(out, d && d.ok ? `<span class="msg ok">Sent. Check your phone or channel.</span>`
    : d && d.ok === false ? `<span class="msg err">Not delivered: ${esc(d.error)}</span>`
    : `<span class="msg err">${esc(d ? d.error : "The panel didn't answer.")}</span>`);
}

async function remove() {
  const c = card(E.sel);
  if (!(await askYes(`It stops running. Its wiki stays on disk in ${c.root}.`,
    {title: `Remove ${c.name}`, ok: "Remove", danger: true}))) return;
  const d = await api("/api/embers/delete", { id: c.id }).catch(() => null);
  if (!d || d.error) return toast(d ? d.error : "The panel didn't answer.", "err");
  toast("Removed. The wiki is still on disk.", "ok");
  E.sel = null;
  await refresh(false);
}

/* ---------- source viewer ---------- */
async function openRaw(sha, quote) {
  const d = await api(`/api/embers/raw?id=${encodeURIComponent(E.sel)}&sha=${encodeURIComponent(sha)}`).catch(() => null);
  if (!d || d.error) return toast(d ? d.error : "The panel didn't answer.", "err");
  const text = d.text;
  let at = quote ? text.indexOf(quote) : -1;
  if (at < 0 && quote) at = text.toLowerCase().indexOf(quote.toLowerCase());
  const body = at < 0 ? esc(text)
    : esc(text.slice(0, at)) + `<mark id="ef-mark">${esc(text.slice(at, at + quote.length))}</mark>` + esc(text.slice(at + quote.length));
  showModal("Source snapshot", `
    <div class="note">${esc(d.source || "")}${d.title ? " · " + esc(d.title) : ""}${d.fetched ? " · read " + esc(ago(d.fetched)) : ""}
      · <code>raw/${esc(sha)}.txt</code></div>
    ${quote ? `<div class="ef-quoteline">Quote: <q>${esc(quote)}</q>${at < 0
      ? ` <span class="note">(found after ignoring spacing and punctuation, so it isn't highlighted)</span>` : ""}</div>` : ""}
    <pre class="ef-raw">${body}</pre>
    ${d.truncated ? `<div class="note">Showing the first part only.</div>` : ""}
    <div class="note">A checked quote means these words are really in the snapshot. Whether they support the
      item's wording is for you to judge.</div>`);
  const m = $("#ef-mark");
  if (m) m.scrollIntoView({ block: "center" });
}

/* ---------- new ember ---------- */
async function openNew() {
  const d = await api("/api/embers/templates").catch(() => null);
  if (!d || d.error) return toast(d ? d.error : "The panel didn't answer.", "err");
  const m = showModal("New ember", `
    <div class="note">Pick a starting point. You can change sources, schedule and model later.</div>
    <div class="ef-tpls">
      <button type="button" class="ef-tpl ef-tplforge" data-forge-new>
        <span class="ef-tpltitle">Talk to Forge</span>
        <span class="ef-tplmission">Say what you want in your own words. Forge asks a few questions and builds an ember
          that fits, using a model you already loaded (12B+ instruct works best).</span>
      </button>${d.templates.map((t, i) => `
      <button type="button" class="ef-tpl" data-tpl="${i}">
        <span class="ef-tpltitle">${esc(t.title)}${t.zero_setup ? ` <span class="ef-tag ok">no setup</span>` : ""}</span>
        <span class="ef-tplmission">${esc(t.mission)}</span>
        ${t.about ? `<span class="note">${esc(t.about)}</span>` : ""}
      </button>`).join("")}</div>
    <div id="ef-newform"></div>`);
  $(".ef-tplforge").onclick = () => { m.close(); openForge(); };
  $$(".ef-tpl:not(.ef-tplforge)").forEach(b => b.onclick = () => {
    $$(".ef-tpl").forEach(x => x.classList.toggle("on", x === b));
    newForm(d.templates[Number(b.dataset.tpl)], m);
  });
}

function newForm(t, m) {
  const el = $("#ef-newform");
  const needs = t.slots.filter(s => !AUTO.includes(s.type));
  setHTML(el, `<form id="ef-createf" class="ef-createf">
      ${needs.length ? `<div class="note">${t.slots.some(s => s.required) ? "Fields marked * are needed."
        : "Fill in at least one source; leave the rest blank."}</div>` : ""}
      ${t.slots.map(s => slotField(s, "")).join("")}
      <div class="ef-slot"><label class="f grow"><span class="lbl">Folder name (optional)</span>
        <input name="id" maxlength="64" placeholder="${esc(t.name)}" pattern="[a-z0-9][a-z0-9-]*"></label>
        <div class="note">Lowercase letters, digits and dashes. Becomes the wiki's folder.</div></div>
      <div class="actions"><button type="submit" class="primary">Create and run</button><span id="ef-newmsg" class="msg"></span></div>
      <div class="note">The first run reads the sources, then writes a brief, using a model that's loaded
        (it waits if you're busy on the GPU). Schedule: ${esc(Object.entries(t.jobs).map(([k, v]) => `${JOB_LABEL[k]} ${v}`).join(" · "))}.</div>
    </form>`);
  const form = $("#ef-createf");
  form.dataset.tpl = t.name;
  form.onsubmit = async e => {
    e.preventDefault();
    const msg = $("#ef-newmsg");
    msg.className = "msg work"; msg.textContent = "Creating…";
    const r = await api("/api/embers/create", {
      template: t.name, id: form.elements.id.value.trim(), bindings: formBindings(form, t.slots) }).catch(() => null);
    if (!r || r.error) { msg.className = "msg err"; msg.textContent = r ? r.error : "The panel didn't answer."; return; }
    m.close();
    E.sel = r.id; E.sub = "brief"; E.page = null; E.forge = false;
    try { localStorage.setItem("lf_ember", r.id); } catch (e2) {}
    toast(r.queued.length ? "Created. First run queued." : "Created.", "ok");
    await refresh(false);
  };
  const first = $("input[name^=slot-]", form);
  if (first) first.focus();
}

/* ---------- Forge ---------- */
function openForge() {
  E.forge = true;
  renderList();
  renderForge($("#ef-main"));
}

/* ---------- events ---------- */
async function onClick(e) {
  if (e.target.closest("[data-forge-open]")) return openForge();
  const fg = e.target.closest("[data-fg]");
  if (fg) return forgeReset();
  const row = e.target.closest("[data-ember]");
  if (row) {
    E.forge = false;
    E.sel = row.dataset.ember; E.page = null; E.sig = sigOf(card(E.sel));
    try { localStorage.setItem("lf_ember", E.sel); } catch (e2) {}
    renderList(); renderDetail();
    return;
  }
  const folder = e.target.closest("[data-folder]");
  if (folder) {
    const act = folder.dataset.folder;
    return act === "edit" ? editFolder() : act === "default" ? saveFolder("") : renderFolder();
  }
  const sub = e.target.closest("[data-sub]");
  if (sub) return showSub(sub.dataset.sub);
  const run = e.target.closest("[data-run]");
  if (run) {
    run.disabled = true;
    const d = await api("/api/embers/run", { id: E.sel, job: run.dataset.run }).catch(() => null);
    run.disabled = false;
    if (!d || d.error) return toast(d ? d.error : "The panel didn't answer.", "err");
    toast("Queued: " + d.queued.map(j => JOB_LABEL[j] || j).join(", "), "ok");
    return refresh(true);
  }
  if (e.target.closest("[data-cancel]")) {
    await api("/api/embers/cancel", { id: E.sel });
    return refresh(true);
  }
  const q = e.target.closest(".ef-quote, .ef-rawlink");
  if (q) {
    e.preventDefault();
    const sib = q.parentElement && q.parentElement.querySelector(":scope > q");
    return openRaw(q.dataset.raw, q.classList.contains("ef-rawlink") ? ""
      : q.getAttribute("title") || (sib ? sib.textContent : ""));
  }
  const link = e.target.closest(".ef-link");
  if (link) {
    e.preventDefault();
    E.page = link.dataset.page;
    if (E.sub !== "wiki") {
      await showSub("wiki");
    }
    return openPage(link.dataset.page, link.dataset.item);
  }
  const pg = e.target.closest(".ef-pg");
  if (pg) { e.preventDefault(); return openPage(pg.dataset.pg); }
}

function onSubmit(e) {
  if (e.target.id === "ef-askf") {
    e.preventDefault();
    const q = $("#ef-q").value.trim();
    if (q) ask(q);
  } else if (e.target.id === "fg-form") {
    e.preventDefault();
    forgeSend();
  } else if (e.target.id === "ef-folderform") {
    e.preventDefault();
    saveFolder(e.target.elements.path.value);
  } else if (e.target.id === "ef-set") {
    e.preventDefault();
    saveSettings(e.target);
  }
}
