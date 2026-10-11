---
title: Embers
section: guides
order: 10
---

# Embers

Embers are small, always-on local agents. Each one keeps watch on a topic, reads the sources you point it at, keeps its own wiki, and writes you a brief on a schedule. They run on a model served by your own llama.cpp router; nothing leaves the machine unless you turn on push notifications.

Embers have no browser, no tools and no way to send anything on your behalf. They read their sources and write their wiki. That's it.

## What it does

An ember has three scheduled jobs:

| Job | Default time | What it does |
|---|---|---|
| `ingest` | `02:00` | Reads new items from the sources, saves an immutable raw snapshot of each, and asks the model to propose wiki updates. |
| `brief` | `07:00` | Writes the brief from the verified items in the wiki. |
| `lint` | `sun 03:00` | Looks for stale items, missing evidence and contradictions. |

The reliability rule is in code, not in the prompt: the model proposes, code decides. A wiki item only counts as verified if it carries a verbatim quote that really occurs in one of the raw snapshots. Owners and due dates have to appear inside such a quote, so the model can't invent who owes what or by when.

### Sources

Sources are read-only. Each adapter caps what it reads, because a source can be large or hostile.

| Type | Reads |
|---|---|
| `folder` | `.md` and `.txt` files in a folder you choose. |
| `ics` | A calendar file or URL. |
| `rss` | An RSS or Atom feed. |
| `git` | Commits in a local repository (treated as finished work). |
| `llamacpp` | New llama.cpp releases. Nothing to bind. |
| `machine` | A snapshot of this machine's GPUs, RAM and installed models, only when it changed. Nothing to bind. |

A source that can't be read is skipped for that run, the run carries on, and the run's note in **Activity** names the source.

### Templates

Two templates ship with LlamaForge:

- **Model Scout** needs no setup beyond a model: it runs on whatever is loaded, or loads one itself if you pick one in its Settings or set an auto-load model in Setup. It watches new llama.cpp releases against this machine's snapshot and tells you each morning what is new and whether it plausibly fits your hardware.
- **Morning Brief** tracks loose ends (promises, things you're waiting on, deadlines, meetings) from a notes folder, a calendar, a repo and a feed.

Templates are treated as untrusted input. They never carry paths, credentials or tool definitions; your folders, URLs and repos live in the ember's own `ember.json`.

### Forge

**Forge** builds an ember by interviewing you. It keeps a live blueprint beside the conversation and only creates the ember when you say yes. It uses a model you already have loaded. Paths and URLs can only come from what you typed, and they must exist on this machine; Forge can't invent one.

## How to use it

1. Open the **Embers** tab and pick a template, or talk to **Forge**.
2. Bind the sources the template asks for (Model Scout has none).
3. Click **Run now** to read the sources and write the first brief. **Brief only** rewrites the brief from the wiki as it is; **Check wiki** runs the lint job.
4. Use the tabs on an ember:
   - **Brief**: today's brief, with each claim citing its source snapshot.
   - **Wiki**: the pages the ember keeps.
   - **Ask**: a question answered from the ember's verified items. If the answer names a person, number or link that neither your question nor those items contain, the panel flags it as not grounded.
   - **Activity**: every run with its result, duration, tokens in and out, and a note, plus the ember's `log.md`.
   - **Settings**: name, model, sources, whether it runs on schedule, job times, push.

**Remove ember** stops it running. Its wiki stays on disk.

## Sharing the GPU

The scheduler only contacts the router when a job is due. A job waits while the router is busy or down, and is skipped after an hour. An ember with no model of its own uses whatever is loaded; with nothing loaded it uses Setup's auto-load model like a pinned one, and with neither it waits. An ember pinned to a different model swaps it in only after the router has been watched idle for 10 minutes, then puts your model back, unless you loaded something else meanwhile. If you load or unload a model while a job runs, the job stops cleanly and the GPU is yours. Loading a model for an ember can be turned off with `embers_swap_models`; a job that needs another model then waits instead.

With [multi-model](setup.md) on, an ember never evicts anything. A job whose model isn't up loads it as a worker beside yours, only when nothing is generating and the planner says it fits (no idle wait), then unloads it afterwards unless you made it the main model or are using it. A model pinned to its own build (outside the router) can't be used by an ember.

## Push notifications (optional)

An ember can push its brief to **ntfy** or a **webhook** (the payload works with Slack and Discord). Only the title, headline and first three bullets are sent: no wiki pages, no quotes, no raw text.

## Reference

| Setting | Where | Default |
|---|---|---|
| Ember data folder | `config.json` `embers_dir`, or the folder chooser in the Embers tab | `<root>/embers` |
| Automatic runs | `config.json` `embers_scheduler` | `true` (**Run now** works either way) |
| Loading a model for an ember | `config.json` `embers_swap_models` | `true` |
| Job times | ember **Settings** | `ingest 02:00`, `brief 07:00`, `lint sun 03:00` |

Times are local, in `HH:MM` or `day HH:MM` form (`mon`–`sun`).

## Troubleshooting

- **"Automatic runs are off"**: `embers_scheduler` is `false` in `config.json`. **Run now** still works.
- **An ember never runs on its own**: check **Runs on schedule** in its **Settings**.
- **A run's note names a source**: it couldn't be read (missing folder, unreachable URL, not a git repo). Fix it in **Settings**; the next run picks it up.
- **Nothing in the brief**: either the ingest found nothing new, or none of the model's proposals had a quote that matched the raw text, so nothing was verified. The run's `log.md` lines in **Activity** show what happened.

See also [MCP Server](mcp.md) and [Connect an Agent](agents.md).
