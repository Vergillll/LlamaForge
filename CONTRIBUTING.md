# Contributing to LlamaForge

Thanks for helping! LlamaForge is a small, solo-maintained project, so a few
conventions keep things moving fast.

## Ground rules
- **Backend is Python standard library only.** No new required dependencies.
  Optional extras (like the tray icon) must be imported lazily and degrade to a no-op.
- **Frontend is vanilla JS** in `web/` — no framework, no build step. Every dynamic
  value goes through `esc()`, and `setHTML()` is the single HTML sink.
- **LlamaForge never does inference itself.** It drives llama.cpp / ik_llama / vLLM.
  Features that belong upstream should go upstream.

## Running tests
```bash
cd tests
python -m unittest discover -s . -p "test_*.py"
```
Tests must be run from `tests/`. Tests that touch config must point
`config.CONFIG` at a temp file — never the real `config.json`.

The frontend has Node tests too (CI uses Node 20), run from the repo root:
```bash
node --test tests/*.mjs
```
One of them fails the build if a view calls the native `confirm()`, `prompt()` or
`alert()`; some hosts silently cancel them. Ask in the panel with `askYes()` /
`askText()` from `web/js/core.js` instead.

## Running the UI
`python backend/server.py` starts the panel on port 8090. It renders even with no
engine running, which is handy for UI work.

## Pull requests
- **Comment on the issue before you open a PR for it.** A line like "I'll take
  this" is enough. It stops two people fixing the same bug, and it gives the
  person who reported it a chance to say they already have a fix.
- **The reporter goes first.** If the person who opened an issue says they have a
  fix, give them the chance to send it. A PR that lands ahead of theirs with the
  same fix may be closed in favour of theirs.
- **Credit what you use.** If your PR includes code or tests someone posted in an
  issue or review, say so in the PR and add them as a co-author
  (`Co-authored-by: Name <email>` in the commit).
- **AI-assisted PRs are welcome** (LlamaForge itself is heavily AI-assisted), as
  long as you've read the diff, run the tests yourself, and will be around to
  answer review comments. Unclaimed PRs that look like they came from an
  automated issue-to-PR pipeline may be closed without review.
- Keep PRs focused — one fix or feature each.
- Include a screenshot for UI changes, ideally in both light and dark themes, and
  check narrow widths (the sidebar changes layout below 900px, 760px and 600px).
- Small PRs get reviewed fastest. For bigger ideas, open an issue or Discussion first.

## Good first issues
Look for the [`good first issue`](https://github.com/dadwritestech/LlamaForge/labels/good%20first%20issue)
label.
