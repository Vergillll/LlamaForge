"""Ember pipelines: create/open, ingest, brief, lint.

Pure orchestration. The model (llm), the clock (now) and source reading
(fetch) are passed in, so tests drive everything with fakes. sqlite holds the
truth; markdown pages are re-rendered from it after each committed batch.

Consistency: model calls never happen inside a db transaction. A batch's
verified ops, its raws' "done" mark and the list of pages to re-render
(meta "dirty_pages") commit together; the markdown is written afterwards and a
page leaves the dirty list only once its file is written. A crash between the
commit and the render is repaired at the start of the next run.
"""
import collections, datetime as dt, json, math, os, re

import atomicio
from embers import prompts, reserved_name, sources, templates, verify, wikifs
from embers.llm import LLMError, PromptTooLarge, ReplyTruncated, RouterUnavailable, ThinkingOverflow
from embers.store import Store, ts

ID_RE            = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
BUDGET_SHARE     = 0.45      # of the model's n_ctx, for raw text in one batch
PROMPT_SHARE     = 0.75      # of n_ctx, for the whole prompt; the rest is left for the reply
CHARS_PER_TOKEN  = 4
MAX_BATCH_TOKENS = 6000      # raw text per batch even when n_ctx is huge: the reply must fit MAX_REPLY_TOKENS
MAX_BATCH_RAWS   = 12        # raws per batch: small models lose track of ids in long lists
OVERFLOW_RETRIES = 2         # extra tries for one raw the router says does not fit
TOKEN_MARGIN     = 1.1       # on top of the router's own count when the estimate proved low
FLAG_MIN_OPS     = 3         # proposed changes before "none of them quoted the sources" is a warning
MIN_REPLY_TOKENS = 256
MAX_REPLY_TOKENS = 4096
CONTEXT_PAGES    = 3
CONTEXT_ITEMS    = 40        # open items shown per matched page
MAX_RAWS_PER_RUN = 200       # pending raws folded per run; the rest wait for the next run
RAW_CAP_BYTES    = 200 * 1024 * 1024
DIRTY_KEY        = "dirty_pages"
ATTEMPTS_KEY     = "raw_attempts"
SOLO_AFTER       = 2         # failed batch attempts before a raw is retried on its own
MAX_ATTEMPTS     = 3         # failed attempts before a raw is marked "failed" for good
MIN_N_CTX        = 2048      # below this the fixed instructions leave no room for sources
MAX_REJECTIONS   = 20        # rejection reasons kept in a run's detail
_ORPHAN_RE       = re.compile(r"[0-9a-f]{12}\.txt")
_WORD = re.compile(r"\w{3,}")
_STOP = frozenset("""the and for that this with from have will your you are was were been into about
there their what when which would could should them they then than just also only over more some
such file not but can our out has had its who how all any""".split())


class Ember:
    """An ember folder opened for work. Use as a context manager or close()."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "ember.json"), encoding="utf-8") as f:
            self.conf = json.load(f)
        self.template = templates.parse_template(self.conf["template"])
        self.store = Store(os.path.join(root, "embers.db"))

    @property
    def id(self):
        return self.conf["id"]

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def create_ember(embers_dir, template, ember_id, bindings, now, name=None, origin=None):
    """template: a parse_template() result. bindings: {slot_id: path_or_url}.
    origin: who made it ("forge"), recorded when given. Returns the new ember's folder."""
    if not isinstance(ember_id, str) or not ID_RE.fullmatch(ember_id):
        raise ValueError("ember id must be lowercase letters, digits and dashes")
    if reserved_name(ember_id):
        raise ValueError(f"ember id {ember_id!r} is a reserved device name")
    root = os.path.join(embers_dir, ember_id)
    if os.path.exists(os.path.join(root, "ember.json")):
        raise ValueError(f"ember {ember_id!r} already exists")
    clean = clean_bindings(template, bindings)
    os.makedirs(root, exist_ok=True)
    wikifs.init_wiki(root, template["title"], template["schema_md"])
    conf = {"id": ember_id, "name": name or template["title"],
            "template": {k: v for k, v in template.items() if k != "dropped"},
            "bindings": clean, "model": "", "enabled": True, "created": ts(now)}
    if origin:
        conf["origin"] = origin
    atomicio.write_json(os.path.join(root, "ember.json"), conf)
    return root


def clean_bindings(template, bindings):
    """{slot_id: path_or_url} checked against the template's slots: unknown
    slots refused, blanks dropped, rss defaults filled, required slots bound."""
    if not isinstance(bindings, dict):
        raise ValueError("bindings must be an object")
    slots = {s["id"]: s for s in template["slots"]}
    unknown = sorted(set(bindings) - set(slots))
    if unknown:
        raise ValueError(f"unknown slot(s): {', '.join(map(str, unknown))}")
    clean = {k: str(v).strip() for k, v in bindings.items() if v is not None and str(v).strip()}
    for s in template["slots"]:
        if s["type"] == "rss" and s.get("default") and s["id"] not in clean:
            clean[s["id"]] = s["default"]
    missing = [s["id"] for s in template["slots"]
               if s["required"] and s["type"] not in templates.AUTO_SLOTS and s["id"] not in clean]
    if missing:
        raise ValueError(f"required slot(s) not bound: {', '.join(missing)}")
    return clean


def _bound(ember):
    for s in ember.template["slots"]:
        b = ember.conf["bindings"].get(s["id"], "")
        if b or s["type"] in templates.AUTO_SLOTS:
            yield s, b


def _terms(text, limit=16):
    """The most frequent unicode words (3+ chars, casefolded) of text, for search."""
    counts = {}
    for w in _WORD.findall(text.casefold()):
        if w not in _STOP:
            counts[w] = counts.get(w, 0) + 1
    return " ".join(sorted(counts, key=lambda w: (-counts[w], w))[:limit])


def _est(text):
    """Conservative token count: ~4 ASCII characters per token, but every
    non-ASCII character (CJK, emoji, accents) counted as a whole token."""
    wide = sum(1 for c in text if ord(c) > 127)
    return wide + -(-(len(text) - wide) // CHARS_PER_TOKEN)


def _cut(text, tokens):
    """The longest prefix of text whose _est() is at most `tokens`."""
    if _est(text) <= tokens:
        return text
    lo, hi = 0, len(text)              # _est of a prefix only grows with its length
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _est(text[:mid]) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def _batches(raws, budget):
    """Group raws into batches of at most `budget` estimated tokens and
    MAX_BATCH_RAWS raws; a raw over the budget is cut to fit and goes alone."""
    batch, size = [], 0
    for r in raws:
        text = _cut(r["text"], budget)
        n = _est(text)
        if batch and (size + n > budget or len(batch) >= MAX_BATCH_RAWS):
            yield batch
            batch, size = [], 0
        batch.append(dict(r, text=text))
        size += n
    if batch:
        yield batch


def _schema_text(ember):
    try:
        with open(os.path.join(ember.root, "SCHEMA.md"), encoding="utf-8") as f:
            return f.read()[:templates.MAX_SCHEMA]
    except (OSError, ValueError):
        return ember.template["schema_md"]


def _prompt(tpl, schema, index_head, context, batch, cap, today=None):
    """Build the ingest prompt and shrink it until it fits `cap` estimated
    tokens: first the matched pages, then the index excerpt, the schema, and
    last the raw text. Returns (messages, size_in_tokens, batch_as_shown)."""
    context = [dict(p, items=list(p["items"])) for p in context]
    batch = [dict(r) for r in batch]
    while True:
        msgs = prompts.ingest_messages(tpl["mission"], schema, tpl["page_kinds"], index_head, context, batch, today)
        size = sum(_est(m["content"]) for m in msgs)
        over = size - cap
        if over <= 0:
            return msgs, size, batch
        if context:
            if context[-1]["items"]:
                context[-1]["items"].pop()
            else:
                context.pop()
        elif index_head:
            index_head = _cut(index_head, max(0, _est(index_head) - over))
        elif schema:
            schema = _cut(schema, max(0, _est(schema) - over))
        else:
            longest = max(batch, key=lambda r: _est(r["text"]), default=None)
            if not longest or not longest["text"]:
                return msgs, size, batch   # the fixed instructions alone exceed the cap
            longest["text"] = _cut(longest["text"], max(0, _est(longest["text"]) - over))


def _render_page(ember, page):
    st = ember.store
    items = st.items_for_page(page)
    p = st.get_page(page) or {"title": page, "summary": ""}
    wikifs.write_page(ember.root, page, p["title"], p["summary"],
                      wikifs.render_items(items, st.evidence_for([i["id"] for i in items])))


def _valid_page(page):
    """True when wikifs can render `page` (format and reserved names)."""
    try:
        wikifs._split_page(page)
    except wikifs.WikiError:
        return False
    return True


def _write_index(ember, lint_md=None):
    # A page row with an invalid name (only possible by editing the db) is left
    # out rather than failing every index write; lint flags it as drift.
    wikifs.write_index(ember.root, [p for p in ember.store.all_pages() if _valid_page(p["page"])], lint_md)


def _dirty(st):
    try:
        v = st.get_meta(DIRTY_KEY, [])
    except (ValueError, RecursionError):   # the list itself is lost: treat every page as dirty
        return [p["page"] for p in st.all_pages()]
    return [p for p in v if isinstance(p, str)] if isinstance(v, list) else []


def _attempts(st):
    """meta "raw_attempts": {sha: {"attempts": n, "error": str}} for raws whose batch failed."""
    try:
        v = st.get_meta(ATTEMPTS_KEY, {})
    except (ValueError, RecursionError):
        return {}
    return {k: a for k, a in v.items() if isinstance(a, dict) and isinstance(a.get("attempts"), int)} \
        if isinstance(v, dict) else {}


def raw_failures(ember):
    """Raws whose batches failed: {sha: {"attempts": n, "error": last_error}}."""
    return _attempts(ember.store)


def _mark_dirty(st, pages):
    """Call inside the transaction that changed these pages."""
    st.set_meta(DIRTY_KEY, sorted(set(_dirty(st)) | set(pages)))


def _flush_dirty(ember):
    """Render every dirty page and the index, then clear the pages that made it
    to disk. Returns error strings; pages that failed stay dirty for next time."""
    st = ember.store
    dirty = _dirty(st)
    if not dirty:
        return []
    errors, done = [], set()
    for page in dirty:
        if not _valid_page(page):          # can never render: drop it instead of retrying forever
            errors.append(f"render {_one_line(page, 80)}: invalid page name")
            done.add(page)
            continue
        try:
            _render_page(ember, page)
            done.add(page)
        except wikifs.WikiError as e:      # a bad name can never render: drop it
            errors.append(f"render {page}: {e}")
            done.add(page)
        except (OSError, ValueError) as e:
            errors.append(f"render {page}: {e}")
    try:
        _write_index(ember)
    except (OSError, ValueError) as e:
        errors.append(f"index: {e}")
        return errors                      # keep everything dirty so the index is redone too
    with st.db:
        st.set_meta(DIRTY_KEY, sorted(set(_dirty(st)) - done))
    return errors


def _apply(ember, ops, new_pages, now):
    """Write verified ops to sqlite (caller holds the transaction).
    Returns the set of pages whose markdown must be re-rendered."""
    st, touched = ember.store, set()
    meta = {p["page"]: p for p in new_pages}
    for op in ops:
        if op["op"] == "add":
            page = op["page"]
            m = meta.get(page, {})
            st.ensure_page(page, m.get("title") or page.split("/")[1].replace("-", " ").title(),
                           m.get("summary", ""), now)
            iid = st.new_item_id(page, op["text"], now)
            st.add_item(iid, page, op["kind"], op["text"], op["owner"], op["due"], op["verified"], now)
        else:
            iid = op["item"]
            old = st.get_item(iid)
            if old is None:
                continue
            page = old["page"]
            fields = {}
            # Unverified text never overwrites an item that is already verified.
            if op["op"] == "update" and op["text"] and (op["verified"] or not old["verified"]):
                fields["text"] = op["text"]
            if op["owner"]:
                fields["owner"] = op["owner"]
            if op["due"]:
                fields["due"] = op["due"]
            if op["verified"]:
                fields["verified"] = 1
            if op["op"] == "close":
                fields["status"] = "closed"
            if fields or op["evidence"]:
                st.update_item(iid, now, **fields)
        for ev in op["evidence"]:
            st.add_evidence(iid, ev["raw"], ev["quote"], now)
        touched.add(page)
    for page in touched:
        st.index_page(page)
    return touched


def _err(e):
    return f"{type(e).__name__}: {e}"


def ingest(ember, llm, now, fetch=sources.fetch, n_ctx=8192):
    """Fetch new source items into raws, then fold pending raws into the wiki
    batch by batch. Returns a stats dict with "status", "summary" and "router_down"
    (True only when a model call raised RouterUnavailable, never from error text).
    The run row is always closed: on an unexpected error it is finished as
    "failed" with the error, then the error propagates."""
    st = ember.store
    with st.db:
        st.abort_running("ingest", now)    # rows left "running" by a process that died
        run = st.start_run("ingest", now)
    errors = []
    try:
        return _ingest(ember, llm, now, fetch, n_ctx, run, errors)
    except BaseException as e:
        _close_failed(st, run, now, "; ".join(errors + [_err(e)]))
        raise


def _close_failed(st, run, now, error):
    """Best-effort close of a run that raised; the original error matters more,
    so nothing here may raise. First the normal finish_run; if that fails too, a
    bare UPDATE in its own statement; if even that fails the row stays
    'running' and the next run of the job aborts it (Store.abort_running).
    Note abort_running would also abort a live concurrent run of the same job:
    callers must hold embers.lock.held(root) around runs (the CLI does; the M2
    scheduler will)."""
    try:
        st.db.rollback()
    except Exception:
        pass
    try:
        with st.db:
            st.finish_run(run, now, "failed", error)
        return
    except Exception:
        pass
    try:
        with st.db:
            st.db.execute("UPDATE runs SET status='failed', finished=?, error=? WHERE id=? AND status='running'",
                          (ts(now), (error or "")[:2000], run))
    except Exception:
        pass


def _settle_failure(st, shas, error, stats):
    """Count a failed attempt against each raw of a batch (in one transaction);
    raws out of attempts are marked "failed" so they stop blocking cursors."""
    with st.db:
        att = _attempts(st)
        for sha in shas:
            a = att.get(sha) or {"attempts": 0}
            att[sha] = {"attempts": a["attempts"] + 1, "error": error[:300]}
            if att[sha]["attempts"] >= MAX_ATTEMPTS:
                st.mark_raws([sha], "failed")
                stats["failed_raws"] += 1
        st.set_meta(ATTEMPTS_KEY, att)


def _ingest(ember, llm, now, fetch, n_ctx, run, errors):
    st, root, tpl = ember.store, ember.root, ember.template
    errors += _flush_dirty(ember)          # repair a run that died between commit and render

    stale, new_by_source, cursors = {}, {}, {}
    for slot, binding in _bound(ember):
        sid = slot["id"]
        try:
            cursor_in = st.get_cursor(sid)
        except (ValueError, RecursionError):
            cursor_in = None
        if not isinstance(cursor_in, dict):
            cursor_in = {}
            errors.append(f"{sid}: corrupt cursor reset")
        try:                               # any failure isolates this source only
            items, cursor = fetch(slot["type"], binding, cursor_in, now)
            shas = []
            with st.db:
                for it in items:
                    text = sources.clip(it["text"])
                    sha = wikifs.write_raw(root, text)
                    st.add_raw(sha, sid, it["ref"], it["title"], now, len(text.encode("utf-8")))
                    shas.append(sha)
        except Exception as e:
            stale[sid] = _err(e) if not isinstance(e, sources.SourceError) else str(e)
            errors.append(f"{sid}: {_err(e)}")
            continue
        new_by_source[sid], cursors[sid] = shas, cursor

    queued = st.pending_raws()
    pending, read_errors = [], 0
    for r in queued[:MAX_RAWS_PER_RUN]:
        try:
            text = wikifs.read_raw(root, r["sha"])
        except FileNotFoundError:
            text = None
        except OSError as e:               # locked, permissions, network share: try again next run
            errors.append(f"raw {r['sha']}: {_err(e)}")
            read_errors += 1
            continue
        except ValueError as e:            # bad id or undecodable bytes: never readable
            errors.append(f"raw {r['sha']}: {_err(e)}")
            text = None
        if text is None:
            with st.db:
                st.mark_raws([r["sha"]], "missing")
        else:
            pending.append(dict(r, text=text))
    full_text = {r["sha"]: r["text"] for r in pending}
    slot_type = {sl["id"]: sl["type"] for sl in tpl["slots"]}
    done = {r["sha"] for r in pending if slot_type.get(r["source"]) in sources.DONE_TYPES}

    stats = {"raws": len(pending), "deferred": max(0, len(queued) - MAX_RAWS_PER_RUN), "batches": 0,
             "failed": 0, "failed_raws": 0, "truncated": 0, "split": 0, "ops": 0, "unverified": 0, "rejected": 0, "finished": 0,
             "tokens_in": 0, "tokens_out": 0}
    rejections = []
    too_small = n_ctx < MIN_N_CTX
    if too_small:
        errors.append(f"model context n_ctx={n_ctx} is below the minimum of {MIN_N_CTX} tokens; "
                      "load the model with a larger context")
        pending = []
    scale = 1.0                            # router tokens per estimated token; raised by an overflow

    def budget():
        return max(1, int(min(n_ctx * BUDGET_SHARE, MAX_BATCH_TOKENS) / scale))
    schema = _schema_text(ember)
    att = _attempts(st)
    first = {}                             # one source's raws share words and pages: batch them together
    for r in pending:
        first.setdefault(r["source"], len(first))
    # Finished work goes last: a model with a small thinking budget spends it on the
    # first raws of a batch, and those should be the ones that can still be open.
    grouped = sorted((r for r in pending if att.get(r["sha"], {}).get("attempts", 0) < SOLO_AFTER),
                     key=lambda r: (r["sha"] in done, first[r["source"]]))
    alone = [r for r in pending if att.get(r["sha"], {}).get("attempts", 0) >= SOLO_AFTER]
    queue = collections.deque((b, 0) for b in list(_batches(grouped, budget()))
                              + [b for r in alone for b in _batches([r], budget())])
    index_error = False
    router_down = False
    while queue:
        batch, tries = queue.popleft()
        stats["batches"] += 1
        context = []
        for page in st.search(_terms(" ".join(r["text"] for r in batch)), CONTEXT_PAGES):
            p = st.get_page(page) or {"title": page}
            context.append({"page": page, "title": p["title"],
                            "items": [i for i in st.items_for_page(page) if i["status"] == "open"][:CONTEXT_ITEMS]})
        try:
            index_head = wikifs.index_head(root)
        except (OSError, ValueError) as e:
            index_head = ""
            if not index_error:
                errors.append(f"index head: {_err(e)}")
                index_error = True
        msgs, size, shown = _prompt(tpl, schema, index_head, context, batch, int(n_ctx * PROMPT_SHARE / scale),
                                    now.date())
        shown_shas = [r["sha"] for r in shown if r["text"]]   # a raw cut to nothing was not seen
        reply_tokens = max(MIN_REPLY_TOKENS, min(MAX_REPLY_TOKENS, n_ctx - math.ceil(size * scale)))
        try:                               # the model call runs outside any transaction
            update, usage = llm(msgs, prompts.UPDATE_SCHEMA, reply_tokens)
        except RouterUnavailable as e:     # not the input's fault: stop, count nothing
            stats["failed"] += 1
            errors.append(_err(e))
            router_down = True
            break
        except ThinkingOverflow as e:      # the model's fault, not the input's: stop, blame no raw
            stats["failed"] += 1
            errors.append(_err(e))
            break
        except (ReplyTruncated, PromptTooLarge) as e:
            # The input was too much for this model, not wrong: send less instead of failing.
            if isinstance(e, PromptTooLarge):
                if e.n_ctx and e.n_ctx < n_ctx:            # e.g. a router slot smaller than /props said
                    n_ctx = e.n_ctx
                seen = e.n_prompt / size if e.n_prompt and size else 0
                scale = max(scale * TOKEN_MARGIN, seen * TOKEN_MARGIN)
            if len(batch) > 1:
                half = len(batch) // 2
                queue.extendleft([(batch[half:], tries), (batch[:half], tries)])
                stats["split"] += 1
            elif isinstance(e, PromptTooLarge) and tries < OVERFLOW_RETRIES:
                queue.appendleft((batch, tries + 1))     # the smaller cap cuts the raw to fit
            else:
                stats["failed"] += 1
                errors.append(_err(e))
                _settle_failure(st, shown_shas, _err(e), stats)
                continue
            stats["batches"] -= 1
            if isinstance(e, PromptTooLarge):          # what is still queued was sized with the old estimate
                queue = collections.deque((b, t) for old, t in queue for b in _batches(old, budget()))
            continue
        except Exception as e:
            stats["failed"] += 1
            errors.append(_err(e))
            _settle_failure(st, shown_shas, _err(e), stats)
            continue
        stats["truncated"] += sum(1 for r in shown if len(r["text"]) < len(full_text[r["sha"]]))
        usage = usage if isinstance(usage, dict) else {}
        for k in ("prompt_tokens", "completion_tokens"):
            try:
                stats["tokens_in" if k == "prompt_tokens" else "tokens_out"] += int(usage.get(k) or 0)
            except (TypeError, ValueError):
                pass
        # Quotes are checked against the full raw, but only raws shown in this batch count.
        raws = {sha: full_text[sha] for sha in shown_shas}
        try:
            ops, new_pages, rejected = verify.verify_batch(update, raws, tpl["page_kinds"], st.known_item_ids(),
                                                           now.date(), done)
            kept = []
            for op in ops:                 # an update/close acts on the page its item lives on
                old = st.get_item(op["item"]) if op["op"] != "add" else None
                if old is not None and old["page"] != op["page"]:
                    op = dict(op, page=old["page"])    # the item id was checked; a small model's page guess was not
                kept.append(op)
            ops = kept
            with st.db:                    # ops, raw status and the render list commit together
                touched = _apply(ember, ops, new_pages, now)
                st.mark_raws(shown_shas, "done")
                att = _attempts(st)
                if any(s in att for s in shown_shas):
                    st.set_meta(ATTEMPTS_KEY, {k: v for k, v in att.items() if k not in shown_shas})
                _mark_dirty(st, touched)
        except Exception as e:             # rolled back: the raws stay pending for the next run
            stats["failed"] += 1
            errors.append(f"apply: {_err(e)}")
            _settle_failure(st, shown_shas, f"apply: {_err(e)}", stats)
            continue
        stats["ops"] += len(ops)
        stats["unverified"] += sum(1 for o in ops if not o["verified"])
        stats["rejected"] += len(rejected)
        stats["finished"] += sum(1 for _, why in rejected if why == verify.DONE_WORK)
        rejections += [str(why) for _, why in rejected][:MAX_REJECTIONS - len(rejections)]
        errors += _flush_dirty(ember)

    with st.db:
        for sid, shas in new_by_source.items():
            # missing/pruned/failed raws can never become done; they must not pin the cursor
            if all(st.raw_status(s) != "pending" for s in shas):
                st.set_cursor(sid, cursors[sid])
    try:
        prune_raws(ember)
    except (OSError, ValueError) as e:
        errors.append(f"prune: {_err(e)}")
    render_failed = bool(_dirty(st))
    proposed = stats["ops"] + stats["rejected"] - stats["finished"]    # commits as loops quoted fine
    never_quoted = proposed >= FLAG_MIN_OPS and stats["ops"] == stats["unverified"]
    if never_quoted:
        errors.append(f"none of the model's {proposed} proposed changes quoted the sources exactly; "
                      "this model may be too small or too heavily quantized for Embers")
    if too_small or (stats["batches"] and stats["failed"] == stats["batches"]):
        status = "failed"
    else:
        status = ("partial" if stats["failed"] or stale or render_failed or read_errors or router_down or never_quoted
                  else "ok")
    summary = (f"{stats['raws']} new, {stats['ops']} ops ({stats['unverified']} unverified, "
               f"{stats['rejected']} rejected), {stats['failed']}/{stats['batches']} batches failed")
    for key, label in (("deferred", "deferred"), ("truncated", "truncated"), ("split", "batches split"), ("failed_raws", "raws given up")):
        if stats[key]:
            summary += f", {stats[key]} {label}"
    if stale:
        summary += f", stale: {', '.join(sorted(stale))}"
    try:
        wikifs.append_log(root, now, "ingest", summary)
    except (OSError, ValueError) as e:
        errors.append(f"log: {_err(e)}")
        if status == "ok":
            status = "partial"
    with st.db:
        st.finish_run(run, now, status, "; ".join(errors), stats["tokens_in"], stats["tokens_out"],
                      dict(stats, stale=stale, rejections=rejections))
    return dict(stats, run=run, status=status, stale=stale, summary=summary, router_down=router_down)


def prune_raws(ember, cap=RAW_CAP_BYTES):
    """Keep raw/ under `cap` bytes by deleting the oldest raws that are already
    settled (done or failed) and cited by no item. Referenced or pending raws are
    never pruned. Also removes orphan raw files (named like a raw, no db row),
    e.g. left by a source whose transaction rolled back. The file goes first,
    then the row: a crash in between leaves a row whose file is gone, which the
    next prune simply marks. Returns how many files were removed."""
    st = ember.store
    rows = st.all_raws()
    removed = 0
    known = {r["sha"] for r in rows}
    raw_dir = wikifs._safe_path(ember.root, "raw")
    try:
        entries = list(os.scandir(raw_dir))
    except FileNotFoundError:
        entries = []
    for entry in entries:
        if not _ORPHAN_RE.fullmatch(entry.name) or entry.name[:-4] in known:
            continue
        try:
            if entry.is_file(follow_symlinks=False):
                os.remove(entry.path)
                removed += 1
        except FileNotFoundError:
            pass
    total = sum(r["size"] or 0 for r in rows if r["status"] in ("pending", "done", "failed"))
    if total <= cap:
        return removed
    keep = st.referenced_raws()
    for r in rows:
        if total <= cap:
            break
        if r["status"] not in ("done", "failed") or r["sha"] in keep \
                or not verify.RAW_RE.fullmatch(r["sha"] or ""):
            continue
        try:
            os.remove(os.path.join(raw_dir, r["sha"] + ".txt"))
        except FileNotFoundError:
            pass
        with st.db:
            st.mark_raws([r["sha"]], "pruned")
        total -= r["size"] or 0
        removed += 1
    return removed


MAX_BRIEF_SECTIONS = 6
MAX_BRIEF_BULLETS  = 12
MAX_BRIEF_ITEMS    = 60      # open items offered to the model before the n_ctx cut
MAX_BRIEF_FLAGS    = 20      # lint flags shown with them
BRIEF_REPLY_TOKENS = 3000
BRIEF_FOOTNOTES    = 2       # evidence footnotes per bullet
LINT_FLAGS_KEY     = "lint_flags"
_one_line = prompts.one_line            # shared with lint
# A capitalised word the model may use at a sentence start without it occurring in the item.
_LEAD = frozenset("""today tomorrow tonight this these new still going waiting due overdue open nothing
one two three a an the you your no next now chase follow reply send remind check review call email ask
finish prepare schedule confirm submit pay book plan update note watch keep ping nudge get remember
expect sign also then and but so just please make look try""".split())
_EDGE = re.compile(r"^\W+|\W+$")
_TOKEN = re.compile(r"\w+")
_SENTENCE_END = tuple(".!?:;")
_LINKISH = re.compile(r"(?i)://|^www\.|@|%%")      # URLs, mail, Obsidian %%comments%%
_LINK_EDGE = ".,;:!?()[]{}<>\"'"
_DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
_MONTHS = "january february march april may june july august september october november december".split()
_DATE_NAMES = re.compile(r"\b(" + "|".join(_DAYS + _MONTHS) + r")\b")


def _needs_ground(word):
    return any(c.isdigit() or c.isupper() for c in word)


def _grounded_text(text, ground, title=False):
    """The brief's version of ingest's owner/due check. Every name or number in
    model text (a word holding a capital letter or a digit) must occur,
    word-bounded (verify._contains), in `ground`: the normalised verified text
    the model was shown. Allowed without grounding: "I"; a common word from
    _LEAD capitalised at a sentence start (text start, after . ! ? : ; or an
    opening parenthesis); with title=True (titles, the headline) any first word.
    A token that could link or hide text (scheme://, www., user@host, %%) must
    be grounded whole, whatever its case.
    Limit: only capitals, digits and link-like tokens are checked, so an
    invented claim in plain lowercase words ("sam was fired", "lawsuit filed")
    passes; a sentence-start capital from _LEAD ("Pay Sam") passes too. The
    bullet always links to the item and its verified evidence footnote.
    Caseless scripts (CJK) are only checked for digits."""
    start = True
    for n, chunk in enumerate(text.split()):
        at_start = start or chunk.startswith("(")
        start = chunk.rstrip(")]}\"'’”").endswith(_SENTENCE_END)
        if _LINKISH.search(chunk.strip(_LINK_EDGE)) or "%%" in chunk:
            core = chunk.strip(_LINK_EDGE)
            if not (core and verify._contains(ground, verify.normalise(core))):
                return False
            continue
        word = _EDGE.sub("", chunk)
        if not word or not _needs_ground(word):
            continue
        if verify._contains(ground, verify.normalise(word)):
            continue
        parts = _TOKEN.findall(word)                                      # Sam's, I'll, <b>SOW</b>
        if parts and (n == 0 and title or at_start and parts[0].isalpha() and parts[0].casefold() in _LEAD):
            parts = parts[1:]
        parts = [t for t in parts if t != "I" and _needs_ground(t)]
        if all(verify._contains(ground, verify.normalise(t)) for t in parts):
            continue
        return False
    return True


def _date_forms(d):
    """Ways a model may write the date `d`: Friday Fri 9 09 October Oct 2026."""
    return f"{d:%A %a} {d.day} {d:%d %B %b %Y}" + (" sept" if d.month == 9 else "")


def _ground(item, evidence, now):
    """Normalised text a bullet about `item` may draw names and numbers from:
    its verified text, the page, today's date and its due date written out, and
    3-letter forms of any day or month name in them (Friday -> Fri)."""
    extra = [_date_forms(now)]
    try:
        extra.append(_date_forms(dt.date.fromisoformat((item["due"] or "").strip()[:10])))
    except ValueError:
        pass
    text = verify.normalise(" ".join(
        [item["text"] or "", item["owner"] or "", item["due"] or "",
         item["page"].replace("/", " ").replace("-", " ")] + [e["quote"] for e in evidence] + extra))
    abbr = sorted({m[:3] for m in _DATE_NAMES.findall(text)} | ({"sept"} if "september" in text else set()))
    return text + (" " + " ".join(abbr) if abbr else "")


def _checked_evidence(ember, items):
    """{item_id: [evidence]} keeping only quotes that still occur in their raw
    (same test as verify.check_quote). Raws are read and normalised once each."""
    norm, out = {}, {}
    for iid, evs in ember.store.evidence_for([i["id"] for i in items]).items():
        for ev in evs:
            sha = ev["raw"]
            if sha not in norm:
                try:
                    text = wikifs.read_raw(ember.root, sha)
                except (OSError, ValueError):
                    text = None
                norm[sha] = verify.raw_forms(text) if text is not None else None
            if norm[sha] is not None and verify.quote_in(norm[sha], ev["quote"]):
                out.setdefault(iid, []).append(ev)
    return out


def _lint_flags(st):
    """meta "lint_flags", reduced to well-formed {"item", "why"} entries (it may be corrupt)."""
    try:
        v = st.get_meta(LINT_FLAGS_KEY, [])
    except (ValueError, RecursionError):
        return []
    return [{"item": f["item"], "why": f["why"]} for f in (v if isinstance(v, list) else [])
            if isinstance(f, dict) and isinstance(f.get("item"), str) and isinstance(f.get("why"), str)]


def _groups(items, titles):
    by_page = {}
    for it in items:
        by_page.setdefault(it["page"], []).append(it)
    return [{"page": pg, "title": titles.get(pg) or pg, "items": its} for pg, its in sorted(by_page.items())]


def _brief_prompt(tpl, now, ranked, titles, new_ids, stale_ids, flags, cap):
    """Offer the first MAX_BRIEF_ITEMS of `ranked`, then shrink until the prompt
    fits `cap` estimated tokens: lint flags first, then the lowest-ranked items,
    and last the text of a single remaining item. Returns (messages, size,
    ids_shown); ids_shown is empty when even the fixed text does not fit."""
    shown = [dict(i) for i in ranked[:MAX_BRIEF_ITEMS]]
    flags = list(flags)
    while True:
        ids = {i["id"] for i in shown}
        fl = [f for f in flags if f["item"] in ids][:MAX_BRIEF_FLAGS]
        msgs = prompts.brief_messages(tpl["mission"], now, _groups(shown, titles), new_ids, stale_ids, fl)
        size = sum(_est(m["content"]) for m in msgs)
        over = size - cap
        if over <= 0:
            return msgs, size, {i["id"] for i in shown if i["text"]}
        if fl:
            flags.remove(fl[-1])
        elif len(shown) > 1:
            shown.pop()
        elif shown and shown[0]["text"]:
            shown[0]["text"] = _cut(shown[0]["text"], max(0, _est(shown[0]["text"]) - over))
        else:
            return msgs, size, set()


def _clean_brief(reply, shown, by_id, grounds, ground_all):
    """Keep bullets that point at an item the model was shown, each item once.
    Bullet text naming a person or number the item's verified text does not
    contain is replaced by the item's own text; an ungrounded title or headline
    by a neutral one. Never trusts the reply's shape.
    Returns (headline, sections, replaced_count)."""
    if not isinstance(reply, dict):
        return "", [], 0
    secs = reply.get("sections")
    sections, used, replaced = [], set(), 0
    for sec in (secs if isinstance(secs, list) else [])[:MAX_BRIEF_SECTIONS]:
        if not isinstance(sec, dict):
            continue
        raw_bullets = sec.get("bullets")
        bullets = []
        for b in (raw_bullets if isinstance(raw_bullets, list) else [])[:MAX_BRIEF_BULLETS]:
            iid = b.get("item") if isinstance(b, dict) else None
            if isinstance(iid, str) and iid not in shown:    # "[it-xxxxxxxx] page": the id inside counts
                iid = next((m for m in verify.ITEM_RE.findall(iid[:200]) if m in shown), iid)
            if not isinstance(iid, str) or iid not in shown or iid in used:
                continue
            used.add(iid)
            text = _one_line(b["text"], 300) if isinstance(b.get("text"), str) else ""
            if not text or not _grounded_text(text, grounds[iid]):
                replaced += bool(text)
                text = by_id[iid]["text"]
            bullets.append({"item": iid, "text": text})
        if bullets:
            title = _one_line(sec["title"], 80) if isinstance(sec.get("title"), str) else ""
            title = title if title and _grounded_text(title, ground_all, title=True) else "Notes"
            same = [s for s in sections if s["title"].casefold() == title.casefold()]
            if same:                       # e.g. two ungrounded titles that both became "Notes"
                same[0]["bullets"] += bullets
            else:
                sections.append({"title": title, "bullets": bullets})
    headline = _one_line(reply["headline"], 200) if isinstance(reply.get("headline"), str) else ""
    ok = headline and _grounded_text(headline, ground_all, title=True)
    return (headline if ok else "Your brief"), sections, replaced


def _fallback_brief(groups, new_ids, stale_ids):
    sections = []
    for g in groups[:MAX_BRIEF_SECTIONS]:
        bullets = [{"item": i["id"], "text": (i["text"] or "") + (" (new)" if i["id"] in new_ids else "")
                    + (" (going stale)" if i["id"] in stale_ids else "")}
                   for i in g["items"][:MAX_BRIEF_BULLETS]]
        sections.append({"title": g["title"], "bullets": bullets})
    return "Open items by page (model unavailable, so no summary).", sections


def _render_brief(ember, now, headline, sections, by_id, evidence):
    """Markdown for the brief. All text (model or stored) goes through
    wikifs.inline/_one_line; page names, item ids and raw ids were checked
    against their formats before they reach a link."""
    name = wikifs._one_line(ember.conf.get("name"), 120) or ember.id
    lines, labels = [f"# {name}: {now:%A %d %B %Y}", "", wikifs.inline(headline, 200), ""], {}
    for sec in sections:
        lines += [f"## {wikifs.inline(sec['title'], 80)}", ""]
        for b in sec["bullets"]:
            it = by_id[b["item"]]
            refs = []
            for ev in evidence.get(it["id"], [])[:BRIEF_FOOTNOTES]:
                label = wikifs.footnote_label(labels, ev)        # one label per (raw, quote)
                if label and label not in refs:
                    refs.append(label)
            link = f"([{it['page']}](../pages/{it['page']}.md#^{it['id']}))"
            lines.append(f"- {wikifs.inline(b['text'], verify.MAX_TEXT)} {link} "
                         f"{' '.join(f'[^{r}]' for r in refs)}".rstrip())
        lines.append("")
    lines += wikifs.footnotes(labels, "../raw")
    return "\n".join(lines).rstrip() + "\n"


def _tokens(usage, key):
    try:
        return int((usage if isinstance(usage, dict) else {}).get(key) or 0)
    except (TypeError, ValueError):
        return 0


def brief(ember, llm, now, n_ctx=8192):
    """Write briefs/YYYY-MM-DD.md from open items whose evidence still verifies
    against its raw. Falls back to a plain list when the model fails or its
    reply names no listed item, so a brief always exists (status "partial").
    The run row is always closed: on an unexpected error (e.g. the brief cannot
    be written) it is finished as "failed" and the error propagates; the
    atomic write leaves no half-written brief."""
    st = ember.store
    with st.db:
        st.abort_running("brief", now)     # rows left "running" by a process that died
        run = st.start_run("brief", now)
    errors = []
    try:
        return _brief(ember, llm, now, n_ctx, run, errors)
    except BaseException as e:
        _close_failed(st, run, now, "; ".join(errors + [_err(e)]))
        raise


def _brief(ember, llm, now, n_ctx, run, errors):
    st, tpl, root = ember.store, ember.template, ember.root
    errors += _flush_dirty(ember)          # links must point at pages that are rendered
    # NEW = created since the last brief that reached the reader (ok or partial) before
    # today, so a same-day re-run shows the same NEW set as the day's first brief.
    last = st.last_run("brief", ("ok", "partial"), before=now.replace(hour=0, minute=0, second=0, microsecond=0))
    since = last["started"] if last else ""
    stale_before = ts(now - dt.timedelta(days=tpl["stale_days"]))
    open_verified = st.open_items(verified_only=True)
    candidates = [i for i in open_verified
                  if isinstance(i["id"], str) and verify.ITEM_RE.fullmatch(i["id"])
                  and isinstance(i["page"], str) and verify.PAGE_RE.fullmatch(i["page"])]
    evidence = _checked_evidence(ember, candidates)
    items = [i for i in candidates if i["id"] in evidence]
    dropped = len(open_verified) - len(items)
    by_id = {i["id"]: i for i in items}
    new_ids = {i["id"] for i in items if (i["created"] or "") > since}
    stale_ids = {i["id"] for i in items if (i["updated"] or "") < stale_before}
    titles = {p["page"]: p["title"] for p in st.all_pages()}
    groups = _groups(items, titles)

    status, usage, replaced, router_down = "ok", {}, 0, False
    headline, sections = "Nothing open right now.", []
    if items:
        if n_ctx < MIN_N_CTX:
            errors.append(f"model context n_ctx={n_ctx} is below the minimum of {MIN_N_CTX} tokens; "
                          "load the model with a larger context")
        else:
            ranked = sorted(items, key=lambda i: i["updated"] or "", reverse=True)
            ranked.sort(key=lambda i: (i["id"] not in new_ids, i["id"] not in stale_ids))
            msgs, size, shown = _brief_prompt(tpl, now, ranked, titles, new_ids, stale_ids,
                                              _lint_flags(st), int(n_ctx * PROMPT_SHARE))
            if not shown:
                errors.append("the brief instructions alone do not fit the model context")
            else:
                reply_tokens = max(MIN_REPLY_TOKENS, min(BRIEF_REPLY_TOKENS, n_ctx - size))
                try:                       # any model failure means the plain list, never a crash
                    reply, usage = llm(msgs, prompts.BRIEF_SCHEMA, reply_tokens)
                except Exception as e:
                    errors.append(_err(e))
                    router_down = isinstance(e, RouterUnavailable)
                else:
                    grounds = {iid: _ground(by_id[iid], evidence[iid], now) for iid in shown}
                    ground_all = " ".join(grounds.values())
                    headline, sections, replaced = _clean_brief(reply, shown, by_id, grounds, ground_all)
                    if not sections:
                        errors.append("model reply named no listed item")
        if not sections:
            status = "partial"
            headline, sections = _fallback_brief(groups, new_ids, stale_ids)

    path = wikifs.write_brief(root, f"{now:%Y-%m-%d}",
                              _render_brief(ember, now, headline, sections, by_id, evidence))
    count = sum(len(s["bullets"]) for s in sections)
    summary = f"{count} bullets from {len(items)} open items ({len(new_ids)} new, {len(stale_ids)} stale)"
    if dropped:
        summary += f", {dropped} dropped (evidence no longer verifies)"
    if replaced:
        summary += f", {replaced} restated from the wiki"
    if status == "ok" and _dirty(st):
        status = "partial"                 # a page the brief links to failed to render
    try:
        wikifs.append_log(root, now, "brief", summary)
    except (OSError, ValueError) as e:
        errors.append(f"log: {_err(e)}")
        if status == "ok":
            status = "partial"
    tokens_in, tokens_out = _tokens(usage, "prompt_tokens"), _tokens(usage, "completion_tokens")
    with st.db:
        st.finish_run(run, now, status, "; ".join(errors), tokens_in, tokens_out,
                      {"bullets": count, "items": len(items), "dropped": dropped, "replaced": replaced})
    return {"run": run, "status": status, "path": path, "headline": headline, "summary": summary,
            "bullets": count, "dropped": dropped, "router_down": router_down}


STALE_LINT_DAYS     = 14
MAX_CONTRA_ITEMS    = 30     # open items offered to the contradiction check before the n_ctx cut
MAX_CONTRA_PAIRS    = 10     # contradiction flags kept from one reply
MAX_REPLY_PAIRS     = 50     # reply pairs looked at (the rest are ignored)
CONTRA_REPLY_TOKENS = 800
MAX_LINT_LINES      = 50
MAX_FLAG_WHY        = 240    # "may contradict it-xxxxxxxx: " + 200 characters of model text


def _flag(kind, item, page, why, **extra):
    """A lint flag: every field a one-line string (the brief and the UI read them)."""
    return dict({"kind": kind, "item": _one_line(item), "page": _one_line(page),
                 "why": _one_line(why, MAX_FLAG_WHY)}, **extra)


def _evidence_flags(ember):
    """One flag per evidence row whose quote no longer occurs (word-bounded)
    in its raw, or whose raw cannot be read. Returns (flags, checked), where
    checked is _checked_evidence() for every item."""
    st = ember.store
    checked = _checked_evidence(ember, [{"id": i} for i in sorted(st.known_item_ids())])
    good = {(iid, ev["raw"], ev["quote"]) for iid, evs in checked.items() for ev in evs}
    flags = []
    for ev in st.all_evidence():
        if (ev["item"], ev["raw"], ev["quote"]) in good:
            continue
        raw = ev["raw"] if isinstance(ev["raw"], str) and verify.RAW_RE.fullmatch(ev["raw"]) else "?"
        flags.append(_flag("evidence", ev["item"], ev["page"], f"quote no longer verifies against raw {raw}",
                           raw=raw, quote=_one_line(ev["quote"], verify.MAX_TEXT)))
    return flags, checked


def _paired(pool):
    """Items of `pool` that share their page with at least one other item."""
    count = {}
    for it in pool:
        count[it["page"]] = count.get(it["page"], 0) + 1
    return [it for it in pool if count[it["page"]] > 1]


def _contra_pool(open_items, checked):
    """Verified open items with well-formed ids whose evidence still verifies,
    from pages holding two or more of them, at most MAX_CONTRA_ITEMS."""
    by_page = {}
    for it in open_items:
        if (it["verified"] and it["id"] in checked and isinstance(it["id"], str)
                and verify.ITEM_RE.fullmatch(it["id"])
                and isinstance(it["page"], str) and verify.PAGE_RE.fullmatch(it["page"])):
            by_page.setdefault(it["page"], []).append(it)
    pool = []
    for page in sorted(by_page):
        room = MAX_CONTRA_ITEMS - len(pool)
        if room < 2:
            break
        if len(by_page[page]) > 1:
            pool += by_page[page][:room]
    return pool


def _contra_prompt(pool, cap):
    """Drop the last items until the prompt fits `cap` estimated tokens,
    keeping only items that still have a page-mate. Returns (messages, size,
    items_shown); items_shown is empty when not even one pair fits."""
    pool = list(pool)
    while True:
        pool = _paired(pool)
        if len(pool) < 2:
            return None, 0, []
        msgs = prompts.contra_messages(pool)
        size = sum(_est(m["content"]) for m in msgs)
        if size <= cap:
            return msgs, size, pool
        pool.pop()


def _clean_pairs(reply, shown):
    """Contradiction flags from an untrusted reply: both ids must be items the
    model was shown (shown: {id: item}), distinct, each pair once.
    Returns (flags, error); error is set when the reply has no pairs list."""
    pairs = reply.get("pairs") if isinstance(reply, dict) else None
    if not isinstance(pairs, list):
        return [], "model reply has no pairs list"
    flags, seen = [], set()
    for pr in pairs[:MAX_REPLY_PAIRS]:
        if len(flags) >= MAX_CONTRA_PAIRS:
            break
        if not isinstance(pr, dict):
            continue
        a, b = pr.get("a"), pr.get("b")
        if not (isinstance(a, str) and isinstance(b, str)) or a == b or a not in shown or b not in shown:
            continue
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        why = _one_line(pr["why"], 200) if isinstance(pr.get("why"), str) else ""
        flags.append(_flag("contradiction", a, shown[a]["page"], f"may contradict {b}" + (f": {why}" if why else "")))
    return flags, ""


def _lint_md(now, flags):
    """The index's ember:lint region. Page links and item ids only when they
    match their formats; every text goes through wikifs.inline; broken
    evidence is cited with a footnote to its raw (wikifs.footnote_label)."""
    lines, labels = [f"## Lint ({now:%Y-%m-%d})", ""], {}
    if not flags:
        return "\n".join(lines + ["All clear."])
    for f in flags[:MAX_LINT_LINES]:
        page, item = f["page"], f["item"]
        target = f"[{page}](pages/{page}.md)" if _valid_page(page) else (wikifs.inline(page, 80) or "?")
        if verify.ITEM_RE.fullmatch(item):
            target += f" `{item}`"
        line = f"- **{f['kind']}** {target}: {wikifs.inline(f['why'], MAX_FLAG_WHY)}"
        if f["kind"] == "evidence":
            label = wikifs.footnote_label(labels, {"raw": f.get("raw"), "quote": f.get("quote")})
            if label:
                line += f" [^{label}]"
        lines.append(line)
    if len(flags) > MAX_LINT_LINES:
        lines.append(f"- … and {len(flags) - MAX_LINT_LINES} more")
    if labels:
        lines += [""] + wikifs.footnotes(labels, "raw")
    return "\n".join(lines)


def lint(ember, llm, now, n_ctx=8192):
    """Mark problems: quotes that no longer verify, stale open items, orphan
    pages, index drift, and (one model call) contradicting items on a page.
    Flags go to meta "lint_flags" and the index's ember:lint region. Never
    closes, deletes or rewrites an item, a page or a raw.
    The run row is always closed: on an unexpected error it is finished as
    "failed" and the error propagates."""
    st = ember.store
    with st.db:
        st.abort_running("lint", now)      # rows left "running" by a process that died
        run = st.start_run("lint", now)
    errors = []
    try:
        return _lint(ember, llm, now, n_ctx, run, errors)
    except BaseException as e:
        _close_failed(st, run, now, "; ".join(errors + [_err(e)]))
        raise


def _lint(ember, llm, now, n_ctx, run, errors):
    st, root = ember.store, ember.root
    errors += _flush_dirty(ember)          # a page awaiting its render is not drift
    flags, checked = _evidence_flags(ember)
    open_items = st.open_items()
    cutoff = ts(now - dt.timedelta(days=STALE_LINT_DAYS))
    for it in open_items:
        if str(it["updated"] or "") < cutoff:
            flags.append(_flag("stale", it["id"], it["page"], f"open with no new evidence for {STALE_LINT_DAYS}+ days"))
    pages = st.all_pages()
    invalid = sorted({p["page"] for p in pages if not _valid_page(p["page"])}, key=str)
    for p in pages:
        if p["page"] not in invalid and not st.items_for_page(p["page"]):
            flags.append(_flag("orphan", "", p["page"], "page has no items"))
    on_disk = set(wikifs.list_page_files(root))
    in_db = {p["page"] for p in pages} - set(invalid)
    for page in sorted(on_disk - in_db):
        flags.append(_flag("drift", "", page, "page file is not in the ember's index"))
    for page in sorted(in_db - on_disk):
        flags.append(_flag("drift", "", page, "indexed page has no file"))
    for page in invalid:                   # only possible by editing the db; never rendered or linked
        flags.append(_flag("drift", "", page, "indexed page has an invalid name"))

    status, usage, router_down = "ok", {}, False
    pool = _contra_pool(open_items, checked)
    if len(_paired(pool)) > 1:
        if n_ctx < MIN_N_CTX:
            errors.append(f"model context n_ctx={n_ctx} is below the minimum of {MIN_N_CTX} tokens; "
                          "load the model with a larger context")
            status = "partial"
        else:
            msgs, size, shown = _contra_prompt(pool, int(n_ctx * PROMPT_SHARE))
            if not shown:
                errors.append("the contradiction check does not fit the model context")
                status = "partial"
            else:
                reply_tokens = max(MIN_REPLY_TOKENS, min(CONTRA_REPLY_TOKENS, n_ctx - size))
                try:                       # any model failure only skips the contradiction check
                    reply, usage = llm(msgs, prompts.CONTRA_SCHEMA, reply_tokens)
                except Exception as e:
                    errors.append(_err(e))
                    status = "partial"
                    router_down = isinstance(e, RouterUnavailable)
                else:
                    found, bad = _clean_pairs(reply, {i["id"]: i for i in shown})
                    flags += found
                    if bad:
                        errors.append(bad)
                        status = "partial"

    _write_index(ember, _lint_md(now, flags))
    counts = {}
    for f in flags:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "all clear"
    if _dirty(st):
        status = "partial"                 # a page failed to render, so drift may be overstated
    try:
        wikifs.append_log(root, now, "lint", summary)
    except (OSError, ValueError) as e:
        errors.append(f"log: {_err(e)}")
        status = "partial"
    # The flags the brief reads and the run close together. If sqlite fails here the index is
    # ahead of meta lint_flags; the run is closed as failed and the next lint reconciles both.
    with st.db:
        st.set_meta(LINT_FLAGS_KEY, flags)
        st.finish_run(run, now, status, "; ".join(errors), _tokens(usage, "prompt_tokens"),
                      _tokens(usage, "completion_tokens"), counts)
    return {"run": run, "status": status, "flags": flags, "summary": summary, "router_down": router_down}
