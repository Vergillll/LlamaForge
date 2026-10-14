"""Evidence checks: the reliability core of Embers.

The model proposes wiki updates; code decides what is believed. A claim counts
as verified only if it carries a verbatim quote that really occurs in an
immutable raw snapshot. Owners and due dates must appear inside such a quote,
so the model cannot invent who owes what or by when.
"""
import datetime as dt, re, unicodedata

from . import reserved_name

MIN_QUOTE    = 12
MAX_TEXT     = 600
MAX_OPS      = 40
MAX_EVIDENCE = 5
OPS          = ("add", "update", "close")
ITEM_KINDS   = ("loop", "fact", "person", "event")
PAGE_RE = re.compile(r"[a-z0-9-]{1,40}/[a-z0-9-]{1,60}")   # <kind>/<slug>
ITEM_RE = re.compile(r"it-[0-9a-f]{8}")                     # also a valid Obsidian block id
RAW_RE  = re.compile(r"[0-9a-f]{12}")
DONE_WORK = "a loop opened on finished work alone"

_QUOTES = dict.fromkeys(map(ord, "'\"`\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u00ab\u00bb"
                         "\u00ad"), None)   # and the soft hyphen, invisible in PDF copies
_DASHES = dict.fromkeys(map(ord, "-\u2010\u2012\u2013\u2014\u2015"), " ")
HYPHEN  = "\u2011"                       # a hyphen inside a word stays part of it
MINUS   = "\u2212"                       # kept only in front of a digit
_MINUS  = re.compile(r"[-" + MINUS + r"](?=\d)")
_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")
_WORD   = r"\w" + MINUS + HYPHEN                      # characters that glue a match to its neighbours
_BREAK  = re.compile(r"(?<=\w)[-‐­][ \t]*\r?\n\s*(?=\w)")   # verifi-<newline>cation


def _contains(hay, needle):
    """Whole-word containment on normalised text (a minus sign counts as part of a word)."""
    return re.search(r"(?<![" + _WORD + r"])" + re.escape(needle) + r"(?![" + _WORD + r"])", hay) is not None


def normalise(s):
    """NFKC, lowercase, quotes removed, dashes to spaces, whitespace collapsed."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.translate(_QUOTES)
    s = _MINUS.sub(MINUS, s)             # a minus in front of a digit is significant
    s = _HYPHEN.sub(HYPHEN, s)           # so is the hyphen of non-refundable
    s = s.translate(_DASHES)
    return " ".join(s.split())


def raw_forms(raw_text):
    """(normalised raw, loose form or None). A raw that breaks words across
    lines with a hyphen (PDF copies) also gets a loose form with the breaks
    joined and in-word hyphens dropped: verifi-/cation and state-of-the-/art
    can't be told apart without a dictionary, and one quote may cross both."""
    raw_text = raw_text or ""
    loose = None
    if _BREAK.search(raw_text):
        loose = normalise(_BREAK.sub("", raw_text)).replace(HYPHEN, "")
    return normalise(raw_text), loose


def quote_in(forms, quote):
    plain, loose = forms
    q = normalise(quote)
    if len(q) < MIN_QUOTE:
        return False
    return _contains(plain, q) or (loose is not None and _contains(loose, q.replace(HYPHEN, "")))


def check_quote(quote, raw_text):
    return quote_in(raw_forms(raw_text), quote)


def page_ok(page, kinds):
    if not (isinstance(page, str) and PAGE_RE.fullmatch(page)):
        return False
    kind, slug = page.split("/")
    return kind in kinds and not reserved_name(kind) and not reserved_name(slug)


def _one_line(s, cap):
    return " ".join(s.split())[:cap] if isinstance(s, str) else ""


def _grounded(value, quotes):
    v = normalise(value)
    return len(v) >= 2 and any(_contains(normalise(q), v) for q in quotes)


_MONTHS = {m: i for i, names in enumerate((
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",),
    ("june", "jun"), ("july", "jul"), ("august", "aug"), ("september", "sept", "sep"),
    ("october", "oct"), ("november", "nov"), ("december", "dec")), 1) for m in names}
_MON  = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?"
_DAY  = r"(\d{1,2})(?:st|nd|rd|th)?(?!\w)"
_YEAR = r"(?:,?\s+(\d{4})(?!\w))?"
_D    = "[-‐‑‒–−]"
_DATE_RES = (
    ("ymd", re.compile(r"(?<!\w)(\d{4})" + _D + r"(\d{2})" + _D + r"(\d{2})(?=t\d|\W|$)")),  # 2026-10-09
    ("ymd", re.compile(r"(?<!\w)(\d{4})(\d{2})(\d{2})(?=t\d|\W|$)")),          # ICS 20261009T150000Z
    ("mdy", re.compile(r"(?<!\w)" + _MON + r"\s+" + _DAY + _YEAR)),            # Oct 9th, 2026
    ("dmy", re.compile(r"(?<!\w)" + _DAY + r"\s+(?:of\s+)?" + _MON + _YEAR)),  # 9 October 2026
)


def _near_year(month, day, ref):
    """A month/day without a year: the occurrence nearest the reference date."""
    best = None
    for y in (ref.year - 1, ref.year, ref.year + 1):
        try:
            d = dt.date(y, month, day)
        except ValueError:
            continue
        if best is None or abs((d - ref).days) < abs((best - ref).days):
            best = d
    return best


def _dates(text, ref):
    """Date mentions in text, in order. Numeric slash dates (10/09) are
    ambiguous between countries, so they never count."""
    s = unicodedata.normalize("NFKC", text or "")[:2000].lower()
    found = []
    for form, rx in _DATE_RES:
        for m in rx.finditer(s):
            try:
                if form == "ymd":
                    d = dt.date(int(m[1]), int(m[2]), int(m[3]))
                else:
                    mon, day = (m[1], m[2]) if form == "mdy" else (m[2], m[1])
                    d = (dt.date(int(m[3]), _MONTHS[mon], int(day)) if m[3]
                         else _near_year(_MONTHS[mon], int(day), ref))
            except ValueError:
                continue
            if d is not None and 1900 <= d.year <= 2200:
                found.append((m.start(), d))
    return [d for _, d in sorted(found, key=lambda t: t[0])]


def due_dates(text, ref_date):
    return set(_dates(text, ref_date))


def ground_due(due, quotes, ref_date):
    """A due date the evidence supports, else "". A date in any wording
    (2026-10-09, Oct 9th, 9 October) that a passing quote also names, in any
    wording, comes back as YYYY-MM-DD; a non-date ("Friday") must appear in a
    quote word for word and is kept as written."""
    if not isinstance(due, str):
        return ""
    ref_date = ref_date or dt.date.today()
    named = set().union(*(due_dates(q, ref_date) for q in quotes)) if quotes else set()
    for d in _dates(due, ref_date):
        if d in named:
            return d.isoformat()
    return _one_line(due, 40) if _grounded(due, quotes) else ""


def cited_raw(sha, quote, raws):
    """The raw id a quote verifiably comes from, else None. Small models garble
    the id (the whole "=== raw <id> (...) ===" header, a source's first line),
    so: the cited id, else an id found inside the cited text, else any raw of
    the batch that contains the quote. The quote itself must always be verbatim."""
    if not isinstance(quote, str):
        return None
    tried = []
    if isinstance(sha, str):
        tried = [sha] + [s for s in RAW_RE.findall(sha[:400]) if s != sha]
    for s in tried + [s for s in raws if s not in tried]:
        if s in raws and check_quote(quote, raws[s]):
            return s
    return None


def _from_done_work(evidence, evs, text, raws, done_raws):
    """True when finished work is all a new loop rests on. With passing quotes,
    every quoted raw is done work. Without them (a small model garbles every
    quote), every raw the op names is done work, or it names none and its text
    is lifted from a done raw."""
    if not done_raws:
        return False
    if evidence:
        return all(e["raw"] in done_raws for e in evidence)
    named = set()
    for ev in evs[:MAX_EVIDENCE]:
        sha = ev.get("raw") if isinstance(ev, dict) else None
        if isinstance(sha, str):
            named.update(s for s in [sha] + RAW_RE.findall(sha[:400]) if s in raws)
    if named:
        return named <= set(done_raws)
    return bool(text) and any(s in raws and check_quote(text, raws[s]) for s in done_raws)


def verify_op(op, raws, kinds, known_items=(), ref_date=None, done_raws=()):
    """Check one proposed op against raws ({sha12: text}). ref_date (the run's
    date) places a month/day without a year. done_raws are raws that record
    finished work (commits): they can close a loop but never open one alone.
    Returns (clean_op, None) or (None, reason)."""
    if not isinstance(op, dict):
        return None, "not an object"
    action = op.get("op")
    if action not in OPS:
        return None, f"unknown op {action!r}"
    page = op.get("page")
    if not page_ok(page, kinds):
        return None, f"bad page {page!r}"
    item = op.get("item")
    text = _one_line(op.get("text"), MAX_TEXT)
    known = isinstance(item, str) and ITEM_RE.fullmatch(item) and item in known_items
    if action == "update" and not known and text and not (isinstance(item, str) and ITEM_RE.search(item)):
        action = "add"                  # no item id at all (a page path, empty): the item is new
    if action == "add":
        item = None                     # code assigns ids; the model never does
    elif not known:
        return None, f"unknown item {item!r}"
    kind = op.get("kind") or "loop"
    if kind not in ITEM_KINDS:
        return None, f"bad kind {kind!r}"
    if action == "add" and not text:
        return None, "add without text"
    evs = op.get("evidence") if isinstance(op.get("evidence"), list) else []
    evidence = []
    for ev in evs[:MAX_EVIDENCE]:
        if not isinstance(ev, dict):
            continue
        quote = ev.get("quote")
        sha = cited_raw(ev.get("raw"), quote, raws)
        if sha is not None:
            evidence.append({"raw": sha, "quote": _one_line(quote, MAX_TEXT)})
    if action == "close" and not evidence:
        return None, "close without passing evidence"
    if action == "add" and kind == "loop" and _from_done_work(evidence, evs, text, raws, done_raws):
        return None, DONE_WORK
    quotes = [e["quote"] for e in evidence]
    owner, due = op.get("owner"), op.get("due")
    return {"op": action, "page": page, "kind": kind, "item": item, "text": text,
            "owner": _one_line(owner, 80) if isinstance(owner, str) and _grounded(owner, quotes) else "",
            "due": ground_due(due, quotes, ref_date),
            "evidence": evidence, "verified": bool(evidence)}, None


def verify_batch(update, raws, kinds, known_items=(), ref_date=None, done_raws=()):
    """Check a whole model reply. Returns (ops, new_pages, rejected), where
    rejected is [(op_index, reason)]."""
    if not isinstance(update, dict):
        return [], [], [(-1, "update is not an object")]
    ops, rejected = [], []
    raw_ops = update.get("ops") if isinstance(update.get("ops"), list) else []
    for i, op in enumerate(raw_ops):
        if i >= MAX_OPS:
            rejected.append((i, "over the op cap"))
            continue
        clean, why = verify_op(op, raws, kinds, known_items, ref_date, done_raws)
        if clean:
            ops.append(clean)
        else:
            rejected.append((i, why))
    pages = []
    raw_pages = update.get("new_pages") if isinstance(update.get("new_pages"), list) else []
    for p in raw_pages[:MAX_OPS]:
        if isinstance(p, dict) and page_ok(p.get("page"), kinds):
            pages.append({"page": p["page"],
                          "title": _one_line(p.get("title"), 120) or p["page"],
                          "summary": _one_line(p.get("summary"), MAX_TEXT)})
    return ops, pages, rejected
