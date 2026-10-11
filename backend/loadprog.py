"""Real load progress for models the llama.cpp router is loading.

The router does not put progress in GET /models; it streams it on /models/sse as
`status_change` events whose `progress` is {stages, current, value}, straight
from llama's own load callback (throttled to one sample per 200 ms). This
module listens to that stream - only while something is loading - and keeps the
latest percent per model so /api/state can show a bar instead of a stopwatch.

A router without /models/sse (older builds) answers 404: we note it and stop
asking for a while; the UI falls back to the elapsed-seconds display.
"""
import json
import threading
import time
import urllib.error
import urllib.request

UNSUPPORTED_RETRY_S = 600   # re-check a router that had no /models/sse
STAGE_ORDER = ("text_model", "spec_model", "mmproj_model")


class LoadProgress:
    def __init__(self, target, opener=urllib.request.urlopen, idle_s=30, retry_s=2):
        """target() -> (url, headers). idle_s: how long the watcher keeps
        listening after the last ensure() (i.e. the last poll that saw a load)."""
        self._target, self._open = target, opener
        self._idle_s, self._retry_s = idle_s, retry_s
        self._lock = threading.Lock()
        self._p = {}                 # model -> {"pct", "stage", "stages"}
        self._want_until = 0.0
        self._unsupported_at = None
        self._running = False

    # ---------- state ----------
    def note(self, obj):
        """Fold one SSE event into the state. Anything unexpected is ignored."""
        if not isinstance(obj, dict) or obj.get("event") != "status_change":
            return
        mid, data = obj.get("model"), obj.get("data")
        if not isinstance(mid, str) or not isinstance(data, dict):
            return
        with self._lock:
            if data.get("status") != "loading":
                self._p.pop(mid, None)
                return
            prog = data.get("progress")
            if not isinstance(prog, dict):
                return
            prev = self._p.get(mid) or {}
            stages = prog.get("stages") if isinstance(prog.get("stages"), list) else prev.get("stages")
            stage = prog.get("current") or prog.get("stage")
            if not isinstance(stage, str):
                return
            value = prog.get("value", 0.0)
            if not isinstance(value, (int, float)) or value != value:
                return
            stages = [s for s in (stages or [stage]) if isinstance(s, str)] or [stage]
            if stage not in stages:
                stages = sorted(set(stages) | {stage}, key=lambda s: (
                    STAGE_ORDER.index(s) if s in STAGE_ORDER else len(STAGE_ORDER), s))
            frac = (stages.index(stage) + min(max(float(value), 0.0), 1.0)) / len(stages)
            pct = int(round(frac * 100))
            self._p[mid] = {"pct": max(pct, prev.get("pct", 0)), "stage": stage, "stages": stages}

    def get(self, mid):
        with self._lock:
            p = self._p.get(mid)
            return {"pct": p["pct"], "stage": p["stage"]} if p else None

    def attach(self, rows):
        """Add load_progress to router-loaded rows that are loading, and keep
        the watcher listening while any are. Rows in their own process (pinned
        to another build) are not the router's, so they get nothing."""
        loading = [m for m in rows if m.get("status") == "loading" and not m.get("process")]
        if not loading:
            return
        self.ensure()
        for m in loading:
            p = self.get(m.get("id"))
            if p:
                m["load_progress"] = p

    def unsupported(self):
        at = self._unsupported_at
        return at is not None and time.time() - at < UNSUPPORTED_RETRY_S

    # ---------- the watcher ----------
    def ensure(self):
        if self.unsupported():
            return
        with self._lock:
            self._want_until = time.time() + self._idle_s
            if self._running:
                return
            self._running = True
        threading.Thread(target=self._loop, daemon=True, name="load-progress").start()

    def _wanted(self):
        return time.time() < self._want_until

    def _loop(self):
        try:
            while self._wanted() and not self.unsupported():
                self._run_once()
                if self._wanted():
                    time.sleep(self._retry_s)
        finally:
            with self._lock:
                self._running = False
                self._p.clear()      # nobody is loading any more: drop stale bars

    def _run_once(self):
        """One connection: read events until the stream ends, errors, or the
        read times out (then the loop reconnects if a load is still on)."""
        try:
            url, headers = self._target()
            req = urllib.request.Request(url, headers=dict(headers or {}, Accept="text/event-stream"))
            r = self._open(req, timeout=10)
        except urllib.error.HTTPError as e:
            if e.code in (404, 405, 501):
                self._unsupported_at = time.time()
            return
        except Exception:
            return
        try:
            self.consume(r)
        except Exception:
            pass
        finally:
            try:
                r.close()
            except Exception:
                pass

    def consume(self, stream):
        for raw in stream:
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            if not line.startswith("data:"):
                continue
            try:
                self.note(json.loads(line[5:].strip()))
            except (ValueError, TypeError):
                continue
            if not self._wanted() and self._want_until:
                return
