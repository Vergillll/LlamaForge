"""Elastic VRAM: when other apps need the GPU, move a loaded model's layers to
the CPU instead of unloading it, keep its conversation (slotkeep.py), and move
them back once the memory is free again.

The planning is pure (ladder, footprints, choose, plan_keys). The controller is
shaped like gamemode.GameMode: a 5 s tick, debounced, with every side effect
behind an injected io object so the tests drive it with a fake."""
import collections
import threading
import time

from footprint import _ALIASES, _flag, _norm

SHRINK_HITS = 2            # ticks under half the headroom before a shrink
GROW_HITS = 6              # ticks (30 s) with room to spare before a grow
GROW_MARGIN_MIB = 1024     # a grow must fit with this much on top of the headroom
COOLDOWN_S = 120           # since the last change, before a grow
COOLDOWN_MAX_S = 1800
THRASH_WINDOW_S = 300      # a shrink this soon after a grow doubles the cooldown
CALM_RESET_S = 3600        # an hour without changes resets it
STILL_BUSY = 0.8           # no grow while others hold >= 80 % of what they had at the shrink


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def ladder(settings, layout):
    """The steps a model can take, from the user's own setting (index 0, no
    keys of ours) down to nothing on the GPU. Each step is {"keys": what we
    write to models.ini, "gpu_layers", "cpu_moe", "layers"}. Never above what
    the user chose. MoE models move experts first (25 % of the blocks at a
    time), which costs far less speed than whole layers. [] without a block
    count."""
    kv = (layout or {}).get("kv") or {}
    n = _int(kv.get("block_count")) or 0
    if n <= 0:
        return []
    s = _norm(settings)
    user_ngl = _int(s.get("n-gpu-layers"))
    top = n if user_ngl is None or user_ngl < 0 else min(user_ngl, n)
    moe = (_int(kv.get("expert_count")) or 0) > 0 and not _flag(s, "cpu-moe")
    cpu_moe = (_int(s.get("n-cpu-moe")) or 0) if moe else 0
    steps = [{"keys": {}, "gpu_layers": top, "cpu_moe": cpu_moe, "layers": n}]
    if moe:
        for pct in (25, 50, 75, 100):
            c = n * pct // 100
            if c > steps[-1]["cpu_moe"]:
                steps.append({"keys": {"n-cpu-moe": str(c)}, "gpu_layers": top,
                              "cpu_moe": c, "layers": n})
    base = dict(steps[-1]["keys"])
    for pct in (75, 50, 25, 0):
        g = n * pct // 100
        if g < steps[-1]["gpu_layers"]:
            steps.append({"keys": {**base, "n-gpu-layers": str(g)}, "gpu_layers": g,
                          "cpu_moe": steps[-1]["cpu_moe"], "layers": n})
    return steps


def label(step):
    t = f"{step['gpu_layers']}/{step['layers']}"
    return t + (f", experts of {step['cpu_moe']} on CPU" if step["cpu_moe"] else "")


def footprints(measured, predicted, gpus):
    """Each step's {gpu: MiB}. measured[i]: footprint.Store's figure for step i,
    or None; predicted[i]: footprint.predict's need_mib. A step nobody measured
    is scaled from one that was (fewer layers shrink every GPU alike); with
    nothing measured the prediction is split evenly over `gpus`, the way
    llama.cpp spreads layers by default."""
    def pred(i):
        return (predicted[i] if i < len(predicted) else 0) or 0
    ref = next((i for i, m in enumerate(measured) if m), None)
    gs = list(gpus) or [0]
    out = []
    for i, m in enumerate(measured):
        if m:
            out.append(dict(m))
        elif ref is not None and pred(ref):
            k = pred(i) / pred(ref)
            out.append({g: int(v * k) for g, v in measured[ref].items()})
        else:
            out.append({g: int(pred(i)) // len(gs) for g in gs})
    return out


def choose(fps, budget, lo=0):
    """Index of the largest step, from lo down, that fits budget ({gpu: MiB})
    on every GPU it uses; None when even the smallest doesn't."""
    for i in range(lo, len(fps)):
        if all(v <= budget.get(g, 0) for g, v in fps[i].items()):
            return i
    return None


def plan_keys(section, rec, keys):
    """What to write into a model's models.ini section to move it to a step
    with `keys`, and the new elastic_placed record (None once nothing of ours
    is left). rec: {"keys": what we wrote, "was": the user's value before we
    touched it, None for unset}. A key whose value is no longer what we wrote
    was edited by the user since: it's theirs, so we neither overwrite it nor
    restore it later - the slotctl `placed` rule."""
    rec = rec or {"keys": {}, "was": {}}
    spelt = {}                           # canonical name -> the spelling the section uses
    for sk in section:
        n = _ALIASES.get(str(sk).lower(), str(sk).lower())
        if n not in spelt or sk == n:
            spelt[n] = sk
    keys = {spelt.get(k, k): v for k, v in keys.items()}
    edited = {k for k, v in rec["keys"].items() if section.get(k) != v}
    ours = {k: v for k, v in rec["keys"].items() if k not in edited}
    was = {k: rec["was"].get(k) for k in ours}
    writes = {}
    for k, v in keys.items():
        if k in edited:
            continue
        if k not in ours:
            was[k] = section.get(k)
        writes[k] = v
        ours[k] = v
    for k in [k for k in ours if k not in keys]:
        writes[k] = was.pop(k)
        del ours[k]
    writes = {k: v for k, v in writes.items() if section.get(k) != v}
    return writes, ({"keys": ours, "was": was} if ours else None)


def _sum(fps):
    out = {}
    for fp in fps:
        for g, v in fp.items():
            out[g] = out.get(g, 0) + v
    return out


class Elastic:
    def __init__(self, cfg, io, clock=time.time):
        self._cfg, self._io, self._clock = cfg, io, clock
        self._run = threading.Lock()     # one tick or swap at a time; a swap can take 20 s
        self._lock = threading.Lock()    # what status() reads; never held across a swap
        self._enabled = self._standby = False
        self._swapping = None
        self._models, self._others, self._waiting = [], {}, []
        self._shrink_hits = self._grow_hits = 0
        self._last_change = self._last_grow = None
        self._cooldown = COOLDOWN_S
        self._order = []                 # shrunk models, oldest first: grow back newest first
        self._others_at = {}             # mid -> (MiB other apps used when it shrank, calm baseline)
        self._calm_others = None         # MiB other apps used on the last tick without pressure
        self._failed_at = {}             # mid -> when a move of it last failed outright
        self._events = collections.deque(maxlen=20)
        self._thread = None

    # -- one look ---------------------------------------------------------
    def tick(self):
        with self._run:
            c = self._cfg()
            enabled = bool(c.get("elastic_vram"))
            standby = enabled and bool(self._io.game_active())
            with self._lock:
                self._enabled, self._standby = enabled, standby
            self._io.tidy()
            if not enabled:
                self._shrink_hits = self._grow_hits = 0
                snap = self._io.snapshot()
                if not snap:
                    return
                head = int(c.get("elastic_headroom_mib", 1536))
                models = [m for m in self._io.models(list(snap)) if m["steps"]]
                others = self._others_of(snap, models)
                self._forgive(models)
                waiting = [m["id"] for m in models if m["at"] > 0]
                with self._lock:
                    self._models, self._others, self._waiting = models, others, waiting
                for m in models:         # turned off while shrunk: one model per tick, only
                    if m["at"] > 0 and choose(           # when its own step fits again
                            m["fps"], self._budget(snap, others, models, m, head), 0) == 0:
                        ev = self._move(c, m, 0, "Elastic VRAM was turned off")
                        if ev["ok"] and ev["to"] == label(m["steps"][0]):
                            with self._lock:
                                self._waiting = [w for w in waiting if w != m["id"]]
                        break
                return
            if standby:                  # Game mode unloads everything anyway
                self._shrink_hits = self._grow_hits = 0
                return
            snap = self._io.snapshot()
            if not snap:
                return
            models = [m for m in self._io.models(list(snap)) if m["steps"]]
            others = self._others_of(snap, models)
            self._forgive(models)
            with self._lock:
                self._models, self._others, self._waiting = models, others, []
            now = self._clock()
            if self._last_change is not None and now - self._last_change >= CALM_RESET_S:
                self._cooldown = COOLDOWN_S
            head = int(c.get("elastic_headroom_mib", 1536))
            if any(s["total"] - s["used"] < head / 2 for s in snap.values()):
                self._grow_hits = 0
                self._shrink_hits += 1
                if self._shrink_hits >= SHRINK_HITS:
                    self._shrink_hits = 0
                    self._shrink(c, snap, others, models, head)
                return
            self._shrink_hits = 0
            self._calm_others = sum(others.values())
            pick = self._grow_pick(snap, others, models, head) if c.get("elastic_grow", True) else None
            if pick is None:
                self._grow_hits = 0
                return
            self._grow_hits += 1
            if self._grow_hits >= GROW_HITS and (
                    self._last_change is None or now - self._last_change >= self._cooldown):
                self._grow_hits = 0
                self._move(c, pick[0], pick[1], "other apps freed VRAM")

    def _forgive(self, models):
        """A model that is loaded at its own settings again (the user reloaded
        it after a give-up) may be shrunk again at once."""
        for m in models:
            if m["at"] == 0:
                self._failed_at.pop(m["id"], None)

    def safe_tick(self):
        try:
            self.tick()
        except Exception:
            pass

    @staticmethod
    def _others_of(snap, models):
        ours = _sum(m["fps"][m["at"]] for m in models)
        return {g: max(0, s["used"] - ours.get(g, 0)) for g, s in snap.items()}

    @staticmethod
    def _budget(snap, others, models, m, head):
        """What m may use on each GPU: the total, less other apps, the
        headroom and the other models where they are now."""
        rest = _sum(o["fps"][o["at"]] for o in models if o["id"] != m["id"])
        return {g: s["total"] - others.get(g, 0) - head - rest.get(g, 0) for g, s in snap.items()}

    def _shrink(self, c, snap, others, models, head):
        best = None
        for m in models:
            last = len(m["steps"]) - 1
            if m["at"] >= last:
                continue
            f = self._failed_at.get(m["id"])
            if f is not None and self._clock() - f < COOLDOWN_S:
                continue
            j = choose(m["fps"], self._budget(snap, others, models, m, head), m["at"] + 1)
            j = last if j is None else j
            freed = sum(m["fps"][m["at"]].values()) - sum(m["fps"][j].values())
            if freed > 0 and (best is None or freed > best[2]):
                best = (m, j, freed)
        if best is None:
            return
        m, j, _ = best
        if self._last_grow is not None and self._clock() - self._last_grow < THRASH_WINDOW_S:
            self._cooldown = min(self._cooldown * 2, COOLDOWN_MAX_S)
        total = sum(others.values())
        self._others_at[m["id"]] = (total, self._calm_others or 0)
        if m["id"] not in self._order:
            self._order.append(m["id"])
        self._move(c, m, j, f"other apps use {total / 1024:.1f} GB")

    def _grow_pick(self, snap, others, models, head):
        by_id = {m["id"]: m for m in models}
        total = sum(others.values())
        for mid in reversed(self._order):
            m = by_id.get(mid)
            if not m or m["at"] == 0:
                continue
            at_shrink, base = self._others_at.get(mid, (float("inf"), 0))
            if at_shrink > base and total - base >= STILL_BUSY * (at_shrink - base):
                continue
            b = self._budget(snap, others, models, m, head + GROW_MARGIN_MIB)
            j = choose(m["fps"], b, 0)
            if j is not None and j < m["at"]:
                return m, j
        for m in models:                 # shrunk before a panel restart: no history
            if m["at"] > 0 and m["id"] not in self._order:
                j = choose(m["fps"], self._budget(snap, others, models, m, head + GROW_MARGIN_MIB), 0)
                if j is not None and j < m["at"]:
                    return m, j
        return None

    # -- the swap ---------------------------------------------------------
    def _move(self, c, m, j, reason):
        """Save the conversation, move m to step j (walking down the rest of the
        ladder when a load fails), restore. Gives up - user keys back, model
        unloaded - only when every step fails. Every io call may
        raise; that becomes a note, never an escape."""
        mid, frm, steps = m["id"], m["at"], m["steps"]
        with self._lock:
            self._swapping = mid
        t0, notes, at, handle, kept = self._clock(), [], None, None, None
        try:
            if c.get("elastic_keep_conversation", True):
                try:
                    handle = self._io.save(mid)
                    if handle and handle.get("note"):
                        notes.append(handle["note"])
                except Exception as e:
                    handle = None
                    notes.append(f"conversation not kept: {type(e).__name__}: {e}")
            for i in range(j, len(steps)):
                try:
                    err = self._io.swap(mid, steps[i])
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                if not err:
                    at = i
                    break
                notes.append(f"{label(steps[i])} didn't load: {err}")
            restored = False
            if at is None:
                try:
                    self._io.giveup(mid)
                except Exception as e:
                    notes.append(f"giving up {mid} failed: {type(e).__name__}: {e}")
                notes.append(f"Elastic VRAM couldn't fit {mid}; it stays unloaded until you load it")
            elif handle and handle.get("files"):
                try:
                    kept = bool(self._io.restore(mid, handle))   # drops its own files
                    restored = True
                except Exception as e:
                    kept = False
                    notes.append(f"conversation not restored: {type(e).__name__}: {e}")
            if handle and not restored and (at is None or handle.get("files")):
                try:
                    self._io.drop(handle)
                except Exception as e:
                    notes.append(f"dropping saved slots failed: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._swapping = None
            now = self._clock()
            self._last_change = now
        if at is None:
            self._failed_at[mid] = now
        if at is not None and at < frm:
            self._last_grow = now
        elif j < frm:                    # a grow that ended where it began, or lower
            self._cooldown = min(self._cooldown * 2, COOLDOWN_MAX_S)
        if at in (0, None):
            if mid in self._order:
                self._order.remove(mid)
            self._others_at.pop(mid, None)
        return self._log(mid, label(steps[frm]), "unloaded" if at is None else label(steps[at]),
                         reason, now - t0, at is not None, kept, "; ".join(notes))

    def _log(self, mid, frm, to, reason, secs, ok, kept, note):
        ev = {"t": self._clock(), "model": mid, "from": frm, "to": to, "reason": reason,
              "secs": round(secs, 1), "ok": ok, "kept": kept, "note": note}
        with self._lock:
            self._events.appendleft(ev)
        return ev

    def grow_now(self, mid):
        """The dashboard's "Grow back now": skip the wait and the cooldown, not
        the fit (no extra margin either)."""
        with self._run:
            c = self._cfg()
            snap = self._io.snapshot() or {}
            models = [m for m in self._io.models(list(snap)) if m["steps"]]
            m = next((x for x in models if x["id"] == mid), None)
            if not m or m["at"] == 0:
                return {"ok": False, "error": f"{mid} isn't resized"}
            others = self._others_of(snap, models)
            head = int(c.get("elastic_headroom_mib", 1536))
            j = choose(m["fps"], self._budget(snap, others, models, m, head), 0)
            if j is None or j >= m["at"]:
                note = "other apps still use the VRAM it needs"
                here = label(m["steps"][m["at"]])
                self._log(mid, here, here, "you asked", 0, False, None, note)
                return {"ok": False, "error": note}
            ev = self._move(c, m, j, "you asked")
            return {"ok": ev["ok"], "to": ev["to"]}

    def safe_grow(self, mid):
        try:
            self.grow_now(mid)
        except Exception:
            pass

    def status(self):
        with self._lock:
            return {
                "enabled": self._enabled, "standby": self._standby, "swapping": self._swapping,
                "cooldown_s": self._cooldown, "waiting": list(self._waiting), "others_mib": dict(self._others),
                "models": [{"id": m["id"], "at": m["at"], "label": label(m["steps"][m["at"]]),
                            "gpu_layers": m["steps"][m["at"]]["gpu_layers"],
                            "layers": m["steps"][m["at"]]["layers"]} for m in self._models],
                "events": list(self._events)}

    # -- background loop --------------------------------------------------
    def start(self, interval=5.0):
        if self._thread:
            return

        def loop():
            while True:
                self.safe_tick()
                time.sleep(interval)
        self._thread = threading.Thread(target=loop, daemon=True, name="elastic-vram")
        self._thread.start()
