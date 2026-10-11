import conftest_paths  # noqa: F401
import datetime as dt, io, json, os, shutil, tempfile, threading, time, unittest
from unittest import mock

import config
import embers_cli
from embers import jobs, lock, scheduler, templates
from embers.llm import LLMError, RouterUnavailable
from embers.sched import SWAP_IDLE, WAIT_MAX

CREATED = dt.datetime(2026, 10, 1, 9, 0)
NOW = dt.datetime(2026, 10, 5, 8, 0)          # a Monday: ingest 02:00 and brief 07:00 are due
TPL = templates.parse_template({
    "name": "sched-test", "title": "Sched Test", "mission": "Test the scheduler.",
    "schema_md": "# Schema", "page_kinds": ["projects"], "slots": [],
    "jobs": {"ingest": "02:00", "brief": "07:00", "lint": "off"}})


class State:
    """Scripted router state shared by every FakeRouter built in one test."""

    def __init__(self):
        self.loaded = "base-model"     # str, None or an Exception to raise
        self.activity = 0              # int or an Exception to raise
        self.ctor_error = None
        self.load_ok = True            # wait_status result after load()
        self.load_error = None         # raised by load()
        self.ctors = 0
        self.calls = []
        self.probe_errors = []         # activity/n_ctx asked about a model that is not loaded
        self.sleeping = None           # model id models() reports as "sleeping"
        self.loading = None            # model id models() reports as "loading"
        self.work = 0                  # n_decode_total of the loaded model (None: no counter)
        self.replies = []              # texts complete() returns first, then "{}"
        self.on_complete = None        # callable() run inside each complete()
        self.complete_error = None     # raised by complete()
        self.cancelled = []            # the cancelled callables wait_status received


class FakeRouter:
    def __init__(self, state, cfg):
        state.ctors += 1
        if state.ctor_error is not None:
            raise state.ctor_error
        self.s = state

    def _current(self):
        return None if isinstance(self.s.loaded, Exception) else self.s.loaded

    def loaded_entry(self):
        self.s.calls.append(("loaded_model",))
        if isinstance(self.s.loaded, Exception):
            raise self.s.loaded
        if self.s.loaded:
            return {"id": self.s.loaded, "status": "loaded", "failed": False}
        if self.s.sleeping:
            return {"id": self.s.sleeping, "status": "sleeping", "failed": False}
        return None

    def loaded_model(self):
        e = self.loaded_entry()
        return e["id"] if e else None

    def loaded_ids(self, main=""):
        e = self.loaded_entry()
        return [e["id"]] if e else []

    def report(self, model):
        self.s.calls.append(("activity", model))
        if model != self._current():           # asleep or not loaded: /metrics would load or wake it
            self.s.probe_errors.append(("activity", model))
        if isinstance(self.s.activity, Exception):
            raise self.s.activity
        return {"busy": self.s.activity, "work": self.s.work}

    def load(self, model):
        self.s.calls.append(("load", model))
        if self.s.load_error is not None:
            raise self.s.load_error
        self.s.loaded = model if self.s.load_ok else None

    def unload(self, model):
        self.s.calls.append(("unload", model))
        self.s.loaded = None

    def wait_status(self, model, want, timeout=600, sleep=time.sleep, clock=time.monotonic,
                    cancelled=None):
        self.s.calls.append(("wait_status", model, want))
        self.s.cancelled.append(cancelled)
        return self.s.load_ok

    def n_ctx(self, model):
        self.s.calls.append(("n_ctx", model))
        if model not in (self._current(), self.s.sleeping):   # /props bypasses sleep
            self.s.probe_errors.append(("n_ctx", model))
        return 16384

    def models(self):
        self.s.calls.append(("models",))
        out = [{"id": self.s.loaded, "status": "loaded"}] if isinstance(self.s.loaded, str) else []
        if self.s.sleeping:
            out.append({"id": self.s.sleeping, "status": "sleeping"})
        if self.s.loading:
            out.append({"id": self.s.loading, "status": "loading"})
        return out

    def complete(self, model, autoload=True):
        def call(messages, schema, max_tokens):
            self.s.calls.append(("chat", model) if autoload is False else ("chat-autoload", model))
            if self.s.on_complete:
                self.s.on_complete()
            if self.s.complete_error is not None:
                raise self.s.complete_error
            return (self.s.replies.pop(0) if self.s.replies else "{}"), {}
        return call


class Runners:
    """Fake jobs: record the call, write a real run row, return a status."""

    def __init__(self, state):
        self.state = state
        self.calls = []
        self.raise_for = set()         # ember ids whose runner raises
        self.on_run = None             # callable(ember_id, job) run inside the job
        self.llm_calls = 0             # model calls each job makes (after on_run)
        self.llm_errors = []           # exceptions those calls raised

    def make(self, job):
        def run(ember, llm, now, n_ctx=8192):
            self.calls.append((ember.id, job, llm.model, n_ctx, now))
            st = ember.store
            with st.db:
                rid = st.start_run(job, now)
            if self.on_run:
                self.on_run(ember.id, job)
            for _ in range(self.llm_calls):
                try:
                    llm([{"role": "user", "content": "x"}], {"type": "object"})
                except RouterUnavailable as e:   # what jobs do: stop, count nothing
                    self.llm_errors.append(e)
                    with st.db:
                        st.finish_run(rid, now, "failed", str(e))
                    return {"status": "failed", "router_down": True}
            if ember.id in self.raise_for:
                with st.db:
                    st.finish_run(rid, now, "failed", "boom")
                raise RuntimeError("boom")
            with st.db:
                st.finish_run(rid, now, "ok")
            return {"status": "ok"}
        return run

    def table(self):
        return {j: self.make(j) for j in ("ingest", "brief", "lint")}


class SchedCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.base = os.path.join(self.dir, "embers")
        self.cfg = {"embers_dir": self.base, "router_port": 9}
        self.state = State()
        self.runners = Runners(self.state)
        self.t = [0.0]
        self.now = [NOW]
        self.sched = self.make_sched()

    def make_sched(self):
        return scheduler.Scheduler(lambda: dict(self.cfg),
                                   router_cls=lambda cfg: FakeRouter(self.state, cfg),
                                   now=lambda: self.now[0], clock=lambda: self.t[0],
                                   sleep=lambda s: None, runners=self.runners.table())

    def tearDown(self):
        self.assertEqual(self.state.probe_errors, [], "activity/n_ctx probed an unloaded model")

    def ember(self, eid, created=CREATED, **conf):
        root = jobs.create_ember(self.base, TPL, eid, {}, created)
        if conf:
            self.set_conf(root, **conf)
        return root

    def set_conf(self, root, **kw):
        path = os.path.join(root, "ember.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.update(kw)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def runs(self, root, job=None):
        with jobs.Ember(root) as e:
            rows = e.store.recent_runs(100)
        return [r for r in rows if job is None or r["job"] == job]

    def ops(self):
        return [c for c in self.state.calls if c[0] in ("load", "unload")]


class NothingDueTest(SchedCase):
    def test_no_embers_no_router(self):
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertIsNone(r["error"])
        self.assertEqual(self.state.ctors, 0)

    def test_nothing_due_makes_no_router_calls(self):
        self.ember("fresh", created=dt.datetime(2026, 10, 4, 23, 0))
        self.now[0] = dt.datetime(2026, 10, 5, 1, 0)
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertEqual(r["waiting"], {})
        self.assertEqual(self.state.ctors, 0)
        self.assertEqual(self.state.calls, [])
        self.assertEqual(self.runners.calls, [])

    def test_scheduler_off_is_idle(self):
        self.ember("a")
        self.cfg["embers_scheduler"] = False
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertEqual(self.state.ctors, 0)
        self.assertEqual(self.runners.calls, [])

    def test_disabled_ember_ignored(self):
        self.ember("a", enabled=False)
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertEqual(self.state.ctors, 0)
        self.assertFalse(self.sched.status()["a"]["enabled"])

    def test_invalid_folders_ignored(self):
        os.makedirs(os.path.join(self.base, "Not_Valid"))
        os.makedirs(os.path.join(self.base, "con"))
        os.makedirs(os.path.join(self.base, "nojson"))
        self.assertIsNone(self.sched.tick()["error"])
        self.assertEqual(self.state.ctors, 0)


class RunTest(SchedCase):
    def test_due_idle_loaded_runs_once(self):
        root = self.ember("a")
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "ok"))
        self.assertEqual(len(self.runners.calls), 1)
        eid, job, model, n_ctx, _ = self.runners.calls[0]
        self.assertEqual((eid, job, model, n_ctx), ("a", "ingest", "base-model", 16384))
        self.assertEqual(self.ops(), [])
        self.assertEqual(r["ran"], ("a", "ingest", "ok"))
        # ingest has an attempt row now: brief runs next, then nothing
        self.assertEqual(self.sched.tick()["ran"], ("a", "brief", "ok"))
        self.assertIsNone(self.sched.tick()["ran"])
        self.assertEqual(len(self.runners.calls), 2)
        self.assertEqual([x["status"] for x in self.runs(root)], ["ok", "ok"])

    def test_ingest_then_brief_order(self):
        self.ember("a")
        self.sched.tick()
        self.sched.tick()
        self.assertEqual([c[1] for c in self.runners.calls], ["ingest", "brief"])

    def test_n_ctx_is_clamped(self):
        self.ember("a")
        with mock.patch.object(FakeRouter, "n_ctx", lambda s, m: 10 ** 12):
            self.sched.tick()
        from embers import llm
        self.assertEqual(self.runners.calls[0][3], llm.MAX_N_CTX)

    def test_status_next_times(self):
        self.ember("a")
        self.ember("b", created=dt.datetime(2026, 10, 4, 23, 0))
        self.now[0] = dt.datetime(2026, 10, 5, 1, 0)
        self.sched.tick()
        st = self.sched.status()
        self.assertEqual(st["b"]["next"], {"ingest": "2026-10-05T02:00", "brief": "2026-10-05T07:00"})
        self.assertTrue(st["b"]["enabled"])
        self.assertIsNone(st["b"]["running"])
        self.assertIn("last_error", st)

    def test_status_is_a_copy(self):
        self.ember("a")
        self.state.activity = 3
        self.sched.tick()
        st = self.sched.status()
        st["a"]["waiting"]["ingest"] = "mutated"
        self.assertNotEqual(self.sched.status()["a"]["waiting"]["ingest"], "mutated")

    def test_running_is_visible_during_the_job(self):
        self.ember("a")
        seen = []
        self.runners.on_run = lambda eid, job: seen.append(self.sched.status()[eid]["running"])
        self.sched.tick()
        self.assertEqual(seen, ["ingest"])
        self.assertIsNone(self.sched.status()["a"]["running"])


class WaitTest(SchedCase):
    def test_busy_waits_then_skips(self):
        root = self.ember("a")
        self.state.activity = 2
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertIn("busy", r["waiting"]["a"]["ingest"])
        self.assertIn("busy", self.sched.status()["a"]["waiting"]["ingest"])
        self.t[0] = WAIT_MAX - 1
        self.assertIn("ingest", self.sched.tick()["waiting"]["a"])
        self.assertEqual(self.runs(root), [])
        self.t[0] = WAIT_MAX
        r = self.sched.tick()
        self.assertEqual(r["skipped"][0][:2], ("a", "ingest"))
        rows = self.runs(root, "ingest")
        self.assertEqual(rows[0]["status"], "skipped")
        self.assertIn("busy", rows[0]["error"])
        self.assertEqual(self.runners.calls, [])
        # the skip counts as the attempt: brief waits next, with a fresh wait clock
        r = self.sched.tick()
        self.assertIn("brief", r["waiting"]["a"])
        self.assertNotIn("ingest", self.sched.status()["a"]["waiting"])

    def test_router_errors_wait_not_reachable(self):
        self.ember("a")
        cases = [("ctor_error", LLMError("bad port")), ("loaded", RouterUnavailable("refused")),
                 ("activity", LLMError("no counters"))]
        for attr, exc in cases:
            self.state = State()
            setattr(self.state, attr, exc)
            sched = self.make_sched()
            r = sched.tick()
            self.assertIsNone(r["ran"], attr)
            self.assertIn("not reachable", r["waiting"]["a"]["ingest"], attr)
            self.assertEqual(self.runners.calls, [], attr)

    def test_no_model_loaded_waits(self):
        self.ember("a")
        self.state.loaded = None
        r = self.sched.tick()
        self.assertIn("no model", r["waiting"]["a"]["ingest"])
        self.assertNotIn(("activity", None), self.state.calls)

    def test_locked_ember_waits(self):
        root = self.ember("a")
        with lock.held(root):
            r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertIn("running in another process", r["waiting"]["a"]["ingest"])
        self.assertEqual(self.runners.calls, [])
        # held for over WAIT_MAX: still no skip while the lock is the only blocker
        self.t[0] = WAIT_MAX * 2
        with lock.held(root):
            r = self.sched.tick()
        self.assertEqual(r["skipped"], [])
        self.assertEqual(self.runs(root), [])
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))


class OwnBuildTest(SchedCase):
    """Embers reach models through the router; a model pinned to another
    llama.cpp build runs in its own process, which the router doesn't serve."""

    class Pool:
        def __init__(self, why):
            self.why = why

        def active(self):
            return False

        def own_build(self, mid):
            return self.why.get(mid, "")

    def test_a_model_on_its_own_build_waits_and_is_never_swapped_in(self):
        root = self.ember("a", model="ik-model")
        self.sched.pool = self.Pool({"ik-model": "ik-model runs on its own build"})
        self.t[0] = SWAP_IDLE * 3
        for _ in range(3):
            r = self.sched.tick()
            self.assertIsNone(r["ran"])
            self.assertIn("own build", r["waiting"]["a"]["ingest"])
        self.assertEqual(self.ops(), [])
        self.t[0] = WAIT_MAX * 3          # a config condition: never turns into a skip
        self.assertEqual(self.sched.tick()["skipped"], [])
        self.assertEqual(self.runs(root), [])

    def test_other_models_are_unaffected(self):
        self.ember("a", model="pinned")
        self.sched.pool = self.Pool({"ik-model": "ik-model runs on its own build"})
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))

    def test_a_pool_without_the_probe_is_fine(self):
        self.ember("a")
        self.sched.pool = mock.Mock(spec=["active"], active=lambda: False)
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))


class SwapTest(SchedCase):
    def test_pinned_waits_for_idle_then_swaps_and_restores(self):
        self.ember("a", model="pinned")
        self.sched.tick()                  # the first look: the idle clock starts here
        self.t[0] = SWAP_IDLE - 1
        r = self.sched.tick()
        self.assertIn("idle minutes", r["waiting"]["a"]["ingest"])
        self.assertEqual(self.ops(), [])
        self.t[0] = SWAP_IDLE
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.runners.calls[0][2], "pinned")
        self.assertEqual(self.ops(), [("load", "pinned"), ("load", "base-model")])
        self.assertEqual(self.state.loaded, "base-model")
        # a swap resets the idle clock: brief needs another 10 idle minutes
        r = self.sched.tick()
        self.assertIn("idle minutes", r["waiting"]["a"]["brief"])
        self.t[0] = 2 * SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "brief", "ok"))

    def test_nothing_loaded_before_unloads_after(self):
        self.ember("a", model="pinned")
        self.state.loaded = None
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.ops(), [("load", "pinned"), ("unload", "pinned")])

    def test_user_swapped_meanwhile_no_restore(self):
        self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE

        def user_takes_gpu(eid, job):
            self.state.loaded = "users-model"
        self.runners.on_run = user_takes_gpu
        self.sched.tick()
        self.assertEqual(self.ops(), [("load", "pinned")])
        self.assertEqual(self.state.loaded, "users-model")

    def test_load_fails_skips_and_restores(self):
        root = self.ember("a", model="pinned")
        self.state.load_ok = False
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertEqual(self.runners.calls, [])
        rows = self.runs(root, "ingest")
        self.assertEqual(rows[0]["status"], "skipped")
        self.assertIn("could not load pinned", rows[0]["error"])
        # the failed load left nothing loaded: the previous model goes back
        self.assertEqual(self.ops(), [("load", "pinned"), ("load", "base-model")])
        self.assertNotIn(("n_ctx", "pinned"), self.state.calls)

    def test_load_router_down_waits_without_skip(self):
        root = self.ember("a", model="pinned")
        self.state.load_error = RouterUnavailable("restarting")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertIn("ingest", r["waiting"]["a"])
        self.assertEqual(self.runs(root), [])
        self.assertEqual(self.ops(), [("load", "pinned")])   # never loaded: nothing to put back

    def test_swapping_off_waits(self):
        self.ember("a", model="pinned")
        self.cfg["embers_swap_models"] = False
        self.t[0] = SWAP_IDLE * 3
        r = self.sched.tick()
        self.assertIn("swapping is off", r["waiting"]["a"]["ingest"])
        self.assertEqual(self.ops(), [])

    def test_waiting_ember_does_not_block_a_runnable_one(self):
        self.ember("a", model="pinned")
        self.ember("b")
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("b", "ingest", "ok"))
        self.assertIn("idle minutes", r["waiting"]["a"]["ingest"])
        self.assertEqual(self.ops(), [])

    def test_first_look_never_swaps(self):
        # Unwatched time is not idle time: nobody saw what the router did before.
        self.ember("a", model="pinned")
        self.t[0] = 100.0
        sched = self.make_sched()          # constructed at t=100
        self.t[0] = 100.0 + 5 * SWAP_IDLE
        self.assertIn("idle minutes", sched.tick()["waiting"]["a"]["ingest"])
        self.t[0] = 100.0 + 6 * SWAP_IDLE - 1
        self.assertIn("idle minutes", sched.tick()["waiting"]["a"]["ingest"])
        self.t[0] = 100.0 + 6 * SWAP_IDLE
        self.assertEqual(sched.tick()["ran"], ("a", "ingest", "ok"))

    def test_activity_resets_idle_clock(self):
        self.ember("a", model="pinned")
        self.state.activity = 1
        self.t[0] = SWAP_IDLE
        self.sched.tick()                  # busy at t=600: the idle clock restarts
        self.state.activity = 0
        self.t[0] = SWAP_IDLE + 10
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["a"]["ingest"])
        self.t[0] = 2 * SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))

    def test_loaded_model_change_resets_idle_clock(self):
        self.ember("a", model="pinned")
        self.t[0] = SWAP_IDLE - 5
        self.sched.tick()
        self.state.loaded = "other"
        self.t[0] = SWAP_IDLE
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["a"]["ingest"])


class FallbackModelTest(SchedCase):
    """An ember with no model of its own uses what is loaded; with nothing
    loaded it falls back to Setup's auto-load model, like a pinned one."""

    def test_nothing_loaded_loads_the_auto_load_model_then_unloads_it(self):
        self.ember("a")
        self.cfg["auto_load_model"] = "everyday"
        self.state.loaded = None
        self.sched.tick()
        self.t[0] = SWAP_IDLE - 1
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["a"]["ingest"])
        self.t[0] = SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.runners.calls[0][2], "everyday")
        self.assertEqual(self.ops(), [("load", "everyday"), ("unload", "everyday")])

    def test_a_loaded_model_still_wins(self):
        self.ember("a")
        self.cfg["auto_load_model"] = "everyday"
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.runners.calls[0][2], "base-model")
        self.assertEqual(self.ops(), [])

    def test_no_auto_load_model_still_waits(self):
        self.ember("a")
        self.state.loaded = None
        self.t[0] = SWAP_IDLE * 3
        note = self.sched.tick()["waiting"]["a"]["ingest"]
        self.assertTrue(note.startswith("no model is loaded; pick a model"), note)
        self.assertIn("auto-load model in Setup", note)
        self.assertEqual(self.ops(), [])

    def test_other_reasons_get_no_hint(self):
        self.ember("a")
        self.state.activity = 2
        note = self.sched.tick()["waiting"]["a"]["ingest"]
        self.assertIn("busy", note)
        self.assertNotIn("pick a model", note)

    def test_the_skip_says_how_to_fix_it(self):
        self.ember("a")
        self.state.loaded = None
        self.sched.tick()
        self.t[0] = WAIT_MAX
        skipped = self.sched.tick()["skipped"]
        self.assertEqual(len(skipped), 1, skipped)
        self.assertIn("for 60 minutes; pick a model", str(skipped[0]))

    def test_swapping_off_never_loads_it(self):
        self.ember("a")
        self.cfg["auto_load_model"] = "everyday"
        self.cfg["embers_swap_models"] = False
        self.state.loaded = None
        self.t[0] = SWAP_IDLE * 3
        self.assertIn("swapping is off", self.sched.tick()["waiting"]["a"]["ingest"])
        self.assertEqual(self.ops(), [])

    def test_a_pinned_model_beats_the_auto_load_model(self):
        self.ember("a", model="pinned")
        self.cfg["auto_load_model"] = "everyday"
        self.state.loaded = None
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.sched.tick()
        self.assertEqual(self.ops(), [("load", "pinned"), ("unload", "pinned")])


class PreemptTest(SchedCase):
    def test_model_calls_go_through_while_ours_is_loaded(self):
        self.ember("a")
        self.runners.llm_calls = 3
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.state.calls.count(("chat", "base-model")), 3)
        self.assertEqual(self.runners.llm_errors, [])

    def test_user_swaps_mid_run_preempts_without_restore(self):
        root = self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.runners.llm_calls = 2

        def user_takes_gpu(eid, job):
            self.state.loaded = "users-model"
        self.runners.on_run = user_takes_gpu
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "failed"))
        self.assertEqual(len(self.runners.llm_errors), 1)
        self.assertIsInstance(self.runners.llm_errors[0], RouterUnavailable)
        self.assertIn("pre-empted: users-model is loaded now", str(self.runners.llm_errors[0]))
        self.assertNotIn(("chat", "pinned"), self.state.calls)
        self.assertEqual(self.ops(), [("load", "pinned")])          # no restore, no reload
        self.assertEqual(self.state.loaded, "users-model")
        self.assertIn("pre-empted", self.runs(root, "ingest")[0]["error"])

    def test_unloaded_mid_run_preempts_without_restore(self):
        self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.runners.llm_calls = 1

        def user_unloads(eid, job):
            self.state.loaded = None
        self.runners.on_run = user_unloads
        self.sched.tick()
        self.assertIn("pre-empted: nothing is loaded now", str(self.runners.llm_errors[0]))
        self.assertEqual(self.ops(), [("load", "pinned")])

    def test_check_error_becomes_router_unavailable(self):
        self.ember("a")
        self.runners.llm_calls = 1

        def router_dies(eid, job):
            self.state.loaded = LLMError("connection reset")
        self.runners.on_run = router_dies
        self.sched.tick()
        self.assertIsInstance(self.runners.llm_errors[0], RouterUnavailable)
        self.assertIn("connection reset", str(self.runners.llm_errors[0]))

    def test_sleeping_model_is_still_ours(self):
        self.ember("a")
        self.runners.llm_calls = 1

        def falls_asleep(eid, job):
            self.state.loaded, self.state.sleeping = None, "base-model"
        self.runners.on_run = falls_asleep
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.runners.llm_errors, [])

    def test_preemption_resets_idle_clock(self):
        self.ember("a", model="pinned")
        self.ember("b", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.runners.llm_calls = 1
        self.runners.on_run = lambda eid, job: setattr(self.state, "loaded", "users-model")
        self.sched.tick()
        self.runners.on_run = None
        self.t[0] = 2 * SWAP_IDLE - 1
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["b"]["ingest"])


class OwnJobActivityTest(SchedCase):
    def test_job_end_counts_as_activity(self):
        self.ember("a")
        self.ember("b", model="pinned")
        self.t[0] = SWAP_IDLE
        clock_at_end = SWAP_IDLE + 300

        def long_job(eid, job):
            self.t[0] = clock_at_end           # the job took 5 minutes
        self.runners.on_run = long_job
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.runners.on_run = None
        self.t[0] = clock_at_end + SWAP_IDLE - 1
        r = self.sched.tick()                  # a brief runs (no swap needed); b still waits
        self.assertEqual(r["ran"], ("a", "brief", "ok"))
        self.t[0] = clock_at_end + SWAP_IDLE - 1
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["b"]["ingest"])
        self.t[0] = clock_at_end + 2 * SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("b", "ingest", "ok"))

    def test_failed_job_counts_as_activity(self):
        self.ember("a")
        self.ember("b", model="pinned")
        self.runners.raise_for = {"a"}
        self.t[0] = SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "error"))
        self.assertEqual(self.sched.tick()["ran"], ("a", "brief", "error"))
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["b"]["ingest"])


class ErrorTest(SchedCase):
    def test_runner_raises_restores_and_others_continue(self):
        ra = self.ember("a", model="pinned")
        self.ember("b")
        self.runners.raise_for = {"a"}
        self.assertEqual(self.sched.tick()["ran"], ("b", "ingest", "ok"))
        self.assertEqual(self.sched.tick()["ran"], ("b", "brief", "ok"))
        self.sched.tick()                  # a's first look after b's jobs
        self.t[0] = SWAP_IDLE
        r = self.sched.tick()
        self.assertIn("boom", r["error"])
        self.assertIn("boom", self.sched.status()["last_error"])
        self.assertEqual(self.ops(), [("load", "pinned"), ("load", "base-model")])
        self.assertEqual(self.runs(ra, "ingest")[0]["status"], "failed")
        self.assertEqual(len(self.runs(ra, "ingest")), 1)

    def test_broken_ember_does_not_stop_others(self):
        broken = os.path.join(self.base, "aaa")
        os.makedirs(broken)
        with open(os.path.join(broken, "ember.json"), "w") as f:
            f.write("{not json")
        self.ember("b")
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("b", "ingest", "ok"))
        self.assertIn("aaa", self.sched.status()["last_error"])

    def test_restore_errors_never_raise(self):
        self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        real_load = FakeRouter.load

        def load(router, model):
            if model == "base-model":
                raise LLMError("restore failed")
            real_load(router, model)
        with mock.patch.object(FakeRouter, "load", load):
            r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "ok"))
        self.assertIn("restore failed", self.sched.status()["last_error"])

    def test_tick_never_raises_on_cfg_error(self):
        def bad():
            raise OSError("disk gone")
        s = scheduler.Scheduler(bad, router_cls=lambda cfg: FakeRouter(self.state, cfg))
        r = s.tick()
        self.assertIn("disk gone", r["error"])
        self.assertIn("disk gone", s.status()["last_error"])


class ThreadTest(SchedCase):
    def test_start_stop_idempotent_and_prompt(self):
        self.ember("a")
        with mock.patch.object(scheduler, "TICK", 0.01):
            self.sched.start()
            self.sched.start()
            names = [t.name for t in threading.enumerate()].count("embers-scheduler")
            self.assertEqual(names, 1)
            deadline = time.monotonic() + 10
            while len(self.runners.calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            t0 = time.monotonic()
            self.sched.stop()
            self.assertLess(time.monotonic() - t0, 5)
            self.sched.stop()
        self.assertEqual([c[1] for c in self.runners.calls], ["ingest", "brief"])
        self.assertNotIn("embers-scheduler", [t.name for t in threading.enumerate()])

    def test_loop_survives_exceptions(self):
        calls = []

        def boom():
            calls.append(1)
            raise ValueError("bad tick")
        with mock.patch.object(scheduler, "TICK", 0.01), \
                mock.patch.object(self.sched, "tick", boom):
            self.sched.start()
            deadline = time.monotonic() + 10
            while len(calls) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.sched.stop()
        self.assertGreaterEqual(len(calls), 3)
        self.assertIn("ValueError: bad tick", self.sched.status()["last_error"])


class ServerImportTest(unittest.TestCase):
    def test_import_starts_nothing(self):
        import server
        self.assertIsNone(server.EMBERS_SCHED)
        self.assertNotIn("embers-scheduler", [t.name for t in threading.enumerate()])

    def test_defaults(self):
        self.assertIs(config.DEFAULTS["embers_scheduler"], True)
        self.assertIs(config.DEFAULTS["embers_swap_models"], True)


class CliTickTest(SchedCase):
    def setUp(self):
        super().setUp()
        self._orig = config.CONFIG
        config.CONFIG = os.path.join(self.dir, "config.json")
        self.addCleanup(setattr, config, "CONFIG", self._orig)
        config.update({"embers_dir": self.base, "router_port": 9})

    def cli(self, now):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(scheduler, "RUNNERS", self.runners.table()):
            code = embers_cli.main(["tick"], now=now, out=out, err=err,
                                   router_cls=lambda cfg: FakeRouter(self.state, cfg))
        return code, out.getvalue(), err.getvalue()

    def test_nothing_due(self):
        self.ember("a", created=dt.datetime(2026, 10, 4, 23, 0))
        code, out, err = self.cli(dt.datetime(2026, 10, 5, 1, 0))
        self.assertEqual(code, 0, err)
        self.assertIn("nothing due", out)
        self.assertIn("2026-10-05T02:00", out)
        self.assertEqual(self.state.ctors, 0)

    def test_ran(self):
        self.ember("a")
        code, out, err = self.cli(NOW)
        self.assertEqual(code, 0, err)
        self.assertIn("ran a ingest: ok", out)

    def test_waiting_and_error_exit(self):
        self.ember("a")
        self.state.activity = 5
        code, out, err = self.cli(NOW)
        self.assertEqual(code, 0, err)
        self.assertIn("a ingest: waiting: the router is busy", out)
        self.runners.raise_for = {"a"}
        self.state.activity = 0
        code, out, err = self.cli(NOW)
        self.assertEqual(code, 1)
        self.assertIn("boom", err)


class ObservedIdleTest(SchedCase):
    """The idle clock only runs across looks that prove nothing happened in between."""

    def test_work_between_two_idle_looks_is_not_idle(self):
        # busy is 0 at both looks, but the decode counter moved: someone used the model.
        self.ember("a", model="pinned")
        self.sched.tick()
        self.state.work = 57
        self.t[0] = SWAP_IDLE
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["a"]["ingest"])
        self.t[0] = 2 * SWAP_IDLE - 1
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["a"]["ingest"])
        self.t[0] = 2 * SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))

    def test_without_counter_only_close_looks_count(self):
        self.ember("a", model="pinned")
        self.state.work = None
        self.sched.tick()
        self.t[0] = SWAP_IDLE              # a 10-minute gap the gauge cannot vouch for
        self.assertIn("idle minutes", self.sched.tick()["waiting"]["a"]["ingest"])
        for k in range(1, 11):             # looks every TICK: trusted
            self.t[0] = SWAP_IDLE + k * scheduler.TICK
            r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "ok"))

    def test_sleeping_model_is_idle_and_never_probed(self):
        # /metrics wakes a sleeping model: the scheduler must not ask it.
        self.ember("a", model="pinned")
        self.state.loaded, self.state.sleeping = None, "base-model"
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertNotIn(("activity", "base-model"), self.state.calls)

    def test_sleeping_unpinned_job_runs_on_it(self):
        self.ember("a")
        self.state.loaded, self.state.sleeping = None, "base-model"
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.runners.calls[0][2], "base-model")


class GuardTest(SchedCase):
    def test_requests_never_autoload(self):
        self.ember("a")
        self.runners.llm_calls = 2
        self.sched.tick()
        self.assertEqual(self.state.calls.count(("chat", "base-model")), 2)
        self.assertNotIn(("chat-autoload", "base-model"), self.state.calls)

    def test_retry_is_guarded_too(self):
        # The first reply is junk; before ask_json's retry the user loads their model.
        self.ember("a")
        self.runners.llm_calls = 1
        self.state.replies = ["not json"]
        self.state.on_complete = lambda: setattr(self.state, "loaded", "users-model")
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "failed"))
        self.assertEqual(self.state.calls.count(("chat", "base-model")), 1)
        self.assertIn("pre-empted: users-model", str(self.runners.llm_errors[0]))

    def test_model_not_loaded_reply_is_preemption(self):
        # Switched between our check and the request: the router refuses instead of loading ours.
        from embers.llm import ModelNotLoaded
        self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.runners.llm_calls = 1

        def switch():
            self.state.loaded = "users-model"
            self.state.complete_error = ModelNotLoaded("router answered 400: model is not loaded")
        self.state.on_complete = switch
        self.sched.tick()
        self.assertIsInstance(self.runners.llm_errors[0], RouterUnavailable)
        self.assertIn("pre-empted: users-model", str(self.runners.llm_errors[0]))
        self.assertEqual(self.ops(), [("load", "pinned")])          # theirs now: no restore

    def test_error_mid_request_after_switch_is_not_the_inputs_fault(self):
        self.ember("a")
        self.runners.llm_calls = 1

        def killed():
            self.state.loaded = "users-model"
            self.state.complete_error = LLMError("router unreachable or unreadable: connection reset")
        self.state.on_complete = killed
        self.sched.tick()
        self.assertIsInstance(self.runners.llm_errors[0], RouterUnavailable)

    def test_plain_error_with_our_model_still_loaded_propagates(self):
        self.ember("a")
        self.runners.llm_calls = 1
        self.state.complete_error = LLMError("router answered 500: oops")
        r = self.sched.tick()
        self.assertEqual(r["ran"], ("a", "ingest", "error"))
        self.assertIn("oops", r["error"])


class RestoreTest(SchedCase):
    def test_no_restore_while_the_user_loads_a_model(self):
        self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.runners.on_run = lambda eid, job: setattr(self.state, "loading", "users-model")
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.ops(), [("load", "pinned")])

    def test_wait_for_load_is_cancellable_by_stop(self):
        self.ember("a", model="pinned")
        self.sched.tick()
        self.t[0] = SWAP_IDLE
        self.sched.tick()
        cancelled = self.state.cancelled[0]
        self.assertFalse(cancelled())
        self.sched.stop()
        self.assertTrue(cancelled())

    def test_default_sleep_is_interrupted_by_stop(self):
        s = scheduler.Scheduler(lambda: {})
        s.stop()
        t0 = time.monotonic()
        s.sleep(30)
        self.assertLess(time.monotonic() - t0, 5)


if __name__ == "__main__":
    unittest.main()


class ManualRunTest(SchedCase):
    """Run now from the panel: request() queues jobs that then go through the
    same lock, router gate and pool rules as scheduled ones."""

    def fresh(self, eid="fresh", **conf):
        root = self.ember(eid, created=dt.datetime(2026, 10, 4, 23, 0), **conf)
        self.now[0] = dt.datetime(2026, 10, 5, 1, 0)      # nothing due yet
        return root

    def test_runs_jobs_that_are_not_due_in_order(self):
        root = self.fresh()
        self.sched.request("fresh", ["brief", "ingest"])
        self.assertEqual(self.sched.tick()["ran"], ("fresh", "ingest", "ok"))
        self.assertEqual(self.sched.tick()["ran"], ("fresh", "brief", "ok"))
        self.assertIsNone(self.sched.tick()["ran"])
        self.assertEqual([r["job"] for r in self.runs(root)], ["brief", "ingest"])
        self.assertEqual(self.sched.status()["fresh"]["queued"], [])

    def test_queued_is_visible_before_the_next_tick(self):
        self.fresh()
        self.sched.request("fresh", ["ingest"])
        self.assertEqual(self.sched.status()["fresh"]["queued"], ["ingest"])
        self.sched.request("fresh", ["ingest", "brief"])      # no duplicates
        self.assertEqual(self.sched.status()["fresh"]["queued"], ["ingest", "brief"])

    def test_bad_requests_raise(self):
        self.fresh()
        for eid, wanted in (("fresh", ["bogus"]), ("fresh", []), ("Bad_Id", ["ingest"]),
                            ("con", ["ingest"]), ("fresh", "ingest")):
            with self.assertRaises(ValueError, msg=(eid, wanted)):
                self.sched.request(eid, wanted)
        self.assertEqual(self.sched.status().get("fresh", {}).get("queued", []), [])

    def test_disabled_ember_runs_when_asked(self):
        self.fresh(enabled=False)
        self.sched.request("fresh", ["ingest"])
        self.assertEqual(self.sched.tick()["ran"], ("fresh", "ingest", "ok"))
        self.assertIsNone(self.sched.tick()["ran"])

    def test_scheduler_off_still_runs_requests_only(self):
        self.ember("due")                                  # due, but the scheduler is off
        self.fresh()
        self.cfg["embers_scheduler"] = False
        self.sched.request("fresh", ["ingest"])
        self.assertEqual(self.sched.tick()["ran"], ("fresh", "ingest", "ok"))
        self.assertIsNone(self.sched.tick()["ran"])
        self.assertEqual([c[0] for c in self.runners.calls], ["fresh"])

    def test_requests_wait_while_the_router_is_busy(self):
        self.fresh()
        self.state.activity = 2
        self.sched.request("fresh", ["ingest"])
        r = self.sched.tick()
        self.assertIsNone(r["ran"])
        self.assertIn("busy", r["waiting"]["fresh"]["ingest"])
        self.assertEqual(self.sched.status()["fresh"]["queued"], ["ingest"])
        self.state.activity = 0
        self.assertEqual(self.sched.tick()["ran"], ("fresh", "ingest", "ok"))
        self.assertEqual(self.sched.status()["fresh"]["queued"], [])

    def test_skipped_after_wait_max_leaves_the_queue(self):
        root = self.fresh()
        self.state.activity = 2
        self.sched.request("fresh", ["ingest"])
        self.sched.tick()
        self.t[0] = WAIT_MAX
        r = self.sched.tick()
        self.assertEqual(r["skipped"][0][:2], ("fresh", "ingest"))
        self.assertEqual(self.sched.status()["fresh"]["queued"], [])
        self.assertEqual(self.runs(root, "ingest")[0]["status"], "skipped")

    def test_due_and_requested_runs_once(self):
        self.ember("a")                                    # ingest and brief due
        self.sched.request("a", ["ingest"])
        self.assertEqual(self.sched.tick()["ran"], ("a", "ingest", "ok"))
        self.assertEqual(self.sched.tick()["ran"], ("a", "brief", "ok"))
        self.assertIsNone(self.sched.tick()["ran"])
        self.assertEqual([c[1] for c in self.runners.calls], ["ingest", "brief"])

    def test_requests_go_before_other_embers_due_jobs(self):
        self.ember("a")                                    # due
        self.ember("b")
        self.sched.request("b", ["lint"])
        self.assertEqual(self.sched.tick()["ran"], ("b", "lint", "ok"))

    def test_cancel_clears_the_queue(self):
        self.fresh()
        self.sched.request("fresh", ["ingest", "brief"])
        self.assertEqual(self.sched.cancel("fresh"), ["ingest", "brief"])
        self.assertIsNone(self.sched.tick()["ran"])
        self.assertEqual(self.sched.status()["fresh"]["queued"], [])
        self.assertEqual(self.sched.cancel("fresh"), [])

    def test_locked_ember_keeps_the_request(self):
        root = self.fresh()
        self.sched.request("fresh", ["ingest"])
        with lock.held(root):
            r = self.sched.tick()
        self.assertIn("running in another process", r["waiting"]["fresh"]["ingest"])
        self.assertEqual(self.sched.tick()["ran"], ("fresh", "ingest", "ok"))

    def test_request_wakes_the_loop(self):
        self.fresh()
        with mock.patch.object(scheduler, "TICK", 30):
            self.sched.start()
            try:
                time.sleep(0.2)                            # first pass: nothing due, then a 30 s wait
                self.sched.request("fresh", ["ingest"])
                deadline = time.monotonic() + 5
                while not self.runners.calls and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                self.sched.stop()
        self.assertEqual([c[1] for c in self.runners.calls], ["ingest"])


class AfterRunHookTest(SchedCase):
    """after_run(ember, job, result, now) sees every finished job while the
    ember is still open; push uses it to send a brief."""

    def make_sched(self):
        self.hooked = []
        self.hook_error = None

        def hook(ember, job, result, now):
            self.hooked.append((ember.id, ember.conf["name"], job, result["status"], now))
            if self.hook_error:
                raise self.hook_error
        return scheduler.Scheduler(lambda: dict(self.cfg),
                                   router_cls=lambda cfg: FakeRouter(self.state, cfg),
                                   now=lambda: self.now[0], clock=lambda: self.t[0],
                                   sleep=lambda s: None, runners=self.runners.table(), after_run=hook)

    def test_hook_sees_each_finished_job(self):
        self.ember("one", created=dt.datetime(2026, 10, 4, 23, 0))
        self.now[0] = dt.datetime(2026, 10, 5, 1, 0)
        self.sched.request("one", ["ingest", "brief"])
        self.sched.tick()
        self.sched.tick()
        self.assertEqual([(h[0], h[2], h[3]) for h in self.hooked], [("one", "ingest", "ok"), ("one", "brief", "ok")])
        self.assertEqual(self.hooked[0][4], self.now[0])

    def test_hook_failure_is_reported_not_fatal(self):
        self.ember("one", created=dt.datetime(2026, 10, 4, 23, 0))
        self.now[0] = dt.datetime(2026, 10, 5, 1, 0)
        self.hook_error = RuntimeError("ntfy exploded")
        self.sched.request("one", ["brief"])
        self.assertEqual(self.sched.tick()["ran"], ("one", "brief", "ok"))
        self.assertIn("ntfy exploded", self.sched.status()["last_error"])

    def test_no_hook_when_the_job_raised(self):
        self.ember("one", created=dt.datetime(2026, 10, 4, 23, 0))
        self.now[0] = dt.datetime(2026, 10, 5, 1, 0)
        self.runners.raise_for.add("one")
        self.sched.request("one", ["brief"])
        self.sched.tick()
        self.assertEqual(self.hooked, [])

    def test_default_hook_is_push(self):
        from embers import push
        self.assertIs(scheduler.Scheduler(lambda: {}).after_run, push.after_run)
