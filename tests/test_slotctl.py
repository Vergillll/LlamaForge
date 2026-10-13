"""SlotManager: plan -> place -> load -> measure, against a fake router and a
fake nvidia-smi whose `used` moves when the fake router loads a model."""
import conftest_paths  # noqa: F401
import os
import tempfile
import unittest
from unittest import mock

import argspec
import config
import slotctl
import slots

G = 1024


class FakeRouter:
    def __init__(self, smi, cost, ini=None):
        self.smi, self.cost = smi, cost            # cost: model -> {gpu: MiB}
        self.ini = ini                             # like the real one, it lists every section
        self.status = {}
        self.calls = []
        self.fail = set()

    def __call__(self, path, method="GET", body=None, timeout=30):
        self.calls.append((path, method, (body or {}).get("model")))
        if path.startswith("/models") and method == "GET":
            ids = [m for m in config.read_sections(self.ini) if m != "*"] if self.ini else []
            ids += [m for m in self.status if m not in ids]
            return 200, {"data": [{"id": m, "status": dict(self.status.get(m) or
                                                           {"value": "unloaded"})}
                                  for m in ids]}
        mid = (body or {}).get("model")
        if path == "/models/load":
            if mid in self.fail:
                self.status[mid] = {"value": "unloaded", "failed": True, "exit_code": 1}
                return 200, {"success": True}
            self.status[mid] = {"value": "loaded"}
            for g, mib in self.cost.get(mid, {}).items():
                self.smi[g]["used_mib"] += mib
                self.smi[g]["free_mib"] -= mib
            return 200, {"success": True}
        if path == "/models/unload":
            if (self.status.get(mid) or {}).get("value") == "loaded":
                for g, mib in self.cost.get(mid, {}).items():
                    self.smi[g]["used_mib"] -= mib
                    self.smi[g]["free_mib"] += mib
            self.status[mid] = {"value": "unloaded"}
            return 200, {"success": True}
        return 404, {"error": "no"}

    def loads(self):
        return [m for p, _, m in self.calls if p == "/models/load"]


class Deps:
    def __init__(self, router, cfg):
        self.router, self._cfg = router, cfg

    def cfg(self):
        return config.load()

    def _active_server_bin(self, c=None):
        return "/bin/llama-server"

    def reload_presets(self):
        return self.router("/models/reload", "POST", {})


LIST = "  CUDA0: GPU A (16000 MiB, 1 MiB free)\n  CUDA1: GPU B (16000 MiB, 1 MiB free)\n"


class Base(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.ini = os.path.join(d, "models.ini")
        self.saved_cfg = config.CONFIG
        config.CONFIG = os.path.join(d, "config.json")
        self.addCleanup(setattr, config, "CONFIG", self.saved_cfg)
        config.update({"multi_model": True, "slot_cap": 3, "slot_headroom_mib": 1536})
        self.write_ini("[*]\nctx-size = 8192\n\n[m1]\nmodel = /m/one.gguf\n\n"
                       "[m2]\nmodel = /m/two.gguf\n\n[w1]\nmodel = /m/w.gguf\n")
        self.smi = [{"index": 0, "name": "GPU A", "total_mib": 16 * G, "used_mib": 0,
                     "free_mib": 16 * G},
                    {"index": 1, "name": "GPU B", "total_mib": 16 * G, "used_mib": 0,
                     "free_mib": 16 * G}]
        self.need = {"m1": 4 * G, "m2": 4 * G, "w1": 4 * G}
        self.router = FakeRouter(self.smi, {}, self.ini)
        self.store_path = os.path.join(d, "fp.json")
        self.mgr = self.make()

    def make(self):
        """A SlotManager over this test's router, ini and store (a second one is
        the dashboard restarting under a running router)."""
        mgr = slotctl.SlotManager(Deps(self.router, None), self.store_path)
        mgr.ini_path = lambda: self.ini
        mgr.gpus = lambda: [dict(g) for g in self.smi]
        mgr.list_devices = lambda sbin: LIST
        mgr.ram_free_mib = lambda: 64 * G
        mgr.build_id = lambda sbin: "b1"
        mgr.sleep = lambda s: None
        mgr.pool = lambda: {"models_max": 3, "autoload": False}
        mgr.predict = lambda mid, s: {"need_mib": self.need[mid], "first_gpu_mib": 0,
                                      "ram_mib": 0, "confident": True,
                                      "ctx_from_model": 0 if s.get("ctx-size") else 131072}
        return mgr

    def write_ini(self, text):
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write(text)

    def raw(self):
        with open(self.ini, encoding="utf-8") as f:
            return f.read()

    def section(self, name):
        return config.read_sections(self.ini).get(name, {})

    def use(self, gpu, mib):
        self.smi[gpu]["used_mib"] += mib
        self.smi[gpu]["free_mib"] -= mib


class Load(Base):
    def test_a_worker_goes_to_the_tightest_gpu_and_the_ini_says_so(self):
        self.use(0, 8 * G)
        self.router.cost["m1"] = {0: 4 * G}
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 200, out)
        self.assertEqual(out["devices"], [0])
        sec = self.section("m1")
        self.assertEqual((sec["device"], sec["split-mode"]), ("CUDA0", "none"))
        paths = [p for p, _, _ in self.router.calls]
        self.assertLess(paths.index("/models/reload"), paths.index("/models/load"))

    def test_the_load_is_measured(self):
        self.router.cost["m1"] = {1: 4100, 0: 231}
        self.use(0, 12 * G)                       # no room on GPU0: it goes to GPU1
        self.mgr.load("m1", "main")
        k = self.mgr.key("m1")
        self.assertEqual(self.mgr.store.measured(k), {(1,): {1: 4100, 0: 231}})

    def test_a_refusal_is_409_and_touches_nothing(self):
        self.need["m1"] = 20 * G
        before = self.raw()
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 409)
        self.assertFalse(out["ok"])
        self.assertIn("GiB", out["reason"])
        self.assertEqual(self.raw(), before)
        self.assertEqual(self.router.loads(), [])

    def test_an_unload_during_the_load_spoils_the_reading(self):
        # w1 frees 2 GiB while m1 takes 4: the delta says 2, which is a lie on
        # the unsafe side
        self.router.cost.update(w1={0: 2 * G}, m1={0: 4 * G})
        self.mgr.load("w1", "worker")
        real = self.mgr._wait_loaded

        def meanwhile(mid):
            self.mgr.unload("w1")
            return real(mid)

        self.mgr._wait_loaded = meanwhile
        status, out = self.mgr.load("m1", "main")
        self.assertEqual(status, 200)
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1")), {})
        self.assertEqual(self.mgr.loaded()[0]["footprint"], out["footprint"])

    def test_a_failed_load_is_reported_and_not_measured(self):
        self.router.fail.add("m1")
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 500)
        self.assertIn("exit", out["error"])
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1")), {})

    def test_already_loaded(self):
        self.router.status["m1"] = {"value": "loaded"}
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 200)
        self.assertTrue(out["already"])
        self.assertEqual(self.router.loads(), [])

    def test_the_cap_is_the_routers(self):
        self.mgr.pool = lambda: {"models_max": 2, "autoload": False}
        for m in ("m1", "m2"):
            self.router.status[m] = {"value": "loaded"}
        status, out = self.mgr.load("w1", "worker")
        self.assertEqual(status, 409)
        self.assertIn("limit is 2", out["reason"])

    def test_one_load_at_a_time(self):
        self.router.status["m2"] = {"value": "loading"}
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 409)
        self.assertIn("m2 is still loading", out["reason"])

    def test_the_dashboard_does_not_wait(self):
        started = []

        class Now:
            def __init__(self, target, args, daemon):
                self.run = lambda: target(*args)

            def start(self):
                started.append(1)
                self.run()

        self.router.cost["m1"] = {0: 4 * G}
        with mock.patch.object(slotctl.threading, "Thread", Now):
            status, out = self.mgr.load("m1", "main", wait=False)
        self.assertEqual((status, out["loading"], started), (200, True, [1]))
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1")), {(0,): {0: 4 * G}})
        self.assertIsNone(self.mgr._loading)

    def test_a_model_the_router_does_not_list_is_refused(self):
        v = self.mgr.plan("nope", "worker")
        self.assertFalse(v["ok"])
        self.assertIn("doesn't list", v["reason"])
        status, out = self.mgr.load("nope", "worker")
        self.assertEqual(status, 404)
        self.assertEqual(self.router.loads(), [])

    def test_an_unset_context_says_so(self):
        self.write_ini("[m1]\nmodel = /m/one.gguf\n")      # no ctx-size anywhere
        self.need["m1"] = 40 * G
        v = self.mgr.plan("m1", "worker")
        self.assertFalse(v["ok"])
        self.assertIn("ctx-size isn't set", v["reason"])

    def test_plan_only_looks(self):
        before = self.raw()
        v = self.mgr.plan("m1", "worker")
        self.assertTrue(v["ok"])
        self.assertEqual(self.raw(), before)
        self.assertEqual(self.router.loads(), [])


class Placement(Base):
    def test_user_pins_are_never_rewritten(self):
        self.write_ini("[m1]\nmodel = /m/one.gguf\ndevice = CUDA1\n")
        status, _ = self.mgr.load("m1", "worker")
        self.assertEqual(status, 200)
        self.assertEqual(self.section("m1"), {"model": "/m/one.gguf", "device": "CUDA1"})

    def test_a_pin_in_star_counts(self):
        self.write_ini("[*]\ndevice = CUDA1\n\n[m1]\nmodel = /m/one.gguf\n")
        self.use(0, 8 * G)
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(out["devices"], [1])
        self.assertNotIn("device", self.section("m1"))

    def test_our_placement_is_not_a_user_pin(self):
        self.use(0, 8 * G)
        self.mgr.load("m1", "worker")
        self.mgr.unload("m1")
        self.use(0, -8 * G)
        self.use(1, 8 * G)                       # now GPU1 is the tight one
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(out["devices"], [1])
        self.assertEqual(self.section("m1")["device"], "CUDA1")

    def test_a_user_edit_to_our_keys_makes_them_theirs(self):
        self.use(0, 8 * G)
        self.mgr.load("m1", "worker")
        self.mgr.unload("m1")
        config.set_keys("m1", {"device": "CUDA1"}, self.ini)
        self.mgr.load("m1", "worker")
        self.assertEqual(self.section("m1")["device"], "CUDA1")
        self.assertNotIn("m1", config.load()["slots"]["placed"])

    def test_main_gpu_is_set_aside_and_restored(self):
        self.write_ini("[m1]\nmodel = /m/one.gguf\nmain-gpu = 1 ; the big card\n")
        self.use(1, 8 * G)
        self.mgr.load("m1", "worker")
        self.assertEqual(self.section("m1")["main-gpu"], "0")
        self.mgr.unload("m1")
        self.mgr.unplace_all()
        self.assertEqual(self.section("m1"), {"model": "/m/one.gguf", "main-gpu": "1"})

    def test_the_users_comment_survives_the_round_trip(self):
        text = "[m1]\nmodel = /m/one.gguf\nmain-gpu = 1 ; the big card\nctx-size = 4096\n"
        self.write_ini(text)
        self.use(1, 8 * G)
        self.mgr.load("m1", "worker")
        self.assertEqual(self.section("m1")["main-gpu"], "0")
        self.assertEqual(self.mgr._effective("m1")["main-gpu"], "1")   # still the user's
        self.mgr.unload("m1")
        self.mgr.unplace_all()
        self.assertEqual(self.raw(), text)

    def test_unplace_all_puts_the_file_back(self):
        before = config.read_sections(self.ini)
        self.use(0, 8 * G)
        self.mgr.load("m1", "worker")
        self.mgr.unload("m1")
        self.mgr.unplace_all()
        self.assertEqual(config.read_sections(self.ini), before)
        self.assertEqual(config.load()["slots"]["placed"], {})

    def test_cuda_order_is_translated(self):
        self.mgr.list_devices = lambda s: ("  CUDA0: GPU B (1 MiB, 1 MiB free)\n"
                                           "  CUDA1: GPU A (1 MiB, 1 MiB free)\n")
        self.use(0, 8 * G)                       # nvidia-smi GPU0 = CUDA1 is the tight one
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(out["devices"], [0])
        self.assertEqual(self.section("m1")["device"], "CUDA1")

    def test_unmatched_devices_refuse_rather_than_guess(self):
        self.mgr.list_devices = lambda s: ""
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 409)
        self.assertIn("list-devices", out["reason"])

    def test_one_gpu_needs_no_map(self):
        del self.smi[1]
        self.mgr.list_devices = lambda s: ""
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 200, out)


class Roles(Base):
    def test_main_evicts_workers_only_with_consent(self):
        self.router.cost["w1"] = {0: 10 * G}
        self.router.cost["m1"] = {0: 9 * G}
        self.need.update(w1=10 * G, m1=9 * G)
        self.use(1, 15 * G)                       # GPU1 is busy elsewhere
        self.assertEqual(self.mgr.load("w1", "worker")[0], 200)
        status, out = self.mgr.load("m1", "main")
        self.assertEqual(status, 409)
        self.assertEqual(out["evict"], ["w1"])
        self.assertEqual(self.router.status["w1"]["value"], "loaded")
        status, out = self.mgr.load("m1", "main", evict=True)
        self.assertEqual(status, 200, out)
        self.assertEqual(self.router.status["w1"]["value"], "unloaded")
        self.assertEqual(out["evicted"], ["w1"])

    def test_a_background_load_never_demotes_the_main(self):
        self.mgr.load("m1", "main")
        status, out = self.mgr.load("m1", "worker", keep_role=True)   # an ember, a beat late
        self.assertEqual((status, out["already"]), (200, True))
        self.assertEqual(config.load()["slots"]["main"], "m1")
        self.mgr.load("m1", "worker")                                  # the user asked for it
        self.assertEqual(config.load()["slots"]["main"], "")

    def test_devices_for_the_poll_never_ask_nvidia_smi(self):
        self.router.cost["m1"] = {0: 4 * G}
        self.mgr.load("m1", "main")
        no_smi = mock.patch.object(self.mgr, "gpus", side_effect=AssertionError("nvidia-smi"))
        with no_smi:
            self.assertEqual(self.mgr.devices(), {"m1": [0]})
        self.mgr.unload("m1")                       # an unload does measure what it freed
        with no_smi:
            self.assertEqual(self.mgr.devices(), {})

    def test_footprints_for_the_poll_never_ask_nvidia_smi(self):
        self.router.cost["m1"] = {0: 4 * G}
        self.mgr.load("m1", "main")
        no_smi = mock.patch.object(self.mgr, "gpus", side_effect=AssertionError("nvidia-smi"))
        with no_smi:
            fp = self.mgr.footprints()
        self.assertEqual(list(fp), ["m1"])
        self.assertEqual(set(fp["m1"]), {"0"})          # JSON-ready GPU keys
        self.assertGreater(fp["m1"]["0"], 0)
        self.mgr.unload("m1")
        with no_smi:
            self.assertEqual(self.mgr.footprints(), {})

    def test_one_main_at_a_time(self):
        self.mgr.load("m1", "main")
        self.mgr.load("m2", "main")
        self.assertEqual(config.load()["slots"]["main"], "m2")
        roles = {s["model"]: s["role"] for s in self.mgr.loaded()}
        self.assertEqual(roles, {"m1": "worker", "m2": "main"})

    def test_after_a_restart_the_main_is_still_where_it_was(self):
        self.router.cost["m1"] = {0: 4 * G}
        self.mgr.load("m1", "main")                 # CUDA0, the fastest
        mgr = self.make()                           # the dashboard restarted; the router didn't
        s = {x["model"]: x for x in mgr.loaded()}
        self.assertEqual((s["m1"]["role"], s["m1"]["devices"]), ("main", [0]))
        self.assertEqual(s["m1"]["footprint"], {0: 4 * G})    # measured before the restart
        # GPU0 is the tighter fit, but it's the main's
        self.assertEqual(mgr.plan("m2", "worker")["devices"], [1])

    def test_a_model_loaded_without_placement_spans_every_gpu(self):
        self.router.status["m2"] = {"value": "loaded"}       # single mode, before the pool
        s = {x["model"]: x for x in self.mgr.loaded()}
        self.assertEqual(s["m2"]["devices"], [0, 1])

    def test_unload_records_what_it_frees(self):
        self.router.cost["m1"] = {0: 4 * G}
        self.mgr.load("m1", "main")
        self.router.cost["m1"] = {0: 4 * G + 500}    # the pool grew on a long prompt
        self.use(0, 500)
        self.mgr.unload("m1")
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1")), {(0,): {0: 4 * G + 500}})


# ---------------------------------------------------------------- process slots

IK_HELP = """\
  -c,    --ctx-size N             size of the prompt context
  -ngl,  --gpu-layers N           number of layers to store in VRAM
  -sm,   --split-mode SPLIT_MODE  how to split the model across multiple GPUs
  -dev,   --device dev1,dev2      comma-separated list of devices to use for offloading
  -mg,   --main-gpu i             the GPU to use for the model
         --api-key KEY            API key to use for authorization
         --metrics                enable prometheus compatible metrics endpoint
  -m,    --model FNAME            model path
"""
MAIN_HELP = IK_HELP + "--spec-type [none|draft-mtp]            type of speculative decoding\n"


class FakeProcs:
    """slotproc.Manager's surface; a started process takes its cost from smi."""

    def __init__(self, smi):
        self.smi, self.cost = smi, {}
        self.recs, self.started, self.stopped, self.fail = {}, [], [], set()
        self.rc = {}                           # mid -> the exit code a failed start reports

    def _move(self, mid, sign):
        for g, mib in self.cost.get(mid, {}).items():
            self.smi[g]["used_mib"] += sign * mib
            self.smi[g]["free_mib"] -= sign * mib

    def status(self):
        return {m: {"state": r["state"], "port": r["port"], "pid": 4242,
                    "endpoint": f"http://127.0.0.1:{r['port']}", "exit_code": r["rc"],
                    "bin": r["bin"], "started_at": 1} for m, r in self.recs.items()}

    def has(self, mid):
        return mid in self.recs

    def start(self, mid, argv, port_base=8100):
        self.started.append((mid, list(argv)))
        failed = mid in self.fail
        port = port_base + len(self.recs)
        self.recs[mid] = {"state": "failed" if failed else "ready", "port": port,
                          "bin": argv[0], "rc": self.rc.get(mid, 1) if failed else None}
        if not failed:
            self._move(mid, +1)
        return True, "", port

    def stop(self, mid, timeout=15):
        self.stopped.append(mid)
        r = self.recs.pop(mid, None)
        if r and r["state"] == "ready":
            self._move(mid, -1)
        return True, ""


class ProcBase(Base):
    def setUp(self):
        super().setUp()
        self.procs = FakeProcs(self.smi)
        self.mgr._d.PROCS = self.procs        # make() runs inside super().setUp()
        self.pins = {"m1": "/bin/ik/llama-server"}
        self.mgr.build_id = lambda sbin: "ik" if "ik" in sbin else "b1"

    def make(self):
        mgr = super().make()
        # a pin starting with "!" stands for a pin that can't be honoured
        mgr._d.pinned_bin = lambda mid: (self.pins.get(mid), "") if not str(
            self.pins.get(mid, "")).startswith("!") else (None, self.pins[mid][1:])
        mgr.help_items = lambda sbin: (argspec.parse_help(IK_HELP) if "ik" in sbin
                                       else argspec.parse_help(MAIN_HELP))
        return mgr


class ProcSlots(ProcBase):
    def test_a_pinned_model_runs_as_its_own_process(self):
        self.use(0, 8 * G)
        self.procs.cost["m1"] = {0: 4 * G}
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 200, out)
        self.assertEqual(self.router.loads(), [])
        (mid, argv), = self.procs.started
        self.assertEqual(argv[:3], ["/bin/ik/llama-server", "-m", "/m/one.gguf"])
        # placed like a router slot, and the process is told so
        self.assertEqual(self.section("m1")["device"], "CUDA0")
        self.assertEqual(argv[argv.index("--device") + 1], "CUDA0")
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "8192")      # from [*]
        self.assertEqual(out["process"]["port"], 8100)

    def test_its_footprint_is_keyed_by_its_own_build(self):
        self.procs.cost["m1"] = {1: 4100}
        self.use(0, 12 * G)
        self.mgr.load("m1", "main")
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1", sbin="/bin/ik/llama-server")),
                         {(1,): {1: 4100}})
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1")), {})

    def test_the_planner_counts_process_slots(self):
        self.procs.cost["m1"] = {0: 4 * G}
        self.mgr.load("m1", "main")
        self.assertEqual([(r["model"], r["devices"]) for r in self.mgr.loaded()], [("m1", [0])])
        status, _ = self.mgr.load("m1", "main")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.procs.started), 1)              # already up

    def test_what_the_build_lacks_is_reported(self):
        self.write_ini("[*]\nctx-size = 8192\n\n[m1]\nmodel = /m/one.gguf\nspec-type = draft-mtp\n")
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 200, out)
        self.assertEqual(out["dropped"], ["spec-type"])
        self.assertNotIn("--spec-type", self.procs.started[0][1])

    def test_a_process_that_exits_is_a_failed_load(self):
        self.procs.fail.add("m1")
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 500)
        self.assertIn("exit", out["error"])
        self.assertIn("log", out["error"])
        self.assertEqual(self.mgr.store.measured(self.mgr.key("m1", sbin="/bin/ik/llama-server")), {})

    def test_a_crash_is_named(self):
        """An ik built before PQ2_0 dies dividing by zero, with nothing in its log."""
        self.procs.fail.add("m1")
        self.procs.rc["m1"] = 0xC0000094
        status, out = self.mgr.load("m1", "worker")
        self.assertIn("integer divide by zero", out["error"])

    def test_unload_stops_the_process(self):
        self.procs.cost["m1"] = {0: 4 * G}
        self.mgr.load("m1", "main")
        status, out = self.mgr.unload("m1")
        self.assertEqual(status, 200)
        self.assertEqual(self.procs.stopped, ["m1"])
        self.assertNotIn(("/models/unload", "POST", "m1"), self.router.calls)
        self.assertEqual(out["freed"], {0: 4 * G})
        self.assertEqual(self.mgr.loaded(), [])

    def test_a_pin_that_cannot_be_honoured_refuses(self):
        self.pins["m1"] = "!build b6000 is no longer installed"
        status, out = self.mgr.load("m1", "worker")
        self.assertEqual(status, 409)
        self.assertIn("b6000", out["reason"])
        self.assertEqual((self.procs.started, self.router.loads()), ([], []))
        self.assertIn("b6000", self.mgr.plan("m1")["reason"])

    def test_unpinned_models_still_go_to_the_router(self):
        status, _ = self.mgr.load("m2", "worker")
        self.assertEqual(status, 200)
        self.assertEqual(self.router.loads(), ["m2"])
        self.assertEqual(self.procs.started, [])

    def test_single_model_mode_swaps_like_the_router_does(self):
        """No pool: one model at a time, the ini as the user wrote it."""
        self.mgr.pool = lambda: None
        self.router.status["m2"] = {"value": "loaded"}
        before = self.raw()
        status, out = self.mgr.load("m1", "main")
        self.assertEqual(status, 200, out)
        self.assertIn(("/models/unload", "POST", "m2"), self.router.calls)
        self.assertEqual(out["evicted"], ["m2"])
        self.assertEqual(self.raw(), before)                       # nothing placed
        self.assertNotIn("--device", self.procs.started[0][1])


if __name__ == "__main__":
    unittest.main()
