"""Elastic VRAM wiring: config, the swap order, the ini record, the routes."""
import conftest_paths  # noqa: F401
import os
import shutil
import tempfile
import unittest
from unittest import mock

import config


class TempConfig(unittest.TestCase):
    def setUp(self):
        self._saved = config.CONFIG
        self._dir = tempfile.mkdtemp()
        config.CONFIG = os.path.join(self._dir, "config.json")
        self.ini = os.path.join(self._dir, "models.ini")
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write("[*]\nctx-size = 8192\n\n[big]\nmodel = D:/m/big.gguf\nn-gpu-layers = 99\n")
        config.update({"models_ini": self.ini})

    def tearDown(self):
        config.CONFIG = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)


class ConfigTest(TempConfig):
    def test_defaults(self):
        c = config.load()
        self.assertEqual((c["elastic_vram"], c["elastic_grow"], c["elastic_keep_conversation"],
                          c["elastic_headroom_mib"], c["elastic_placed"]),
                         (False, True, True, 1536, {}))

    def test_settings_are_writable_and_public(self):
        import routes
        code, out = routes.post_config(routes.Req({
            "elastic_vram": True, "elastic_grow": False, "elastic_keep_conversation": False,
            "elastic_headroom_mib": 2048}))
        self.assertEqual(code, 200)
        self.assertNotIn("rejected", out)
        self.assertEqual(out["config"]["elastic_headroom_mib"], 2048)
        self.assertIs(out["config"]["elastic_grow"], False)
        self.assertIs(out["config"]["elastic_vram"], True)
        self.assertNotIn("elastic_placed", out["config"])

    # post_config names refused keys in out["rejected"] when something else
    # was accepted, and raises ApiError(400) naming them when nothing was.
    def test_the_record_is_not_writable_from_the_browser(self):
        import routes
        _, out = routes.post_config(routes.Req({"elastic_placed": {"big": {}}, "elastic_vram": True}))
        self.assertIn("elastic_placed", out.get("rejected", []))
        with self.assertRaises(routes.ApiError) as cm:
            routes.post_config(routes.Req({"elastic_placed": {"big": {}}}))
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("elastic_placed", cm.exception.message)
        self.assertEqual(config.load()["elastic_placed"], {})

    def test_headroom_bounds(self):
        import routes
        for bad in (10, 20000, True, "2048"):
            _, out = routes.post_config(routes.Req({"elastic_headroom_mib": bad, "elastic_vram": True}))
            self.assertIn("elastic_headroom_mib", out.get("rejected", []), bad)
        self.assertEqual(config.load()["elastic_headroom_mib"], 1536)


STEP48 = {"keys": {"n-gpu-layers": "48"}, "gpu_layers": 48, "cpu_moe": 0, "layers": 64}
TOP = {"keys": {}, "gpu_layers": 64, "cpu_moe": 0, "layers": 64}


def write_ini(path, big_section):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[*]\nctx-size = 8192\n\n[big]\nmodel = D:/m/big.gguf\n" + big_section)


class SwapTest(TempConfig):
    def run_swap(self, step, load_error=None):
        import routes
        order = []

        def reload(mid, clean):
            order.append(("ini+reload", mid, clean))
            config.set_keys(mid, clean)
            return True

        def load(mid, role):
            order.append(("load", mid))
            if load_error:
                raise RuntimeError(load_error)
        snaps = iter([{0: {"used": 3000, "total": 16000}}, {0: {"used": 12000, "total": 16000}}])
        with mock.patch.object(routes, "_apply_knobs_and_reload", reload), \
             mock.patch.object(routes, "_load_and_wait", load), \
             mock.patch.object(routes, "_slots_on", return_value=False), \
             mock.patch.object(routes, "_el_snapshot", lambda: next(snaps)), \
             mock.patch.object(routes.SLOTS.store, "record") as rec:
            err = routes._el_swap("big", step)
        return err, order, rec

    def test_shrink_writes_the_step_reloads_loads_and_records(self):
        err, order, rec = self.run_swap(STEP48)
        self.assertEqual(err, "")
        self.assertEqual(order, [("ini+reload", "big", {"n-gpu-layers": "48"}), ("load", "big")])
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "48")
        self.assertEqual(config.load()["elastic_placed"],
                         {"big": {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}}})
        k, devices, fp = rec.call_args.args[:3]
        self.assertEqual((devices, fp), ([0], {0: 9000}))

    def test_back_at_the_top_puts_the_user_value_back(self):
        self.run_swap(STEP48)
        self.run_swap(TOP)
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "99")
        self.assertEqual(config.load()["elastic_placed"], {})

    def test_a_failed_load_says_why(self):
        err, _, rec = self.run_swap(STEP48, load_error="CUDA out of memory")
        self.assertEqual(err, "CUDA out of memory")
        rec.assert_not_called()

    def test_giveup_restores_the_keys(self):
        import routes
        self.run_swap(STEP48)
        with mock.patch.object(routes, "_apply_knobs_and_reload",
                               lambda mid, clean: config.set_keys(mid, clean)):
            routes._el_giveup("big")
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "99")
        self.assertEqual(config.load()["elastic_placed"], {})

    def test_an_aliased_section_keeps_its_spelling_and_footprint_key(self):
        import routes, footprint
        write_ini(self.ini, "gpu-layers = 99\n")
        err, order, rec = self.run_swap(STEP48)
        self.assertEqual(err, "")
        self.assertEqual(order[0], ("ini+reload", "big", {"gpu-layers": "48"}))
        big = config.read_sections()["big"]
        self.assertEqual(big.get("gpu-layers"), "48")
        self.assertNotIn("n-gpu-layers", big)
        self.assertEqual(config.load()["elastic_placed"],
                         {"big": {"keys": {"gpu-layers": "48"}, "was": {"gpu-layers": "99"}}})
        # the same identity the pool computes from the section as it now reads
        want = footprint.key("big", routes.SLOTS.build_id(routes._active_server_bin()),
                             {"ctx-size": "8192", "model": "D:/m/big.gguf", "gpu-layers": "48"})
        self.assertEqual(rec.call_args.args[0], want)
        self.run_swap(TOP)
        self.assertEqual(config.read_sections()["big"]["gpu-layers"], "99")
        self.assertEqual(config.load()["elastic_placed"], {})

    def test_the_record_lands_before_the_ini_so_a_failed_reload_is_still_undoable(self):
        import routes

        def reload_then_die(mid, clean):
            config.set_keys(mid, clean)
            raise OSError("router went away")
        with mock.patch.object(routes, "_apply_knobs_and_reload", reload_then_die):
            with self.assertRaises(OSError):
                routes._el_swap("big", STEP48)
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "48")
        self.assertEqual(config.load()["elastic_placed"],
                         {"big": {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}}})

    def test_a_grow_that_dies_before_writing_keeps_the_record(self):
        import routes
        self.run_swap(STEP48)
        with mock.patch.object(routes, "_apply_knobs_and_reload",
                               side_effect=OSError("router went away")):
            with self.assertRaises(OSError):
                routes._el_swap("big", TOP)
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "48")
        self.assertIn("big", config.load()["elastic_placed"])


class TidyTest(TempConfig):
    def test_keys_of_a_model_that_is_no_longer_loaded_go_back(self):
        import routes
        config.set_keys("big", {"n-gpu-layers": "48"})
        config.update({"elastic_placed": {"big": {"keys": {"n-gpu-layers": "48"},
                                                  "was": {"n-gpu-layers": "99"}}}})
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append(path)
            return 200, {"data": [{"id": "big", "status": {"value": "unloaded"}}]}
        with mock.patch.object(routes, "router", fake_router):
            routes._el_tidy()
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "99")
        self.assertEqual(config.load()["elastic_placed"], {})
        self.assertIn("/models?reload=1", calls)

    def test_a_loaded_model_is_left_alone(self):
        import routes
        config.set_keys("big", {"n-gpu-layers": "48"})
        placed = {"big": {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}}}
        config.update({"elastic_placed": placed})
        with mock.patch.object(routes, "router", lambda *a, **k: (
                200, {"data": [{"id": "big", "status": {"value": "loaded"}}]})):
            routes._el_tidy()
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "48")
        self.assertEqual(config.load()["elastic_placed"], placed)

    def test_a_router_that_does_not_answer_changes_nothing(self):
        import routes
        config.update({"elastic_placed": {"big": {"keys": {"n-gpu-layers": "48"},
                                                  "was": {"n-gpu-layers": "99"}}}})
        with mock.patch.object(routes, "router", lambda *a, **k: (599, {"error": "refused"})):
            routes._el_tidy()
        self.assertIn("big", config.load()["elastic_placed"])

    def test_a_failure_midway_still_reloads_and_the_next_tidy_finishes(self):
        import routes
        with open(self.ini, "a", encoding="utf-8") as f:
            f.write("\n[small]\nmodel = D:/m/small.gguf\nn-gpu-layers = 40\n")
        config.set_keys("big", {"n-gpu-layers": "48"})
        config.set_keys("small", {"n-gpu-layers": "20"})
        config.update({"elastic_placed": {
            "big": {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}},
            "small": {"keys": {"n-gpu-layers": "20"}, "was": {"n-gpu-layers": "40"}}}})
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append(path)
            return 200, {"data": []}
        real = config.set_keys

        def flaky(mid, updates, *a, **k):
            if mid == "small":
                raise OSError("ini locked")
            return real(mid, updates, *a, **k)
        with mock.patch.object(routes, "router", fake_router), \
             mock.patch.object(config, "set_keys", flaky):
            with self.assertRaises(OSError):
                routes._el_tidy()
        self.assertEqual(config.read_sections()["big"]["n-gpu-layers"], "99")
        self.assertIn("/models?reload=1", calls)          # what was written is picked up
        self.assertEqual(list(config.load()["elastic_placed"]), ["small"])
        with mock.patch.object(routes, "router", fake_router):
            routes._el_tidy()
        self.assertEqual(config.read_sections()["small"]["n-gpu-layers"], "40")
        self.assertEqual(config.load()["elastic_placed"], {})


class ModelsTest(TempConfig):
    LAYOUT = {"arch": "x", "kv": {"block_count": 64}, "tensors": []}

    def models(self):
        import routes
        with mock.patch.object(routes, "_router_loaded_ids", return_value=["big"]), \
             mock.patch.object(routes.PROCS, "has", return_value=False), \
             mock.patch.object(routes, "_el_layout", return_value=self.LAYOUT), \
             mock.patch.object(routes.footprint, "predict", return_value={"need_mib": 8000}), \
             mock.patch.object(routes.SLOTS.store, "measured", return_value={}) as measured:
            return routes._el_models([0]), measured

    def test_a_loaded_model_gets_its_ladder_and_current_step(self):
        config.update({"elastic_vram": True,
                       "elastic_placed": {"big": {"keys": {"n-gpu-layers": "48"},
                                                  "was": {"n-gpu-layers": "99"}}}})
        config.set_keys("big", {"n-gpu-layers": "48"})
        ms, _ = self.models()
        self.assertEqual([m["id"] for m in ms], ["big"])
        self.assertEqual(ms[0]["at"], 1)                        # 48/64: the user's 99 is the top
        self.assertEqual(len(ms[0]["fps"]), len(ms[0]["steps"]))

    def test_an_aliased_section_finds_its_step(self):
        import routes, footprint
        write_ini(self.ini, "gpu-layers = 48\n")
        config.update({"elastic_vram": True,
                       "elastic_placed": {"big": {"keys": {"gpu-layers": "48"},
                                                  "was": {"gpu-layers": "99"}}}})
        ms, measured = self.models()
        self.assertEqual(ms[0]["steps"][0]["gpu_layers"], 64)  # the user's 99, capped
        self.assertEqual(ms[0]["at"], 1)
        build = routes.SLOTS.build_id(routes._active_server_bin())
        want = footprint.key("big", build, {"ctx-size": "8192", "model": "D:/m/big.gguf",
                                            "gpu-layers": "48"})
        self.assertEqual(measured.call_args_list[1].args[0], want)

    def test_off_with_nothing_placed_costs_nothing(self):
        import routes
        with mock.patch.object(routes, "_router_loaded_ids",
                               side_effect=AssertionError("asked the router while off")):
            self.assertEqual(routes._el_models([0]), [])


class SnapshotTest(unittest.TestCase):
    def test_gpus_by_index_in_mib_and_errors_skipped(self):
        import routes
        tele = [{"index": 0, "name": "a", "used": 3000, "total": 16303, "util": 1, "temp": 40},
                {"index": 1, "name": "b", "used": 500, "total": 16311, "util": 0, "temp": 35}]
        with mock.patch.object(routes.osplat, "IS_MAC", False), \
             mock.patch.object(routes, "_gpu_telemetry", return_value=tele):
            self.assertEqual(routes._el_snapshot(), {0: {"used": 3000, "total": 16303},
                                                     1: {"used": 500, "total": 16311}})
        with mock.patch.object(routes.osplat, "IS_MAC", False), \
             mock.patch.object(routes, "_gpu_telemetry", return_value=[{"error": "no smi"}]):
            self.assertEqual(routes._el_snapshot(), {})


class RouteTest(TempConfig):
    def test_routes_are_registered(self):
        import routes
        self.assertIs(routes.GET_ROUTES["/api/elastic"], routes.get_elastic)
        self.assertIs(routes.POST_ROUTES["/api/elastic/grow"], routes.post_elastic_grow)

    def test_grow_refuses_a_model_that_is_not_resized(self):
        import routes
        code, out = routes.post_elastic_grow(routes.Req({"model": "big"}))
        self.assertEqual(code, 400)
        self.assertFalse(out["ok"])

    def test_state_carries_the_status(self):
        import routes
        self.assertIn("events", routes.ELASTIC.status())
        with mock.patch.object(routes, "REGISTRY") as reg, \
             mock.patch.object(routes._GPU_TELEMETRY, "get", return_value=[]), \
             mock.patch.object(routes, "_slots_state", return_value={}):
            reg.state.return_value = {"models": []}
            reg.enabled.return_value = []
            _, s = routes.get_state(routes.Req())
        self.assertEqual(s["elastic"], routes.ELASTIC.status())


if __name__ == "__main__":
    unittest.main()
