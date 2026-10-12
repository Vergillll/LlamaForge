"""The per-load facts the editor shows beside the context: per-slot context and
slot count, read from the loaded instance's /props once per load."""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import routes

PROPS = {"default_generation_settings": {"n_ctx": 64000}, "total_slots": 1}


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        routes._RUNTIME.clear()
        routes._RUNTIME_LIVE.clear()
        self.calls = []

    def _router(self, answer):
        def fake(path, method="GET", body=None, timeout=30):
            self.calls.append(path)
            return answer(path)
        return mock.patch.object(routes, "router", fake)

    def test_a_loaded_model_is_read_once_per_load(self):
        with self._router(lambda path: (200, PROPS)):
            for _ in range(3):                 # three polls
                rt = routes._runtime("m", "loaded")
        self.assertEqual(rt, {"n_ctx_slot": 64000, "n_slots": 1})
        self.assertEqual(self.calls, ["/props?model=m&autoload=false"])

    def test_props_never_ask_the_router_to_load(self):
        with self._router(lambda path: (200, PROPS)):
            routes._runtime("a b/c", "loaded")
        self.assertEqual(self.calls, ["/props?model=a%20b%2Fc&autoload=false"])

    def test_the_last_load_is_kept_after_unload_and_read_again_next_load(self):
        answers = iter([(200, PROPS),
                        (200, {"default_generation_settings": {"n_ctx": 15360}, "total_slots": 4})])
        with self._router(lambda path: next(answers)):
            routes._runtime("m", "loaded")
            self.assertEqual(routes._runtime("m", "unloaded")["n_ctx_slot"], 64000)
            rt = routes._runtime("m", "loaded")
        self.assertEqual(rt, {"n_ctx_slot": 15360, "n_slots": 4})
        self.assertEqual(len(self.calls), 2)

    def test_an_unloaded_model_is_not_asked(self):
        with self._router(lambda path: (200, PROPS)):
            self.assertIsNone(routes._runtime("m", "unloaded"))
            self.assertIsNone(routes._runtime("m", "loading"))
        self.assertEqual(self.calls, [])

    def test_a_failed_or_odd_answer_leaves_nothing_and_is_retried(self):
        for bad in [(404, {"error": "not found"}), (599, {"error": "refused"}),
                    (200, {"default_generation_settings": {"n_ctx": 0}}),
                    (200, {"default_generation_settings": None}), (200, [])]:
            with self._router(lambda path, bad=bad: bad):
                self.assertIsNone(routes._runtime("m", "loaded"), bad)
        self.assertEqual(len(self.calls), 5)            # each poll tries again


class ModelStateRuntimeTest(unittest.TestCase):
    def setUp(self):
        routes._RUNTIME.clear()
        routes._RUNTIME_LIVE.clear()

    def _state(self, rows):
        def fake(path, method="GET", body=None, timeout=30):
            if path == "/models":
                return 200, {"data": rows}
            if path.startswith("/props?model="):
                return 200, PROPS
            return 404, {}
        with mock.patch.object(routes, "router", fake), \
             mock.patch.object(routes.config, "read_sections",
                               return_value={"m": {}, "idle": {}}), \
             mock.patch.object(routes.config, "vision_parked", return_value=set()), \
             mock.patch.object(routes, "_overlay_procs"), \
             mock.patch.object(routes, "_file_gib", return_value=None), \
             mock.patch.object(routes.LOADPROG, "attach"):
            return {m["id"]: m for m in routes.model_state()["models"]}

    def test_a_loaded_row_carries_runtime_an_idle_one_does_not(self):
        rows = self._state([{"id": "m", "status": {"value": "loaded"}},
                            {"id": "idle", "status": {"value": "unloaded"}}])
        self.assertEqual(rows["m"]["runtime"], {"n_ctx_slot": 64000, "n_slots": 1})
        self.assertNotIn("runtime", rows["idle"])


if __name__ == "__main__":
    unittest.main()
