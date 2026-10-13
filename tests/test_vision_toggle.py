"""Vision on/off per model without deleting the projector file (#86)."""
import conftest_paths  # noqa: F401
import os, tempfile, unittest
from unittest import mock

import config, routes


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.ini = os.path.join(self.tmp, "models.ini")
        self.proj = os.path.join(self.tmp, "mmproj-F16.gguf")
        open(self.proj, "wb").close()
        cfg = config.load()
        cfg["models_ini"] = self.ini
        config.save(cfg)
        config.set_keys("gemma", {"model": "g.gguf", "mmproj": self.proj, "ctx-size": "8192"})
        config.set_keys("plain", {"model": "p.gguf"})

    def tearDown(self):
        config.CONFIG = self._saved

    def sec(self, mid):
        return config.read_sections(self.ini).get(mid, {})


class StorageTest(_Case):
    def test_off_parks_the_line_and_keeps_the_file(self):
        self.assertEqual(config.set_vision("gemma", False), {"vision": False, "mmproj": self.proj})
        self.assertNotIn("mmproj", self.sec("gemma"))
        self.assertEqual(self.sec("gemma")["ctx-size"], "8192")      # nothing else touched
        self.assertTrue(os.path.exists(self.proj))
        self.assertEqual(config.vision_parked(), {"gemma": self.proj})

    def test_on_puts_it_back(self):
        config.set_vision("gemma", False)
        self.assertEqual(config.set_vision("gemma", True), {"vision": True, "mmproj": self.proj})
        self.assertEqual(self.sec("gemma")["mmproj"], self.proj)
        self.assertEqual(config.vision_parked(), {})

    def test_off_twice_is_harmless(self):
        config.set_vision("gemma", False)
        self.assertEqual(config.set_vision("gemma", False)["mmproj"], self.proj)
        self.assertEqual(config.vision_parked(), {"gemma": self.proj})

    def test_a_model_without_a_projector_cannot_be_turned_off(self):
        with self.assertRaises(ValueError):
            config.set_vision("plain", False)

    def test_on_refuses_when_the_file_is_gone(self):
        config.set_vision("gemma", False)
        os.remove(self.proj)
        with self.assertRaises(ValueError):
            config.set_vision("gemma", True)
        self.assertNotIn("mmproj", self.sec("gemma"))
        self.assertEqual(config.vision_parked(), {"gemma": self.proj})   # still parked

    def test_on_for_a_model_with_vision_already_on_is_a_no_op(self):
        self.assertEqual(config.set_vision("gemma", True)["vision"], True)
        self.assertEqual(self.sec("gemma")["mmproj"], self.proj)

    def test_prune_forgets_it(self):
        config.set_vision("gemma", False)
        config.prune_vision("gemma")
        self.assertEqual(config.vision_parked(), {})

    def test_parking_is_per_engine(self):
        config.set_vision("gemma", False)
        self.assertEqual(config.vision_parked("ikllama"), {})


class ScanRespectsParkingTest(_Case):
    def test_a_rescan_does_not_turn_vision_back_on(self):
        config.set_vision("gemma", False)
        newer = os.path.join(self.tmp, "mmproj-BF16.gguf")
        routes._persist_scanned_entries([{"id": "gemma", "model": "g.gguf", "mmproj": newer}])
        self.assertNotIn("mmproj", self.sec("gemma"))
        self.assertEqual(config.vision_parked(), {"gemma": newer})   # parked path follows the scan

    def test_an_unparked_model_still_gets_its_projector(self):
        routes._persist_scanned_entries([{"id": "plain", "model": "p.gguf", "mmproj": self.proj}])
        self.assertEqual(self.sec("plain")["mmproj"], self.proj)


class RouteTest(_Case):
    def _post(self, body):
        req = mock.Mock(body=body)
        with mock.patch.object(routes, "_model_status", return_value={"value": "unloaded"}), \
             mock.patch.object(routes, "router", return_value=(200, {})) as r:
            out = routes.post_model_vision(req)
        return out, r

    def test_off_then_on(self):
        (code, out), r = self._post({"model": "gemma", "on": False})
        self.assertEqual(code, 200)
        self.assertEqual(out["vision"], False)
        self.assertFalse(out["was_running"])
        r.assert_any_call("/models/reload", "POST", {})
        self.assertNotIn("mmproj", self.sec("gemma"))
        (code, out), _ = self._post({"model": "gemma", "on": True})
        self.assertTrue(out["vision"])
        self.assertEqual(self.sec("gemma")["mmproj"], self.proj)

    def test_bad_requests(self):
        with self.assertRaises(routes.ApiError) as e:
            self._post({"model": "plain", "on": False})
        self.assertEqual(e.exception.status, 400)
        with self.assertRaises(routes.ApiError):
            self._post({"on": False})
        with self.assertRaises(routes.ApiError):
            self._post({"model": "gemma", "on": "yes"})


if __name__ == "__main__":
    unittest.main()
