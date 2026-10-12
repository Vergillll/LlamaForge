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


if __name__ == "__main__":
    unittest.main()
