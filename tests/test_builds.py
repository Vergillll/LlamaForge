"""Per-model build pins (config.json model_builds)."""
import conftest_paths  # noqa: F401
import os
import shutil
import tempfile
import unittest

import builds


class BuildsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        def exe(*parts):
            p = os.path.join(self.tmp, *parts)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
            return p

        self.router = exe("llama.cpp", "build", "bin", "llama-server.exe")
        self.ik = exe("ik", "llama-server.exe")
        self.b8000 = exe("engines", "b8000-cuda", "llama-server.exe")
        self.b7000 = exe("engines", "b7000-cuda", "llama-server.exe")
        self.installs = [
            {"dir": os.path.dirname(self.b8000), "server_bin": self.b8000,
             "label": "b8000 (CUDA 12)", "tag": "b8000"},
            {"dir": os.path.dirname(self.b7000), "server_bin": self.b7000, "tag": "b7000"},
            {"dir": os.path.join(self.tmp, "engines", "broken"), "server_bin": None},
        ]
        self.cfg = {"server_bin": self.router, "active_engine": "llamacpp",
                    "ik_llama_server_bin": self.ik, "model_builds": {}}

    def pin(self, **pins):
        return dict(self.cfg, model_builds=pins)

    # ---- options
    def test_options_list_installs_and_ik(self):
        opts = builds.options(self.cfg, self.installs)
        self.assertEqual([o["ref"] for o in opts], ["b8000-cuda", "b7000-cuda", builds.IK])
        self.assertEqual(opts[0]["label"], "b8000 (CUDA 12)")
        self.assertEqual(opts[1]["label"], "b7000")
        self.assertEqual(opts[2]["label"], "ik_llama.cpp")
        self.assertFalse(any(o["router"] for o in opts))

    def test_the_build_the_router_runs_is_marked(self):
        cfg = dict(self.cfg, server_bin=self.b8000)
        opts = {o["ref"]: o for o in builds.options(cfg, self.installs)}
        self.assertTrue(opts["b8000-cuda"]["router"])
        self.assertFalse(opts["b7000-cuda"]["router"])

    def test_ik_is_offered_only_when_its_binary_exists(self):
        cfg = dict(self.cfg, ik_llama_server_bin=os.path.join(self.tmp, "nope.exe"))
        self.assertNotIn(builds.IK, [o["ref"] for o in builds.options(cfg, self.installs)])
        cfg = dict(self.cfg, ik_llama_server_bin="")
        self.assertNotIn(builds.IK, [o["ref"] for o in builds.options(cfg, self.installs)])

    # ---- resolve / pinned_bin
    def test_resolve(self):
        self.assertEqual(builds.resolve("b7000-cuda", self.cfg, self.installs), (self.b7000, ""))
        self.assertEqual(builds.resolve(builds.IK, self.cfg, self.installs), (self.ik, ""))

    def test_resolve_names_what_is_missing(self):
        sbin, err = builds.resolve("b6000-cuda", self.cfg, self.installs)
        self.assertIsNone(sbin)
        self.assertIn("b6000-cuda", err)
        self.assertIn("Build / Update", err)
        sbin, err = builds.resolve("broken", self.cfg, self.installs)
        self.assertIsNone(sbin)
        sbin, err = builds.resolve(builds.IK, dict(self.cfg, ik_llama_server_bin=""), self.installs)
        self.assertIsNone(sbin)
        self.assertIn("ik_llama", err)

    def test_unpinned_models_belong_to_the_router(self):
        self.assertEqual(builds.pinned_bin("qwen", self.cfg, self.installs), (None, ""))
        self.assertEqual(builds.pinned_bin("qwen", dict(self.cfg, model_builds=None),
                                           self.installs), (None, ""))

    def test_a_pin_gets_its_own_binary(self):
        cfg = self.pin(qwen=builds.IK, gemma="b7000-cuda")
        self.assertEqual(builds.pinned_bin("qwen", cfg, self.installs), (self.ik, ""))
        self.assertEqual(builds.pinned_bin("gemma", cfg, self.installs), (self.b7000, ""))

    def test_a_pin_to_the_routers_own_build_is_no_pin(self):
        cfg = dict(self.pin(qwen="b8000-cuda"), server_bin=self.b8000)
        self.assertEqual(builds.pinned_bin("qwen", cfg, self.installs), (None, ""))

    def test_a_dangling_pin_is_an_error_not_a_silent_router_load(self):
        """Loading it on the router instead could be the very build that can't
        read the file (the reason it was pinned)."""
        sbin, err = builds.pinned_bin("qwen", self.pin(qwen="b6000-cuda"), self.installs)
        self.assertIsNone(sbin)
        self.assertTrue(err)

    # ---- validate / protect
    def test_source_dir_is_the_checkout_of_the_build_that_loads_the_model(self):
        cfg = dict(self.cfg, llama_src=os.path.join(self.tmp, "llama.cpp"),
                   ik_llama_src=os.path.join(self.tmp, "ik-src"))
        self.assertEqual(builds.source_dir("m", cfg, self.installs), cfg["llama_src"])
        pinned = dict(cfg, model_builds={"m": builds.IK})
        self.assertEqual(builds.source_dir("m", pinned, self.installs), cfg["ik_llama_src"])
        ik_router = dict(cfg, active_engine="ikllama")
        self.assertEqual(builds.source_dir("m", ik_router, self.installs), cfg["ik_llama_src"])

    def test_no_source_dir_for_a_prebuilt_or_a_broken_pin(self):
        cfg = dict(self.cfg, llama_src=os.path.join(self.tmp, "llama.cpp"))
        prebuilt = dict(cfg, model_builds={"m": "b8000-cuda"})
        self.assertEqual(builds.source_dir("m", prebuilt, self.installs), "")
        broken = dict(cfg, model_builds={"m": "gone"})
        self.assertEqual(builds.source_dir("m", broken, self.installs), "")
        elsewhere = dict(cfg, server_bin=self.b7000)
        self.assertEqual(builds.source_dir("m", elsewhere, self.installs), "")

    def test_validate_ref(self):
        self.assertEqual(builds.validate("b7000-cuda", self.cfg, self.installs), "")
        self.assertEqual(builds.validate("", self.cfg, self.installs), "")
        self.assertTrue(builds.validate("../etc", self.cfg, self.installs))
        self.assertTrue(builds.validate("b6000-cuda", self.cfg, self.installs))
        self.assertTrue(builds.validate(7, self.cfg, self.installs))

    def test_pinned_installs_are_protected_from_pruning(self):
        cfg = self.pin(qwen="b7000-cuda", gemma=builds.IK, old="gone")
        self.assertEqual(builds.protected_dirs(cfg, self.installs),
                         [os.path.dirname(self.b7000)])


if __name__ == "__main__":
    unittest.main()
