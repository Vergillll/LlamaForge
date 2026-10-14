"""Which llama.cpp family can load a GGUF, from its tensor type ids."""
import conftest_paths  # noqa: F401
import os
import shutil
import tempfile
import unittest
from unittest import mock

import compat
import diag
from test_gguf_layout import kv_str, write_gguf

# The real Ternary-Bonsai-2-27B-PQ2_0.gguf: F32, BF16 and PrismML's PQ2_0.
BONSAI = {0, 30, 142}


class Classify(unittest.TestCase):
    def test_common_quants_run_anywhere(self):
        self.assertEqual(compat.classify({0, 1, 8, 12, 14}), "any")

    def test_mxfp4_is_shared(self):
        # gpt-oss / gemma mxfp4 builds: ik added 39 "so we are compatible with mainline"
        self.assertEqual(compat.classify({0, 39}), "any")

    def test_ik_quants(self):
        self.assertEqual(compat.classify(BONSAI), "ik-only")
        self.assertEqual(compat.classify({0, 12, 139}), "ik-only")       # IQ4_K
        self.assertEqual(compat.classify({0, 212}), "ik-only")           # Q4_K_R4
        self.assertEqual(compat.classify({0, 134}), "ik-only")           # IQ1_BN

    def test_ids_mainline_dropped_but_ik_kept(self):
        self.assertEqual(compat.classify({0, 31}), "ik-only")            # Q4_0_4_4
        self.assertEqual(compat.classify({0, 36}), "ik-only")            # ik's I2_S

    def test_mainline_quants(self):
        self.assertEqual(compat.classify({0, 34}), "mainline-only")      # TQ1_0
        self.assertEqual(compat.classify({0, 35}), "mainline-only")      # TQ2_0
        self.assertEqual(compat.classify({0, 40}), "mainline-only")      # NVFP4
        self.assertEqual(compat.classify({0, 42}), "mainline-only")      # Q2_0

    def test_one_id_two_meanings_is_unknown(self):
        # mainline Q1_0 and ik Q1_0_G128 share 41; don't guess which this is
        self.assertEqual(compat.classify({0, 41}), "unknown")

    def test_nobody_knows_it(self):
        self.assertEqual(compat.classify({0, 500}), "unknown")
        self.assertEqual(compat.classify({0, 142, 40}), "unknown")       # neither has both
        self.assertEqual(compat.classify({4}), "unknown")                # removed everywhere
        self.assertEqual(compat.classify(set()), "unknown")


class Names(unittest.TestCase):
    def test_names(self):
        self.assertEqual(compat.type_name(142), "PQ2_0")
        self.assertEqual(compat.type_name(12), "Q4_K")
        self.assertEqual(compat.type_name(40), "NVFP4")
        self.assertEqual(compat.type_name(999), "type 999")

    def test_deciding_types_are_the_ones_not_shared(self):
        self.assertEqual(compat.deciding(BONSAI), ["PQ2_0"])
        self.assertEqual(compat.deciding({0, 12}), [])


class ForModel(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        compat._CACHE.clear()

    def gguf(self, types, name="m.gguf"):
        p = os.path.join(self.dir, name)
        write_gguf(p, [kv_str("general.architecture", "llama")],
                   [(f"t{i}", t, 64) for i, t in enumerate(types)])
        return p

    def test_reads_the_file(self):
        out = compat.for_model(self.gguf([0, 30, 142, 142]))
        self.assertEqual(out, {"class": "ik-only", "types": ["PQ2_0"], "ids": [0, 30, 142]})

    def test_unreadable_is_unknown(self):
        gone = {"class": "unknown", "types": [], "ids": []}
        self.assertEqual(compat.for_model(os.path.join(self.dir, "gone.gguf")), gone)
        self.assertEqual(compat.for_model(""), gone)

    def test_cached_until_the_file_changes(self):
        p = self.gguf([0, 12])
        with mock.patch.object(compat.gguf, "layout", wraps=compat.gguf.layout) as lay:
            compat.for_model(p)
            compat.for_model(p)
            self.assertEqual(lay.call_count, 1)
            self.gguf([0, 40, 40])                     # rewritten in place: new size
            self.assertEqual(compat.for_model(p)["class"], "mainline-only")
            self.assertEqual(lay.call_count, 2)


class Advice(unittest.TestCase):
    def test_on_llamacpp(self):
        self.assertIn("ik_llama.cpp", compat.advice("ik-only", ["PQ2_0"], "llamacpp"))
        self.assertIn("PQ2_0", compat.advice("ik-only", ["PQ2_0"], "llamacpp"))
        self.assertEqual(compat.advice("mainline-only", ["NVFP4"], "llamacpp"), "")

    def test_on_ik(self):
        self.assertIn("NVFP4", compat.advice("mainline-only", ["NVFP4"], "ikllama"))
        self.assertEqual(compat.advice("ik-only", ["PQ2_0"], "ikllama"), "")

    def test_nothing_to_say(self):
        self.assertEqual(compat.advice("any", [], "llamacpp"), "")
        self.assertEqual(compat.advice("unknown", [], "llamacpp"), "")

    def test_the_fix_is_a_per_model_build_when_ik_is_there(self):
        a = compat.advice("ik-only", ["PQ2_0"], "llamacpp", ik_built=True)
        self.assertIn("this model's build", a)
        self.assertNotIn("Build / Update", a)
        a = compat.advice("ik-only", ["PQ2_0"], "llamacpp")
        self.assertIn("Build / Update", a)
        self.assertIn("this model's build", a)

    def test_the_models_own_build_is_what_counts(self):
        self.assertEqual(compat.advice("ik-only", ["PQ2_0"], "llamacpp", build="ik_llama"), "")
        a = compat.advice("mainline-only", ["NVFP4"], "llamacpp", build="ik_llama")
        self.assertIn("NVFP4", a)
        self.assertIn("llama.cpp build", a)
        # pinned to a mainline install while the router runs ik
        self.assertEqual(compat.advice("mainline-only", ["NVFP4"], "ikllama", build="b8000-cuda"), "")

    def test_an_ik_build_older_than_the_file(self):
        """ik retired id 142 and later reused it for PQ2_0; an ik built in between
        crashes on a PQ2_0 file instead of refusing it."""
        for kw in ({"build": "ik_llama"}, {}):
            engine = "llamacpp" if kw else "ikllama"
            a = compat.advice("ik-only", ["PQ2_0"], engine, ik_missing=["PQ2_0"], **kw)
            self.assertIn("PQ2_0", a)
            self.assertIn("Update ik_llama.cpp", a)
        a = compat.advice("ik-only", ["PQ2_0"], "llamacpp", ik_built=True, ik_missing=["PQ2_0"])
        self.assertIn("Build / Update", a)
        self.assertIn("this model's build", a)


class BuildTypes(unittest.TestCase):
    """The ggml.h of the checkout a build came from says what that build knows,
    which today's upstream table can't."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        compat._SRC_CACHE.clear()

    def header(self, body):
        inc = os.path.join(self.dir, "ggml", "include")
        os.makedirs(inc, exist_ok=True)
        with open(os.path.join(inc, "ggml.h"), "w", encoding="utf-8") as f:
            f.write(body)

    def test_reads_the_enum_without_retired_ids(self):
        # this machine's ik checkout of 2026-07-29, trimmed
        self.header("    enum ggml_type {\n"
                    "        GGML_TYPE_F32     = 0,\n"
                    "        GGML_TYPE_BF16    = 30,\n"
                    "        GGML_TYPE_IQ6_K   = 141,\n"
                    "        // depricated: GGML_TYPE_IQ2_TN  = 142,\n"
                    "        GGML_TYPE_IQ4_KS  = 144,\n"
                    "        GGML_TYPE_COUNT,\n"
                    "    };\n")
        known = compat.build_types(self.dir)
        self.assertEqual(known, {0, 30, 141, 144})
        self.assertEqual(compat.missing(sorted(BONSAI), known), ["PQ2_0"])

    def test_unknown_when_it_cant_tell(self):
        self.assertIsNone(compat.build_types(""))
        self.assertIsNone(compat.build_types(self.dir))           # no header
        self.header("int nothing_here;\n")
        self.assertIsNone(compat.build_types(self.dir))
        self.assertEqual(compat.missing(sorted(BONSAI), None), [])


class BuildArchs(unittest.TestCase):
    """src/llama-arch.cpp of the checkout says which architectures a build
    loads: an older one stops with "unknown model architecture"."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        compat._ARCH_CACHE.clear()

    def table(self, body):
        src = os.path.join(self.dir, "src")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "llama-arch.cpp"), "w", encoding="utf-8") as f:
            f.write(body)

    # upstream's table of 2026-10-14, trimmed; the local unsloth fork of
    # 2026-08-26 has gemma-embedding but not gemma-embedding2
    TABLE = ("static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n"
             "    { LLM_ARCH_CLIP,             \"clip\"             }, // dummy, only used by llama-quantize\n"
             "    { LLM_ARCH_GEMMA3,           \"gemma3\"           },\n"
             "    { LLM_ARCH_GEMMA_EMBEDDING,  \"gemma-embedding\"  },\n"
             "    { LLM_ARCH_UNKNOWN,          \"(unknown)\"        },\n"
             "};\n"
             "static const std::map<llm_kv, const char *> LLM_KV_NAMES = {\n"
             "    { LLM_KV_GENERAL_ARCHITECTURE, \"general.architecture\" },\n"
             "};\n")

    def test_reads_the_name_table_only(self):
        self.table(self.TABLE)
        self.assertEqual(compat.build_archs(self.dir), {"gemma3", "gemma-embedding"})

    def test_unknown_when_it_cant_tell(self):
        self.assertIsNone(compat.build_archs(""))
        self.assertIsNone(compat.build_archs(self.dir))            # no file
        self.table("int nothing_here;\n")
        self.assertIsNone(compat.build_archs(self.dir))

    def test_advice_for_a_newer_architecture(self):
        self.table(self.TABLE)
        known = compat.build_archs(self.dir)
        msg = compat.arch_advice("gemma-embedding2", known)
        self.assertIn("gemma-embedding2", msg)
        self.assertIn("Build / Update", msg)
        self.assertEqual(compat.arch_advice("gemma-embedding", known), "")
        self.assertEqual(compat.arch_advice("gemma-embedding2", None), "")   # can't tell: say nothing
        self.assertEqual(compat.arch_advice(None, known), "")
        self.assertEqual(compat.arch_advice("clip", known), "")


# Verbatim from this machine's llama-server on the Ternary-Bonsai file (2026-10-05).
PQ2_LOG = """\
srv  load_models: spawning server instance with name=bonsai on port 55001
[55001] 0.00.143.684 E gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 43)
[55001] 0.00.143.688 E gguf_init_from_reader: failed to read tensor info
[55001] 0.00.149.600 E llama_model_load: error loading model: llama_model_loader: failed to load model from D:/m/Bonsai-PQ2_0.gguf
[55001] 0.00.197.563 E srv  llama_server: exiting due to model loading error
srv  operator(): instance name=bonsai exited with status 1
"""


class Diag(unittest.TestCase):
    def test_unknown_quant_type_names_it(self):
        d = diag.diagnose(PQ2_LOG, {}, model="bonsai")
        self.assertIn("invalid ggml type 142", d["error"])
        self.assertIn("PQ2_0", d["suggestion"])
        self.assertIn("ik_llama.cpp", d["suggestion"])

    def test_a_type_nobody_defines(self):
        d = diag.diagnose(PQ2_LOG.replace("type 142", "type 777"), {}, model="bonsai")
        self.assertIn("type 777", d["suggestion"])
        self.assertNotIn("ik_llama.cpp can", d["suggestion"])


class Route(unittest.TestCase):
    """The editor's GGUF card carries the verdict: read when a row opens,
    never on the dashboard's poll."""
    def setUp(self):
        import routes
        self.routes = routes
        self.cfg = {"active_engine": "llamacpp"}
        mock.patch.object(routes, "cfg", side_effect=lambda: dict(self.cfg)).start()
        mock.patch.object(routes.config, "read_sections",
                          return_value={"bonsai": {"model": "D:/m/b.gguf"}}).start()
        mock.patch.object(routes.gguf, "metadata", return_value={"architecture": "qwen3"}).start()
        self.for_model = mock.patch.object(routes.compat, "for_model",
                                           return_value={"class": "ik-only", "types": ["PQ2_0"],
                                                         "ids": sorted(BONSAI)}).start()
        self.addCleanup(mock.patch.stopall)

    def test_metadata_carries_the_verdict_and_advice(self):
        status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "bonsai"}))
        self.assertEqual(out["metadata"], {"architecture": "qwen3"})
        self.assertEqual((out["compat"]["class"], out["compat"]["types"]), ("ik-only", ["PQ2_0"]))
        self.assertIn("ik_llama.cpp", out["compat"]["advice"])
        self.for_model.assert_called_once_with("D:/m/b.gguf")

    def test_on_ik_an_ik_quant_needs_no_advice(self):
        self.cfg["active_engine"] = "ikllama"
        status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "bonsai"}))
        self.assertEqual(out["compat"]["advice"], "")

    def test_unknown_model(self):
        status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "nope"}))
        self.assertEqual(out["metadata"], {})
        self.assertEqual(out["compat"], {"class": "unknown", "types": [], "ids": [], "advice": "",
                                         "arch_advice": ""})

    def test_advice_knows_the_models_pin_and_whether_ik_is_built(self):
        self.cfg["model_builds"] = {"bonsai": "ik_llama"}
        with mock.patch.object(self.routes.builds, "options", return_value=[
                {"ref": "ik_llama", "label": "ik_llama.cpp", "server_bin": "/ik", "router": False}]):
            status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "bonsai"}))
        self.assertEqual(out["compat"]["advice"], "")
        self.assertEqual(out["builds"]["pinned"], "ik_llama")
        self.assertEqual([o["ref"] for o in out["builds"]["options"]], ["ik_llama"])

    def test_advice_reads_the_ik_checkout(self):
        self.cfg.update(model_builds={"bonsai": "ik_llama"}, ik_llama_src="D:/ik")
        with mock.patch.object(self.routes.builds, "options", return_value=[
                {"ref": "ik_llama", "label": "ik_llama.cpp", "server_bin": "/ik", "router": False}]), \
             mock.patch.object(self.routes.compat, "build_types", return_value=frozenset({0, 30})) as bt:
            status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "bonsai"}))
        bt.assert_called_once_with("D:/ik")
        self.assertIn("Update ik_llama.cpp", out["compat"]["advice"])

    def test_an_architecture_newer_than_the_routers_checkout(self):
        self.cfg.update(server_bin="D:/llama.cpp/build/bin/llama-server.exe", llama_src="D:/llama.cpp")
        with mock.patch.object(self.routes.compat, "build_archs",
                               return_value=frozenset({"llama", "gemma3"})) as ba:
            status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "bonsai"}))
        ba.assert_called_once_with("D:/llama.cpp")
        self.assertIn("qwen3", out["compat"]["arch_advice"])
        with mock.patch.object(self.routes.compat, "build_archs", return_value=frozenset({"qwen3"})):
            status, out = self.routes.get_model_metadata(self.routes.Req(qs={"model": "bonsai"}))
        self.assertEqual(out["compat"]["arch_advice"], "")


if __name__ == "__main__":
    unittest.main()
