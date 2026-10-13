"""The Load strip's VRAM estimate: KV sizes for the record cache types and the
low-bit ones some builds add, the exact-tail rows, the slider limits, and the
/api/model/predict endpoint that serves them to the editor."""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import footprint
import routes
import slots
from routes import Req

M = 1024 * 1024


def dense(n_layer=4, kv_heads=8, head_dim=128, **extra):
    """n_layer blocks of 100 MiB, no output split, plain attention."""
    tensors = [("token_embd.weight", 14, 100 * M), ("output.weight", 14, 100 * M)]
    tensors += [(f"blk.{i}.attn_q.weight", 14, 100 * M) for i in range(n_layer)]
    kv = {"block_count": n_layer, "embedding_length": 4096, "head_count": 32,
          "head_count_kv": kv_heads, "key_length": head_dim, "value_length": head_dim,
          "context_length": 131072}
    kv.update(extra)
    return {"arch": "llama", "kv": kv, "tensors": tensors}


def kv_bytes(p):
    return p["kv_gpu_mib"]


class CacheTypes(unittest.TestCase):
    def test_low_bit_types_are_sized_not_counted_as_f16(self):
        base = footprint.predict({"ctx-size": "8192", "parallel": "1"}, dense())["kv_gpu_mib"]
        for t, frac in (("q6_0", 26 / 64), ("q6_1", 28 / 64), ("q3_0", 14 / 64),
                        ("q3_1", 16 / 64), ("q2_0", 10 / 64), ("q2_1", 12 / 64)):
            p = footprint.predict({"ctx-size": "8192", "parallel": "1",
                                   "cache-type-k": t, "cache-type-v": t}, dense())
            self.assertEqual(p["kv_gpu_mib"], round(base * frac), t)
            self.assertTrue(p["confident"], t)

    def test_an_unknown_cache_type_says_the_estimate_is_uncertain(self):
        p = footprint.predict({"ctx-size": "8192", "cache-type-k": "turbo3"}, dense())
        self.assertFalse(p["confident"])
        self.assertEqual(p["uncertain"], ["cache type turbo3 is unknown here: counted as f16"])


class KvarnRecords(unittest.TestCase):
    def test_record_bytes(self):
        # 128-dim: 128x128 values at 4 bits + (2*128 + 128) uint16 = 8192 + 768
        self.assertEqual(footprint._kvarn_record(128, 4, False), 8960)
        self.assertEqual(footprint._kvarn_record(128, 4, True), 8960)
        # 64-dim records are 64x128 (K) / 128x64 (V): their metadata differs
        self.assertEqual(footprint._kvarn_record(64, 4, False), 4096 + (2 * 64 + 128) * 2)
        self.assertEqual(footprint._kvarn_record(64, 4, True), 4096 + (2 * 128 + 64) * 2)
        self.assertEqual(footprint._kvarn_record(128, 5, False), 10240 + 768)

    def test_a_full_attention_layer(self):
        """4 layers, 8 KV heads of 128, one slot, 8192 cells: 64 groups of
        records, an F16 ring of 3 groups (live + rollback + sink), and the
        minimum exact tail of one group."""
        p = footprint.predict({"ctx-size": "8192", "parallel": "1",
                               "cache-type-k": "kvarn4", "cache-type-v": "kvarn4"}, dense())
        body = 4 * 64 * 8 * (8960 + 8960)
        stage = 4 * 2 * 8 * 256 * 128 * 3
        self.assertEqual(p["kv_body_mib"], round(body / M))
        exact = 4 * (128 + 1) * 8 * 256 * 2         # one group exact even with no tail asked for
        self.assertEqual(p["kv_stage_mib"], round(stage / M))
        self.assertEqual(p["kv_exact_mib"], round(exact / M))
        self.assertEqual(p["kv_gpu_mib"], round((body + stage + exact) / M))
        self.assertTrue(p["confident"])

    def test_the_exact_tail_is_rounded_to_groups_and_kept_per_sequence(self):
        p = footprint.predict({"ctx-size": "8192", "parallel": "2", "cache-type-k": "kvarn4",
                               "cache-type-v": "kvarn4", "kv-tail-tokens": "1000"}, dense())
        rows = (1024 + 1) * 2                         # rounded up to 128, + rollback, x 2 slots
        self.assertEqual(p["kv_exact_mib"], round(4 * rows * 8 * 256 * 2 / M))

    def test_head_dims_of_256_and_512_are_sliced_into_128_records(self):
        p = footprint.predict({"ctx-size": "8192", "parallel": "1", "cache-type-k": "kvarn4",
                               "cache-type-v": "kvarn4"}, dense(kv_heads=2, head_dim=512))
        self.assertEqual(p["kv_body_mib"], round(4 * 64 * 2 * 4 * (8960 + 8960) / M))

    def test_a_tail_covering_the_window_replaces_a_sliding_layer_cache(self):
        lay = dense(n_layer=2, sliding_window=1024, sliding_window_pattern=[True, False])
        s = {"ctx-size": "8192", "parallel": "1", "cache-type-k": "kvarn4",
             "cache-type-v": "kvarn4", "kv-tail-tokens": "2048"}
        p = footprint.predict(s, lay)
        # the global layer keeps its records; the sliding one is only W + R rows
        glob_body = 64 * 8 * (8960 + 8960)
        exact = ((1024 + 1) + (2048 + 1)) * 8 * 256 * 2
        self.assertEqual(p["kv_body_mib"], round(glob_body / M))
        self.assertEqual(p["kv_exact_mib"], round(exact / M))

    def test_a_kvarn_key_with_a_plain_value_is_flagged(self):
        p = footprint.predict({"ctx-size": "8192", "cache-type-k": "kvarn4",
                               "cache-type-v": "q8_0"}, dense())
        self.assertFalse(p["confident"])


class NextTokenLayers(unittest.TestCase):
    def test_mtp_blocks_are_not_loaded_without_mtp_speculation(self):
        """Qwen 3.8 27B: blk.64 (250 MiB) is the next-token layer; the loader
        logs it as an unused tensor and skips it, unless MTP speculation is on."""
        lay = dense(n_layer=5, nextn_predict_layers=1)
        plain = footprint.predict({"ctx-size": "8192"}, lay)
        mtp = footprint.predict({"ctx-size": "8192", "spec-type": "mtp"}, lay)
        self.assertEqual(mtp["weights_gpu_mib"] - plain["weights_gpu_mib"], 100)
        self.assertEqual(plain["weights_ram_mib"], 100)        # token_embd only: the block isn't anywhere

class Slots(unittest.TestCase):
    def test_parallel_auto_is_four_slots_on_one_pool(self):
        for v in ("-1", None):
            s = {"ctx-size": "8192"} if v is None else {"ctx-size": "8192", "parallel": v}
            p = footprint.predict(s, dense())
            self.assertEqual((p["n_seq"], p["ctx_slot"]), (4, 8192), v)

    def test_explicit_slots_split_the_context(self):
        p = footprint.predict({"ctx-size": "8192", "parallel": "4"}, dense())
        self.assertEqual((p["n_seq"], p["ctx_slot"]), (4, 2048))


class Merge(unittest.TestCase):
    def test_unsaved_values_override_and_blanks_unset_any_spelling(self):
        saved = {"gpu-layers": "40", "ctx-size": "8192", "model": "F:/m.gguf"}
        out = footprint.merge(saved, {"n-gpu-layers": "", "ctx-size": "16384", "fit": "off"})
        self.assertEqual(out, {"ctx-size": "16384", "model": "F:/m.gguf", "fit": "off"})
        self.assertEqual(footprint.merge(saved, {"ngl": "10"})["n-gpu-layers"], "10")


class Limits(unittest.TestCase):
    def test_context_and_layer_limits_track_the_budget(self):
        lay = dense(n_layer=4)
        s = {"ctx-size": "8192", "parallel": "1", "fit": "off"}
        need = footprint.gpu_total(footprint.predict(s, lay))
        lim = footprint.fit_limits(s, lay, need, 131072, 4, False)
        self.assertEqual(lim["ctx"], 8192)                      # exactly the budget
        self.assertEqual(lim["layers"], 5)                      # every layer + output
        # 60 MiB short: the 128 MiB of f16 KV at 8192 halves at 4096, one layer is 100 MiB + its KV
        tight = footprint.fit_limits(s, lay, need - 60, 131072, 4, False)
        self.assertEqual(tight["ctx"], 4096)
        self.assertEqual(tight["layers"], 4)
        self.assertEqual(footprint.fit_limits(s, lay, 10, 131072, 4, False),
                         {"ctx": None, "layers": None})


class Endpoint(unittest.TestCase):
    def _call(self, settings, section=None, runtime=None, gpus=None, measured=None, status="unloaded"):
        section = section or {"model": "F:/m.gguf", "ctx-size": "8192", "parallel": "1", "fit": "off"}
        ini = {"*": {}, "m": section}
        rows = {"models": [{"id": "m", "status": status, "in_ini": True,
                            **({"runtime": runtime} if runtime else {})}]}
        gpus = gpus if gpus is not None else [{"index": 0, "name": "GPU <0>", "used": 1000, "total": 16000}]
        store = mock.Mock()
        store.measured.return_value = measured or {}
        with mock.patch.object(routes.config, "read_sections", return_value=ini), \
             mock.patch.object(routes, "_layout", return_value=dense()), \
             mock.patch.object(routes, "model_state", return_value=rows), \
             mock.patch.object(routes._GPU_TELEMETRY, "get", return_value=gpus), \
             mock.patch.object(routes.slotctl, "_size", return_value=0), \
             mock.patch.object(routes.SLOTS, "store", store), \
             mock.patch.object(routes.SLOTS, "key", return_value="k"):
            return routes.post_model_predict(Req(body={"model": "m", "settings": settings}))[1]

    def test_unsaved_values_against_the_saved_ones(self):
        out = self._call({"ctx-size": "16384"})
        self.assertEqual(out["predicted"]["ctx"], 16384)
        self.assertEqual(out["saved"]["ctx"], 8192)
        self.assertGreater(out["predicted"]["total_mib"], out["saved"]["total_mib"])
        self.assertEqual(out["gpu"], {"name": "GPU <0>", "total_mib": 16000,
                                      "others_mib": 1000, "budget_mib": 15000})
        self.assertTrue(out["fits"])
        self.assertEqual(set(out["limits"]), {"ctx", "layers"})
        self.assertEqual(out["notes"], [])

    def test_the_breakdown_adds_up(self):
        p = self._call({})["predicted"]
        self.assertEqual(p["total_mib"], p["weights_mib"] + p["kv_mib"] + p["exact_mib"]
                         + p["mmproj_mib"] + p["compute_mib"])
        self.assertGreaterEqual(p["compute_mib"], slots.OVERHEAD_MIB)

    def test_left_to_fit_says_so_and_uses_the_last_load(self):
        sect = {"model": "F:/m.gguf"}
        out = self._call({}, section=sect)
        self.assertTrue(out["fit_on"])
        self.assertTrue(any("context left to fit" in n for n in out["notes"]))
        self.assertTrue(any("placement left to fit" in n for n in out["notes"]))
        out = self._call({}, section=sect, runtime={"n_ctx_slot": 15360, "n_slots": 1})
        self.assertEqual(out["predicted"]["ctx"], 15360)
        self.assertTrue(any("fit chose on the last load" in n for n in out["notes"]))

    def test_a_loaded_model_is_not_counted_as_another_app(self):
        out = self._call({}, status="loaded", gpus=[{"index": 0, "name": "g", "used": 9000, "total": 16000}])
        self.assertEqual(out["gpu"]["others_mib"], max(0, 9000 - out["saved"]["total_mib"]))

    def test_the_last_measured_load(self):
        self.assertEqual(self._call({}, measured={(0,): {0: 9876}})["measured_mib"], 9876)
        self.assertIsNone(self._call({})["measured_mib"])

    def test_no_gpu_reading_still_predicts(self):
        out = self._call({}, gpus=[{"error": "nvidia-smi missing"}])
        self.assertIsNone(out["gpu"])
        self.assertIsNone(out["fits"])
        self.assertIn("total_mib", out["predicted"])

    def test_an_unknown_model_is_a_404(self):
        with mock.patch.object(routes.config, "read_sections", return_value={"*": {}}):
            with self.assertRaises(routes.ApiError):
                routes.post_model_predict(Req(body={"model": "nope", "settings": {}}))


class PlainLoadMeasure(unittest.TestCase):
    """One model at a time through the router: measured like the slot manager's
    loads, into the same store."""

    def setUp(self):
        routes._PLAIN["live"].clear()
        self.store = mock.Mock()
        self.used = iter([])
        self.status = {}
        patches = [
            mock.patch.object(routes, "_spawn", lambda fn, *a: fn(*a)),
            mock.patch.object(routes, "_stop_procs"),
            mock.patch.object(routes, "_slotted", return_value=False),
            mock.patch.object(routes, "_slots_on", return_value=False),
            mock.patch.object(routes.PROCS, "has", return_value=False),
            mock.patch.object(routes, "router", return_value=(200, {"success": True})),
            mock.patch.object(routes.SLOTS, "store", self.store),
            mock.patch.object(routes.SLOTS, "key", return_value="m|build|hash"),
            mock.patch.object(routes.SLOTS, "_statuses", side_effect=lambda: self.status),
            mock.patch.object(routes.SLOTS, "_used", side_effect=lambda: next(self.used)),
            mock.patch.object(routes.SLOTS, "_wait_loaded", return_value=(True, "")),
            mock.patch.object(routes.SLOTS, "sleep"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_a_load_with_nothing_else_up_is_recorded(self):
        self.used = iter([{0: 900}, {0: 15600}])
        routes.post_load(Req(body={"model": "m"}))
        self.store.record.assert_called_once_with("m|build|hash", (0,), {0: 14700}, "load")
        self.assertEqual(routes._PLAIN["live"]["m"], ("m|build|hash", (0,)))

    def test_a_load_that_evicts_another_model_is_not(self):
        self.status = {"other": {"value": "loaded"}}
        self.used = iter([{0: 15000}, {0: 15600}])
        routes.post_load(Req(body={"model": "m"}))
        self.store.record.assert_not_called()

    def test_a_load_overtaken_by_another_request_is_not(self):
        self.used = iter([{0: 900}, {0: 15600}])
        with mock.patch.object(routes.SLOTS, "_wait_loaded",
                               side_effect=lambda mid: (routes._PLAIN.__setitem__("seq", routes._PLAIN["seq"] + 1), (True, ""))[1]):
            routes.post_load(Req(body={"model": "m"}))
        self.store.record.assert_not_called()

    def test_the_unload_records_what_came_back(self):
        self.used = iter([{0: 900}, {0: 15600}, {0: 16100}, {0: 880}])
        routes.post_load(Req(body={"model": "m"}))
        routes.post_unload(Req(body={"model": "m"}))
        self.assertEqual(self.store.record.call_args_list[-1],
                         mock.call("m|build|hash", (0,), {0: 15220}, "unload"))
        self.assertNotIn("m", routes._PLAIN["live"])

    def test_an_unmeasured_model_is_not_recorded_at_unload(self):
        routes.post_unload(Req(body={"model": "m"}))
        self.store.record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
