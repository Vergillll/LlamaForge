"""Elastic VRAM: shrink models when other apps need the GPU, grow them back."""
import conftest_paths  # noqa: F401
import unittest

import elastic


def lay(blocks, experts=0):
    kv = {"block_count": blocks}
    if experts:
        kv["expert_count"] = experts
    return {"arch": "x", "kv": kv, "tensors": []}


def ngl(steps):
    return [s["gpu_layers"] for s in steps]


class LadderTest(unittest.TestCase):
    def test_dense_full_offload_steps_down_by_quarters(self):
        steps = elastic.ladder({"model": "m.gguf"}, lay(64))
        self.assertEqual(ngl(steps), [64, 48, 32, 16, 0])
        self.assertEqual(steps[0]["keys"], {})                 # the top is the user's own setting
        self.assertEqual(steps[1]["keys"], {"n-gpu-layers": "48"})
        self.assertEqual(steps[-1]["keys"], {"n-gpu-layers": "0"})

    def test_never_above_what_the_user_chose(self):
        steps = elastic.ladder({"n-gpu-layers": "40"}, lay(64))
        self.assertEqual(ngl(steps), [40, 32, 16, 0])
        self.assertEqual(steps[0]["keys"], {})

    def test_ngl_above_the_block_count_or_negative_means_all(self):
        self.assertEqual(ngl(elastic.ladder({"n-gpu-layers": "99"}, lay(32))), [32, 24, 16, 8, 0])
        self.assertEqual(ngl(elastic.ladder({"ngl": "-1"}, lay(32))), [32, 24, 16, 8, 0])

    def test_moe_moves_experts_first_then_layers(self):
        steps = elastic.ladder({}, lay(48, experts=128))
        self.assertEqual([s["keys"] for s in steps], [
            {},
            {"n-cpu-moe": "12"}, {"n-cpu-moe": "24"}, {"n-cpu-moe": "36"}, {"n-cpu-moe": "48"},
            {"n-cpu-moe": "48", "n-gpu-layers": "36"},
            {"n-cpu-moe": "48", "n-gpu-layers": "24"},
            {"n-cpu-moe": "48", "n-gpu-layers": "12"},
            {"n-cpu-moe": "48", "n-gpu-layers": "0"},
        ])

    def test_moe_skips_expert_steps_the_user_already_took(self):
        steps = elastic.ladder({"n-cpu-moe": "30"}, lay(48, experts=8))
        self.assertEqual([s["cpu_moe"] for s in steps[:3]], [30, 36, 48])

    def test_cpu_moe_flag_means_experts_are_already_on_the_cpu(self):
        steps = elastic.ladder({"cpu-moe": ""}, lay(48, experts=8))
        self.assertEqual(ngl(steps), [48, 36, 24, 12, 0])

    def test_unknown_layout_gives_no_ladder(self):
        self.assertEqual(elastic.ladder({}, None), [])
        self.assertEqual(elastic.ladder({}, lay(0)), [])

    def test_labels(self):
        steps = elastic.ladder({}, lay(48, experts=8))
        self.assertEqual(elastic.label(steps[0]), "48/48")
        self.assertEqual(elastic.label(steps[2]), "48/48, experts of 24 on CPU")


class FootprintsTest(unittest.TestCase):
    def test_measured_beats_predicted(self):
        fps = elastic.footprints([{0: 9000}, None], [12000, 6000], [0])
        self.assertEqual(fps[0], {0: 9000})

    def test_an_unmeasured_step_scales_from_a_measured_one(self):
        fps = elastic.footprints([{0: 8000, 1: 4000}, None], [12000, 6000], [0, 1])
        self.assertEqual(fps[1], {0: 4000, 1: 2000})

    def test_nothing_measured_splits_the_prediction_evenly(self):
        fps = elastic.footprints([None, None], [12000, 6000], [0, 1])
        self.assertEqual(fps, [{0: 6000, 1: 6000}, {0: 3000, 1: 3000}])


class ChooseTest(unittest.TestCase):
    FPS = [{0: 12000}, {0: 9000}, {0: 6000}, {0: 1000}]

    def test_the_largest_step_that_fits(self):
        self.assertEqual(elastic.choose(self.FPS, {0: 10000}), 1)
        self.assertEqual(elastic.choose(self.FPS, {0: 20000}), 0)

    def test_from_a_lower_bound(self):
        self.assertEqual(elastic.choose(self.FPS, {0: 20000}, lo=2), 2)

    def test_nothing_fits(self):
        self.assertIsNone(elastic.choose(self.FPS, {0: 500}))

    def test_every_gpu_must_fit(self):
        fps = [{0: 8000, 1: 8000}, {0: 4000, 1: 4000}]
        self.assertEqual(elastic.choose(fps, {0: 9000, 1: 5000}), 1)
        self.assertIsNone(elastic.choose(fps, {0: 9000}))          # GPU 1 has no room at all


class PlanKeysTest(unittest.TestCase):
    def test_first_shrink_records_what_the_user_had(self):
        writes, rec = elastic.plan_keys({"model": "m.gguf", "n-gpu-layers": "99"}, None,
                                        {"n-gpu-layers": "48"})
        self.assertEqual(writes, {"n-gpu-layers": "48"})
        self.assertEqual(rec, {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}})

    def test_an_unset_key_is_remembered_as_unset(self):
        _, rec = elastic.plan_keys({}, None, {"n-cpu-moe": "12"})
        self.assertEqual(rec["was"], {"n-cpu-moe": None})

    def test_back_at_the_top_puts_the_user_values_back(self):
        rec = {"keys": {"n-cpu-moe": "48", "n-gpu-layers": "24"},
               "was": {"n-cpu-moe": None, "n-gpu-layers": "99"}}
        writes, rec2 = elastic.plan_keys({"n-cpu-moe": "48", "n-gpu-layers": "24"}, rec, {})
        self.assertEqual(writes, {"n-cpu-moe": None, "n-gpu-layers": "99"})
        self.assertIsNone(rec2)

    def test_a_middle_step_keeps_the_original_was(self):
        rec = {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}}
        writes, rec2 = elastic.plan_keys({"n-gpu-layers": "48"}, rec, {"n-gpu-layers": "16"})
        self.assertEqual(writes, {"n-gpu-layers": "16"})
        self.assertEqual(rec2["was"], {"n-gpu-layers": "99"})

    def test_a_user_edit_wins(self):
        rec = {"keys": {"n-gpu-layers": "48"}, "was": {"n-gpu-layers": "99"}}
        # the user typed 30 into the knob while it was shrunk: it's theirs now
        writes, rec2 = elastic.plan_keys({"n-gpu-layers": "30"}, rec, {})
        self.assertEqual(writes, {})
        self.assertIsNone(rec2)
        writes, rec3 = elastic.plan_keys({"n-gpu-layers": "30"}, rec, {"n-gpu-layers": "16"})
        self.assertEqual(writes, {})
        self.assertIsNone(rec3)


if __name__ == "__main__":
    unittest.main()
