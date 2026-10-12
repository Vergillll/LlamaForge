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


if __name__ == "__main__":
    unittest.main()
