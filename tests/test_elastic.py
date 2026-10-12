"""Elastic VRAM: shrink models when other apps need the GPU, grow them back."""
import conftest_paths  # noqa: F401
import io as _io
import unittest
from unittest import mock

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


GiB = 1024
LADDER = elastic.ladder({}, lay(64))[:3] + elastic.ladder({}, lay(64))[-1:]   # 64, 48, 32, 0


class FakeIO:
    def __init__(self, total=16, others=2):
        self.total, self.others = total * GiB, others * GiB
        self.fps = {"big": [{0: 12 * GiB}, {0: 9 * GiB}, {0: 6 * GiB}, {0: 1 * GiB}]}
        self.at = {"big": 0}
        self.game = False
        self.fail = set()               # step indexes whose load fails
        self.files = [(0, "lf-x-0.bin")]
        self.calls = []
        self.room = {}                  # mid -> the room kept with its record (config.json)

    def snapshot(self):
        ours = sum(self.fps[m][a][0] for m, a in self.at.items())
        return {0: {"used": self.others + ours, "total": self.total}}

    def models(self, gpus):
        return [{"id": m, "steps": LADDER, "fps": self.fps[m], "at": a, "room": self.room.get(m)}
                for m, a in self.at.items()]

    def game_active(self):
        return self.game

    def tidy(self):
        pass

    def save(self, mid):
        self.calls.append(("save", mid))
        return {"files": list(self.files), "note": ""}

    def swap(self, mid, step, room=None):
        i = LADDER.index(step)
        self.calls.append(("swap", mid, i))
        if i in self.fail:
            return "out of memory"
        self.at[mid] = i
        if i == 0:                      # back at the top: the record goes, the room with it
            self.room.pop(mid, None)
        elif room:
            self.room[mid] = dict(room)
        return ""

    def restore(self, mid, handle):
        self.calls.append(("restore", mid))
        return len(handle["files"])

    def drop(self, handle):
        self.calls.append(("drop",))

    def giveup(self, mid):
        self.calls.append(("giveup", mid))
        del self.at[mid]
        self.room.pop(mid, None)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def controller(io, fresh=False, **cfg):
    """fresh: the controller's first look is the baseline (just turned on).
    Otherwise it has already seen the GPU calm with no other apps, so the
    others a test starts with count as apps that grew."""
    c = {"elastic_vram": True, "elastic_grow": True, "elastic_keep_conversation": True,
         "elastic_headroom_mib": 1536}
    c.update(cfg)
    clock = Clock()
    el = elastic.Elastic(lambda: c, io, clock=clock)
    if not fresh:
        el._calm_others = 0
    return el, c, clock


def swaps(io):
    return [c for c in io.calls if c[0] == "swap"]


class ShrinkTest(unittest.TestCase):
    def test_two_ticks_under_half_the_headroom_then_one_step_that_fits(self):
        io = FakeIO(); el, _, _ = controller(io)
        el.tick()
        self.assertEqual(io.calls, [])                          # 2 GiB free: fine
        io.others = 4 * GiB                                     # 0 free now
        el.tick()
        self.assertEqual(io.calls, [])                          # one sighting is not enough
        el.tick()
        self.assertEqual(io.calls, [("save", "big"), ("swap", "big", 1), ("restore", "big")])
        ev = el.status()["events"][0]
        self.assertEqual((ev["model"], ev["from"], ev["to"], ev["ok"], ev["kept"]),
                         ("big", "64/64", "48/64", True, True))

    def test_keeping_conversations_off_skips_save_and_restore(self):
        io = FakeIO(others=4); el, _, _ = controller(io, elastic_keep_conversation=False)
        el.tick(); el.tick()
        self.assertEqual(io.calls, [("swap", "big", 1)])
        self.assertIsNone(el.status()["events"][0]["kept"])

    def test_a_failed_load_walks_down_the_ladder(self):
        io = FakeIO(others=4); io.fail = {1}
        el, _, _ = controller(io)
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1), ("swap", "big", 2)])
        ev = el.status()["events"][0]
        self.assertEqual(ev["to"], "32/64")
        self.assertIn("48/64 didn't load: out of memory", ev["note"])

    def test_failing_all_the_way_down_gives_up(self):
        io = FakeIO(others=4); io.fail = {1, 2, 3}
        el, _, _ = controller(io)
        el.tick(); el.tick()
        self.assertEqual(io.calls[-2:], [("giveup", "big"), ("drop",)])
        ev = el.status()["events"][0]
        self.assertEqual((ev["ok"], ev["to"]), (False, "unloaded"))
        self.assertIn("couldn't fit big", ev["note"])

    def test_one_model_per_tick_the_one_that_frees_most(self):
        io = FakeIO(total=24, others=2)
        io.fps["small"] = [{0: 3 * GiB}, {0: 2304}, {0: 1536}, {0: 512}]
        io.at["small"] = 0
        el, _, _ = controller(io)
        io.others = 9 * GiB
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])

    def test_stands_down_while_game_mode_is_active(self):
        io = FakeIO(others=4); io.game = True
        el, _, _ = controller(io)
        el.tick(); el.tick(); el.tick()
        self.assertEqual(io.calls, [])
        self.assertTrue(el.status()["standby"])

    def test_off_does_nothing(self):
        io = FakeIO(others=4)
        el, _, _ = controller(io, elastic_vram=False)
        el.tick(); el.tick()
        self.assertEqual(io.calls, [])
        self.assertFalse(el.status()["enabled"])

    def test_errors_in_one_tick_do_not_kill_the_loop(self):
        io = FakeIO()

        def boom(gpus):
            raise OSError("router down")
        io.models = boom
        el, _, _ = controller(io)
        el.safe_tick()                                          # no exception escapes


class GrowTest(unittest.TestCase):
    def shrunk(self, **cfg):
        io = FakeIO(others=4); el, c, clock = controller(io, **cfg)
        el.tick(); el.tick()                                    # shrink to 48/64
        io.calls.clear()
        return io, el, c, clock

    def test_six_quiet_ticks_and_the_cooldown_then_straight_to_the_top(self):
        io, el, _, clock = self.shrunk()
        io.others = 1 * GiB
        for _ in range(6):
            clock.t += 5
            el.tick()
        self.assertEqual(swaps(io), [])                         # 30 s: still inside the 120 s cooldown
        clock.t += 100
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0)])
        self.assertEqual(el.status()["events"][0]["to"], "64/64")

    def test_needs_an_extra_gib_of_margin(self):
        io, el, _, clock = self.shrunk()
        io.others = 2 * GiB + 512        # step 0 fits the headroom, but not headroom + 1 GiB
        clock.t += 500
        for _ in range(7):
            el.tick()
        self.assertEqual(swaps(io), [])

    def test_grow_off(self):
        io, el, _, clock = self.shrunk(elastic_grow=False)
        io.others = 0
        clock.t += 500
        for _ in range(7):
            el.tick()
        self.assertEqual(swaps(io), [])

    def test_not_while_the_other_app_still_holds_most_of_it(self):
        io = FakeIO(total=20, others=8); el, _, clock = controller(io, elastic_headroom_mib=512)
        el.tick(); el.tick()                                    # shrink with others at 8 GiB
        io.calls.clear()
        io.others = 6 * GiB + 512        # 81 %: it's still there (step 0 would fit with the margin)
        clock.t += 500
        for _ in range(7):
            el.tick()
        self.assertEqual(swaps(io), [])

    def test_a_shrink_soon_after_a_grow_doubles_the_cooldown(self):
        io, el, _, clock = self.shrunk()
        io.others = 1 * GiB
        clock.t += 200
        for _ in range(6):
            el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0)])
        io.others = 4 * GiB
        clock.t += 60
        el.tick(); el.tick()
        self.assertEqual(swaps(io)[-1], ("swap", "big", 1))
        self.assertEqual(el.status()["cooldown_s"], 240)

    def test_an_hour_of_calm_resets_the_cooldown(self):
        io, el, _, clock = self.shrunk()
        el._cooldown = 1800
        clock.t += 3600
        el.tick()
        self.assertEqual(el.status()["cooldown_s"], 120)

    def test_turning_it_off_grows_everything_back(self):
        io, el, c, clock = self.shrunk()
        c["elastic_vram"] = False
        io.others = 1 * GiB              # step 0 fits again
        clock.t += 200                   # and the cooldown since the shrink has passed
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0)])
        self.assertIn("turned off", el.status()["events"][0]["reason"])

    def test_grow_now_skips_the_wait_but_not_the_fit(self):
        io, el, _, _ = self.shrunk()
        out = el.grow_now("big")                                # others still at 4 GiB
        self.assertFalse(out["ok"])
        self.assertEqual(swaps(io), [])
        self.assertIn("still use", el.status()["events"][0]["note"])
        io.others = 1 * GiB
        self.assertEqual(el.grow_now("big"), {"ok": True, "to": "64/64"})
        self.assertEqual(swaps(io), [("swap", "big", 0)])

    def test_grow_now_on_a_model_that_is_not_shrunk(self):
        io = FakeIO(); el, _, _ = controller(io)
        self.assertFalse(el.grow_now("big")["ok"])
        ev = el.status()["events"][0]                           # the log says why nothing happened
        self.assertEqual((ev["model"], ev["reason"], ev["ok"]), ("big", "you asked", False))
        self.assertIn("big isn't resized any more", ev["note"])


class StatusTest(unittest.TestCase):
    def test_status_lists_models_and_others(self):
        io = FakeIO(others=4); el, _, _ = controller(io)
        el.tick(); el.tick(); el.tick()
        st = el.status()
        m = st["models"][0]
        self.assertEqual((m["id"], m["at"], m["gpu_layers"], m["layers"], m["label"]),
                         ("big", 1, 48, 64, "48/64"))
        self.assertEqual(st["others_mib"], {0: 4 * GiB})
        self.assertIsNone(st["swapping"])
        self.assertEqual(len(st["events"]), 1)

    def test_the_log_keeps_the_last_twenty(self):
        io = FakeIO(); el, _, _ = controller(io)
        for i in range(25):
            el._log("big", "64/64", "48/64", "test", 0, True, None, "")
        self.assertEqual(len(el.status()["events"]), 20)


class QuietGiveupIO(FakeIO):
    """giveup leaves the model in the pool, as a router that failed to unload would."""
    def giveup(self, mid):
        self.calls.append(("giveup", mid))


class ExceptionSafetyTest(unittest.TestCase):
    def test_a_raising_swap_counts_as_a_failed_load(self):
        io = QuietGiveupIO(others=4)

        def boom(mid, step):
            io.calls.append(("swap", mid, LADDER.index(step)))
            raise OSError("boom")
        io.swap = boom
        el, _, _ = controller(io)
        el.tick(); el.tick()
        evs = el.status()["events"]
        self.assertEqual(len(evs), 1)
        self.assertFalse(evs[0]["ok"])
        self.assertIn("OSError: boom", evs[0]["note"])
        self.assertIsNotNone(el._last_change)
        self.assertEqual(io.calls.count(("giveup", "big")), 1)
        self.assertEqual(io.calls.count(("drop",)), 1)

    def test_no_new_attempt_on_a_model_that_just_failed(self):
        io = QuietGiveupIO(others=4); io.fail = {1, 2, 3}
        el, _, clock = controller(io)
        el.tick(); el.tick()
        io.at["big"] = 1; io.others = 8 * GiB                   # still listed, not at its own step
        n = len(io.calls)
        for _ in range(4):
            el.tick()
        self.assertEqual(len(io.calls), n)
        clock.t += elastic.COOLDOWN_S + 1
        el.tick(); el.tick()
        self.assertGreater(len(io.calls), n)

    def test_a_raising_save_still_swaps(self):
        io = FakeIO(others=4)

        def boom(mid):
            raise OSError("disk")
        io.save = boom
        el, _, _ = controller(io)
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])
        ev = el.status()["events"][0]
        self.assertTrue(ev["ok"])
        self.assertIn("conversation not kept: OSError: disk", ev["note"])

    def test_a_raising_restore_drops_the_files(self):
        io = FakeIO(others=4)

        def boom(mid, handle):
            raise OSError("gone")
        io.restore = boom
        el, _, _ = controller(io)
        el.tick(); el.tick()
        self.assertEqual(io.calls[-1], ("drop",))
        ev = el.status()["events"][0]
        self.assertTrue(ev["ok"])
        self.assertFalse(ev["kept"])

    def test_the_files_are_dropped_when_the_swap_fails_without_a_restore(self):
        io = FakeIO(others=4); io.fail = {1, 2, 3}
        el, _, _ = controller(io)
        el.tick(); el.tick()
        self.assertNotIn(("restore", "big"), io.calls)
        self.assertEqual(io.calls.count(("drop",)), 1)


class OffPathTest(unittest.TestCase):
    def shrunk_off(self):
        io = FakeIO(others=4); el, c, clock = controller(io)
        el.tick(); el.tick()
        io.calls.clear()
        c["elastic_vram"] = False
        clock.t += 200                   # the cooldown since the shrink has passed
        return io, el, c

    def test_off_waits_until_step_zero_fits(self):
        io, el, _ = self.shrunk_off()
        el.tick(); el.tick()                                    # others still at 4 GiB
        self.assertEqual(io.calls, [])
        io.others = 1 * GiB
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0)])

    def test_off_gives_up_only_when_every_step_fails(self):
        io, el, _ = self.shrunk_off()
        io.others = 1 * GiB; io.fail = {0, 1, 2, 3}
        el.tick()
        self.assertEqual(io.calls.count(("giveup", "big")), 1)
        el.tick(); el.tick()
        self.assertEqual(len(swaps(io)), 4)                     # the model is gone: no retries


class BaselineBusyTest(unittest.TestCase):
    def test_a_big_baseline_does_not_block_the_grow(self):
        io = FakeIO()
        io.total = 19968
        io.fps["big"] = [{0: 11264}, {0: 9216}, {0: 6 * GiB}, {0: 1 * GiB}]
        io.others = 7 * GiB                                     # the baseline
        el, _, clock = controller(io, elastic_headroom_mib=512)
        el.tick()                                               # calm: records the baseline
        io.others = 7 * GiB + 1536                              # a 1.5 GiB spike
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])
        io.calls.clear()
        io.others = 7 * GiB
        for _ in range(6):
            clock.t += 5
            el.tick()
        clock.t += 200
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0)])


class ShrinkGuardsTest(unittest.TestCase):
    def test_a_step_that_frees_nothing_is_not_a_candidate(self):
        io = FakeIO(others=4)
        io.fps["big"] = [{0: 12 * GiB}] * 4
        el, _, _ = controller(io)
        el.tick(); el.tick()
        self.assertEqual(io.calls, [])


class FailedGrowTest(unittest.TestCase):
    def test_a_grow_never_ends_below_where_it_started_and_doubles_the_cooldown(self):
        io = FakeIO(others=4); el, _, clock = controller(io)
        el.tick(); el.tick()
        io.calls.clear()
        io.others = 1 * GiB; io.fail = {0}
        clock.t += 200
        for _ in range(6):
            el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0), ("swap", "big", 1)])
        self.assertEqual(el.status()["cooldown_s"], 240)

    def test_an_unloaded_grow_also_doubles_it(self):
        io = QuietGiveupIO(others=4); el, _, clock = controller(io)
        el.tick(); el.tick()
        io.others = 1 * GiB; io.fail = {0, 1, 2, 3}
        clock.t += 200
        for _ in range(6):
            el.tick()
        self.assertEqual(el.status()["cooldown_s"], 240)


class AliasTest(unittest.TestCase):
    def test_ladder_reads_aliases(self):
        self.assertEqual(ngl(elastic.ladder({"gpu-layers": "20"}, lay(64))), [20, 16, 0])
        self.assertEqual(ngl(elastic.ladder({"ngl": "20"}, lay(64))), [20, 16, 0])
        steps = elastic.ladder({"ncmoe": "30"}, lay(48, experts=8))
        self.assertEqual([s["cpu_moe"] for s in steps[:3]], [30, 36, 48])
        self.assertEqual(ngl(elastic.ladder({"cmoe": ""}, lay(48, experts=8))), [48, 36, 24, 12, 0])

    def test_cpu_moe_false_is_off(self):
        steps = elastic.ladder({"cpu-moe": "false"}, lay(48, experts=8))
        self.assertIn({"n-cpu-moe": "12"}, [s["keys"] for s in steps])

    def test_plan_keys_overwrites_the_users_spelling(self):
        writes, rec = elastic.plan_keys({"gpu-layers": "20"}, None, {"n-gpu-layers": "16"})
        self.assertEqual(writes, {"gpu-layers": "16"})
        self.assertEqual(rec, {"keys": {"gpu-layers": "16"}, "was": {"gpu-layers": "20"}})
        writes, rec2 = elastic.plan_keys({"gpu-layers": "16"}, rec, {})
        self.assertEqual(writes, {"gpu-layers": "20"})
        self.assertIsNone(rec2)


class FootprintGuardsTest(unittest.TestCase):
    def test_none_and_short_predictions_read_as_zero(self):
        fps = elastic.footprints([None, None, None], [None, 6000], [0, 1])
        self.assertEqual(fps, [{0: 0, 1: 0}, {0: 3000, 1: 3000}, {0: 0, 1: 0}])
        fps = elastic.footprints([{0: 8000}, None, None], [12000, None], [0])
        self.assertEqual(fps, [{0: 8000}, {0: 0}, {0: 0}])


class Round2Test(unittest.TestCase):
    def test_a_grow_that_lands_back_does_not_block_the_next_shrink(self):
        io = FakeIO(others=4); el, _, clock = controller(io)
        el.tick(); el.tick()                                    # shrunk to step 1
        io.others = 1 * GiB; io.fail = {0}
        clock.t += 200
        for _ in range(6):
            el.tick()
        self.assertEqual(io.at["big"], 1)                       # the grow fell back to step 1
        io.calls.clear()
        io.others = 7 * GiB                                     # real pressure
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 2)])

    def test_a_model_back_at_step_zero_clears_its_failure(self):
        io = QuietGiveupIO(others=4); io.fail = {1, 2, 3}
        el, _, clock = controller(io)
        el.tick(); el.tick()
        self.assertIn("big", el._failed_at)
        io.fail = set(); io.at["big"] = 0                       # the user loaded it again
        el.tick()
        self.assertNotIn("big", el._failed_at)

    def test_off_path_failure_walks_down_and_stays_loaded(self):
        io = FakeIO(others=4); el, c, clock = controller(io)
        el.tick(); el.tick()
        io.calls.clear()
        c["elastic_vram"] = False
        io.others = 1 * GiB; io.fail = {0}
        clock.t += 200                                          # past the cooldown
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0), ("swap", "big", 1)])
        self.assertNotIn(("giveup", "big"), io.calls)
        ev = el.status()["events"][0]
        self.assertTrue(ev["ok"])
        self.assertIn("didn't load", ev["note"])

    def test_grow_now_never_unloads(self):
        io = FakeIO(others=4); el, _, _ = controller(io)
        el.tick(); el.tick()
        io.calls.clear()
        io.others = 1 * GiB; io.fail = {0, 1}
        out = el.grow_now("big")
        self.assertTrue(out["ok"])
        self.assertEqual(io.at["big"], 2)
        self.assertNotIn(("giveup", "big"), io.calls)

    def test_status_while_off_lists_the_waiting_models(self):
        io = FakeIO(others=4); el, c, clock = controller(io)
        el.tick(); el.tick()
        self.assertEqual(el.status()["waiting"], [])
        c["elastic_vram"] = False
        el.tick()                                               # step 0 does not fit yet
        st = el.status()
        self.assertEqual(st["waiting"], ["big"])
        self.assertEqual(st["models"][0]["at"], 1)
        self.assertEqual(st["others_mib"], {0: 4 * GiB})
        io.others = 1 * GiB
        clock.t += 200
        el.tick()                                               # moves it back
        self.assertEqual(el.status()["waiting"], [])


class OffPathBackoffTest(unittest.TestCase):
    def test_a_failing_step_zero_is_retried_with_a_growing_cooldown(self):
        io = FakeIO(others=4); el, c, clock = controller(io)
        el.tick(); el.tick()
        io.calls.clear()
        c["elastic_vram"] = False
        io.others = 1 * GiB; io.fail = {0}
        clock.t += 200                                          # past the cooldown
        for _ in range(6):
            el.tick()
            clock.t += 5
        self.assertEqual(swaps(io), [("swap", "big", 0), ("swap", "big", 1)])
        clock.t += 250                                          # past the doubled cooldown
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0), ("swap", "big", 1)] * 2)


class RobustnessTest(unittest.TestCase):
    def test_a_failing_tidy_does_not_stop_shrinking(self):
        io = FakeIO(others=4)

        def boom():
            raise OSError("ini locked")
        io.tidy = boom
        el, _, _ = controller(io)
        with mock.patch("sys.stderr", new_callable=_io.StringIO) as err:
            el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])
        self.assertIn("ini locked", err.getvalue())

    def test_safe_tick_says_what_went_wrong(self):
        io = FakeIO()

        def boom(gpus):
            raise OSError("router down")
        io.models = boom
        el, _, _ = controller(io)
        with mock.patch("sys.stderr", new_callable=_io.StringIO) as err:
            el.safe_tick()
        self.assertIn("router down", err.getvalue())

    def test_a_grow_that_blows_up_is_logged_and_shows_in_the_events(self):
        io = FakeIO(others=4); el, _, _ = controller(io)
        el.tick(); el.tick()

        def boom(gpus):
            raise OSError("router down")
        io.models = boom
        with mock.patch("sys.stderr", new_callable=_io.StringIO) as err:
            el.safe_grow("big")
        self.assertIn("router down", err.getvalue())
        ev = el.status()["events"][0]
        self.assertEqual((ev["model"], ev["ok"], ev["reason"]), ("big", False, "you asked"))
        self.assertIn("router down", ev["note"])

    def test_status_tells_the_ui_the_ladder_size_and_the_full_size(self):
        io = FakeIO(others=4); el, _, _ = controller(io)
        el.tick(); el.tick(); el.tick()                         # the third sees the shrink
        m = el.status()["models"][0]
        self.assertEqual((m["at"], m["steps"], m["full"]), (1, len(LADDER), "64/64"))


class FreshStatusTest(unittest.TestCase):
    """Every tick leaves status() saying what is true now."""
    def shrunk(self):
        io = FakeIO(others=4); el, c, clock = controller(io)
        el.tick(); el.tick(); el.tick()
        self.assertEqual(el.status()["models"][0]["at"], 1)
        return io, el, c, clock

    def test_turned_off_grow_back_updates_the_chip_at_once(self):
        io, el, c, clock = self.shrunk()
        c["elastic_vram"] = False
        io.others = 1 * GiB
        clock.t += 200
        el.tick()                                               # grows it back to 0
        st = el.status()
        self.assertEqual([(m["id"], m["at"]) for m in st["models"]], [("big", 0)])
        self.assertEqual(st["waiting"], [])

    def test_grow_now_updates_the_chip_at_once(self):
        io, el, _, _ = self.shrunk()
        io.others = 1 * GiB
        self.assertTrue(el.grow_now("big")["ok"])
        self.assertEqual(el.status()["models"][0]["at"], 0)

    def test_nothing_to_look_at_clears_the_models(self):
        for on in (True, False):
            io, el, c, clock = self.shrunk()
            c["elastic_vram"] = on
            io.snapshot = lambda: {}                            # nothing of ours left
            el.tick()
            st = el.status()
            self.assertEqual((st["models"], st["waiting"], st["others_mib"]), ([], [], {}), on)

    def test_a_failed_look_clears_the_models(self):
        io, el, _, _ = self.shrunk()

        def boom():
            raise OSError("nvidia-smi hung")
        io.snapshot = boom
        with mock.patch("sys.stderr", new_callable=_io.StringIO):
            el.safe_tick()
        self.assertEqual(el.status()["models"], [])

    def test_standby_clears_the_models(self):
        io, el, _, _ = self.shrunk()
        io.game = True                                          # Game mode unloads everything
        el.tick()
        st = el.status()
        self.assertTrue(st["standby"])
        self.assertEqual(st["models"], [])


GROW_TICKS = elastic.GROW_HITS + 5


def tight(io, fp0, others=300):
    """A model whose own size leaves only a little free on a 16 GiB GPU."""
    io.others = others
    io.fps["big"] = [{0: fp0}, {0: 10 * GiB}, {0: 6 * GiB}, {0: 1 * GiB}]


FULL = 16 * GiB - 300 - 212          # leaves 212 MiB free next to 300 MiB of other apps


class NoReasonTest(unittest.TestCase):
    """A model that nearly fills the GPU is not shrunk unless other apps grew,
    and comes back to what the user had once they leave."""
    def watched_before_load(self, fp0=FULL):
        io = FakeIO(); tight(io, fp0)
        loaded, io.at = io.at, {}
        el, c, clock = controller(io)
        el.tick()                                               # calm: nothing loaded yet
        io.at = loaded                                          # the user loads it
        return io, el, c, clock

    def squeeze_and_release(self, io, el, clock, app=3 * GiB):
        io.others += app
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])       # a real squeeze still shrinks
        io.calls.clear()
        io.others -= app
        for _ in range(GROW_TICKS):
            clock.t += 30
            el.tick()

    def test_a_full_model_alone_is_left_alone(self):
        io, el, _, clock = self.watched_before_load()
        for _ in range(20):
            clock.t += 5
            el.tick()
        self.assertEqual(swaps(io), [])
        self.assertEqual(el.status()["models"][0]["at"], 0)

    def test_a_full_model_comes_back_after_a_real_squeeze_and_stays(self):
        io, el, _, clock = self.watched_before_load()
        for _ in range(3):
            el.tick()                                           # full, nobody else grew
        self.squeeze_and_release(io, el, clock)
        self.assertEqual(swaps(io), [("swap", "big", 0)])
        io.calls.clear()
        for _ in range(40):                                     # no shrink, grow, shrink
            clock.t += 5
            el.tick()
        self.assertEqual(swaps(io), [])

    def test_the_room_the_user_had_is_enough_to_grow_back(self):
        # 2.08 GB free at full size: less than headroom + the 1 GiB grow margin
        io = FakeIO(); tight(io, 16 * GiB - 300 - 2130)
        el, _, clock = controller(io)
        el.tick()                                               # calm, with the model loaded
        self.squeeze_and_release(io, el, clock)
        self.assertEqual(swaps(io), [("swap", "big", 0)])

    def test_grow_now_and_turning_off_need_only_the_room_the_user_had(self):
        for how in ("now", "off"):
            io, el, c, clock = self.watched_before_load()
            el.tick()
            io.others += 3 * GiB
            el.tick(); el.tick()
            io.calls.clear()
            io.others -= 3 * GiB
            clock.t += 200
            if how == "now":
                self.assertEqual(el.grow_now("big"), {"ok": True, "to": "64/64"})
            else:
                c["elastic_vram"] = False
                el.tick()
            self.assertEqual(swaps(io), [("swap", "big", 0)], how)

    def test_pressure_without_growth_does_not_count_toward_a_shrink(self):
        io, el, _, _ = self.watched_before_load()
        el.tick()                                               # pressure, others unchanged
        io.others += 3 * GiB
        el.tick()                                               # first sighting of growth
        self.assertEqual(swaps(io), [])
        el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])

    def test_noise_below_the_floor_is_not_growth(self):
        io, el, _, _ = self.watched_before_load()
        io.others += elastic.GROWTH_MIN_MIB
        for _ in range(5):
            el.tick()
        self.assertEqual(swaps(io), [])

    def test_middle_steps_keep_the_margin(self):
        # only step 0 is relaxed: a grow that can't reach it needs the full
        # headroom + margin for a middle step
        io = FakeIO()
        io.others = 300
        io.fps["big"] = [{0: 16 * GiB - 300 - 2130}, {0: 12 * GiB}, {0: 8 * GiB}, {0: 1 * GiB}]
        el, _, clock = controller(io)
        el.tick()
        io.others += 8 * GiB                                    # down to step 3
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 3)])
        io.calls.clear()
        io.others = 2200                 # step 1 would leave 1.85 GiB: not head + margin
        for _ in range(GROW_TICKS):
            clock.t += 30
            el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 2)])


class FirstLookTest(unittest.TestCase):
    """Turned on with the model already loaded (or the panel restarted under
    a running router): the first look is the baseline, not a squeeze."""
    def test_pressure_on_the_first_look_with_no_growth_is_left_alone(self):
        # 16 GiB GPU, a 15.5 GB model, 300 MiB of other apps, on from tick 1
        io = FakeIO(); tight(io, FULL)
        el, _, clock = controller(io, fresh=True)
        for _ in range(40):
            clock.t += 5
            el.tick()
        self.assertEqual(swaps(io), [])
        self.assertEqual(el.status()["models"][0]["at"], 0)

    def test_a_real_app_after_the_first_look_shrinks_and_grows_back(self):
        io = FakeIO(); tight(io, FULL)
        el, _, clock = controller(io, fresh=True)
        el.tick()                                               # the baseline, under pressure
        io.others += 3 * GiB
        el.tick(); el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 1)])
        io.calls.clear()
        io.others -= 3 * GiB
        for _ in range(GROW_TICKS):
            clock.t += 30
            el.tick()
        self.assertEqual(swaps(io), [("swap", "big", 0)])

    def test_turning_it_on_again_takes_a_fresh_look(self):
        io = FakeIO(); tight(io, FULL)
        el, c, clock = controller(io, fresh=True)
        el.tick()
        c["elastic_vram"] = False
        el.tick()
        io.others += 1 * GiB                                    # an app started while off
        c["elastic_vram"] = True
        for _ in range(10):
            clock.t += 5
            el.tick()
        self.assertEqual(swaps(io), [])


if __name__ == "__main__":
    unittest.main()
