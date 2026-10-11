import conftest_paths  # noqa: F401
import io
import json
import time
import unittest
import urllib.error

import loadprog


def ev(model, event, data=None):
    obj = {"model": model, "event": event}
    if data is not None:
        obj["data"] = data
    return obj


def loading(model, value, current="text_model", stages=("text_model",)):
    return ev(model, "status_change", {"status": "loading",
              "progress": {"stages": list(stages), "current": current, "value": value}})


class NoteTest(unittest.TestCase):
    def setUp(self):
        self.lp = loadprog.LoadProgress(lambda: ("http://x/models/sse", {}))

    def test_one_stage_is_its_own_percent(self):
        self.lp.note(loading("a", 0.43))
        self.assertEqual(self.lp.get("a"), {"pct": 43, "stage": "text_model"})

    def test_later_stages_add_the_earlier_ones(self):
        st = ("text_model", "spec_model", "mmproj_model")
        self.lp.note(loading("a", 0.5, "spec_model", st))
        self.assertEqual(self.lp.get("a")["pct"], 50)      # (1 + 0.5) / 3
        self.assertEqual(self.lp.get("a")["stage"], "spec_model")

    def test_a_stage_marker_without_a_value_is_the_start_of_that_stage(self):
        st = ("text_model", "mmproj_model")
        self.lp.note(loading("a", 1.0, "text_model", st))
        self.lp.note(ev("a", "status_change", {"status": "loading",
                        "progress": {"stage": "mmproj_model"}}))
        self.assertEqual(self.lp.get("a"), {"pct": 50, "stage": "mmproj_model"})

    def test_never_goes_backwards_within_a_load(self):
        self.lp.note(loading("a", 0.6))
        self.lp.note(loading("a", 0.2))
        self.assertEqual(self.lp.get("a")["pct"], 60)

    def test_finished_or_unloaded_forgets_it(self):
        self.lp.note(loading("a", 0.6))
        self.lp.note(ev("a", "status_change", {"status": "loaded"}))
        self.assertIsNone(self.lp.get("a"))
        self.lp.note(loading("a", 0.1))        # a fresh load starts from scratch
        self.assertEqual(self.lp.get("a")["pct"], 10)

    def test_junk_is_ignored(self):
        for bad in ({}, {"model": "a"}, ev("a", "status_change", "x"),
                    ev("a", "download_progress", {"x": 1}),
                    loading("a", "nan?"), ev("a", "status_change", {"status": "loading"})):
            self.lp.note(bad)
        self.assertIsNone(self.lp.get("a"))

    def test_percent_is_clamped(self):
        self.lp.note(loading("a", 7))
        self.assertEqual(self.lp.get("a")["pct"], 100)


class StreamTest(unittest.TestCase):
    def test_reads_sse_data_lines(self):
        lp = loadprog.LoadProgress(lambda: ("u", {}))
        body = (b": keepalive\n\n"
                b"data: " + json.dumps(loading("a", 0.25)).encode() + b"\n\n"
                b"data: not json\n\n"
                b"data: " + json.dumps(loading("b", 0.5)).encode() + b"\n\n")
        lp.consume(io.BytesIO(body))
        self.assertEqual(lp.get("a")["pct"], 25)
        self.assertEqual(lp.get("b")["pct"], 50)

    def test_a_router_without_the_endpoint_is_not_hammered(self):
        calls = []

        def opener(req, timeout=None):
            calls.append(req.full_url)
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

        lp = loadprog.LoadProgress(lambda: ("http://r/models/sse", {}), opener=opener)
        lp._run_once()
        self.assertTrue(lp.unsupported())
        lp.ensure()                       # no thread while unsupported
        self.assertFalse(lp._running)
        self.assertEqual(len(calls), 1)

    def test_sends_the_auth_header(self):
        seen = {}

        def opener(req, timeout=None):
            seen.update(req.headers)
            return io.BytesIO(b"")

        lp = loadprog.LoadProgress(lambda: ("http://r/models/sse", {"Authorization": "Bearer k"}),
                                   opener=opener)
        lp._run_once()
        self.assertEqual(seen.get("Authorization"), "Bearer k")

    def test_ensure_starts_one_thread_that_stops_when_nobody_asks(self):
        opened = []

        def opener(req, timeout=None):
            opened.append(1)
            return io.BytesIO(b"")

        lp = loadprog.LoadProgress(lambda: ("http://r/models/sse", {}), opener=opener, idle_s=0.2, retry_s=0.01)
        lp.ensure(); lp.ensure()
        self.assertTrue(lp._running)
        deadline = time.time() + 3
        while lp._running and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(lp._running)
        self.assertGreaterEqual(len(opened), 1)


class AttachTest(unittest.TestCase):
    def test_only_loading_rows_get_it_and_the_watcher_is_woken(self):
        lp = loadprog.LoadProgress(lambda: ("u", {}))
        woken = []
        lp.ensure = lambda: woken.append(1)
        lp.note(loading("a", 0.3))
        rows = [{"id": "a", "status": "loading"}, {"id": "b", "status": "loaded"},
                {"id": "c", "status": "loading", "process": {"port": 1}}]
        lp.attach(rows)
        self.assertEqual(rows[0]["load_progress"]["pct"], 30)
        self.assertNotIn("load_progress", rows[1])
        self.assertNotIn("load_progress", rows[2])     # own process: not the router's
        self.assertEqual(woken, [1])

    def test_nothing_loading_never_wakes_it(self):
        lp = loadprog.LoadProgress(lambda: ("u", {}))
        lp.ensure = lambda: self.fail("woken")
        lp.attach([{"id": "b", "status": "loaded"}])


if __name__ == "__main__":
    unittest.main()
