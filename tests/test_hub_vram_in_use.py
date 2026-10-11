"""Discover reports VRAM in use right now next to total VRAM (asked on X)."""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import routes

GPUS = [{"index": 0, "name": "a", "used": 3000, "total": 16000},
        {"index": 1, "name": "b", "used": 1000, "total": 16000}]


class VramInUseTest(unittest.TestCase):
    def test_snapshot_sums_total_and_used(self):
        with mock.patch.object(routes, "_gpu_telemetry", return_value=GPUS):
            self.assertEqual(routes.vram_snapshot_mib(), (32000, 4000))

    def test_an_error_row_counts_as_nothing(self):
        with mock.patch.object(routes, "_gpu_telemetry", return_value=[{"error": "no nvidia-smi"}]):
            self.assertEqual(routes.vram_snapshot_mib(), (0, 0))

    def test_gguf_search_returns_both(self):
        req = mock.Mock(body={"query": "x", "sort": "downloads"})
        with mock.patch.object(routes, "_gpu_telemetry", return_value=GPUS), \
             mock.patch.object(routes.hub, "search", return_value=[]), \
             mock.patch.object(routes, "installed_repos", return_value=[]):
            code, out = routes.post_hub_search(req)
        self.assertEqual(code, 200)
        self.assertEqual((out["vram_mib"], out["vram_used_mib"]), (32000, 4000))

    def test_vllm_search_returns_both(self):
        req = mock.Mock(body={"query": "x", "sort": "downloads"})
        with mock.patch.object(routes, "_gpu_telemetry", return_value=GPUS), \
             mock.patch.object(routes.vllm_hub, "search", return_value=[]), \
             mock.patch.object(routes, "installed_repos", return_value=[]):
            code, out = routes.post_vllm_hub_search(req)
        self.assertEqual((out["vram_mib"], out["vram_used_mib"]), (32000, 4000))


if __name__ == "__main__":
    unittest.main()
