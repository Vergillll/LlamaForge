import conftest_paths  # noqa: F401
import unittest
from unittest import mock
import hardware, osplat


class TestDetectRamGb(unittest.TestCase):
    def test_converts_bytes_to_gb(self):
        orig = osplat.total_ram_bytes
        osplat.total_ram_bytes = lambda: 32 * 1000 * 1000 * 1000
        try:
            self.assertAlmostEqual(hardware.detect_ram_gb(), 32.0, delta=0.1)
        finally:
            osplat.total_ram_bytes = orig

    def test_zero_when_unknown(self):
        orig = osplat.total_ram_bytes
        osplat.total_ram_bytes = lambda: 0
        try:
            self.assertEqual(hardware.detect_ram_gb(), 0.0)
        finally:
            osplat.total_ram_bytes = orig


class TestRamUsage(unittest.TestCase):
    """Live system RAM for the Stats tab, in MiB like the GPU rows."""
    GIB = 1024 ** 3

    def usage(self, total, available):
        with mock.patch.object(osplat, "total_ram_bytes", return_value=total), \
             mock.patch.object(osplat, "available_ram_bytes", return_value=available):
            return osplat.ram_usage()

    def test_used_is_total_minus_available(self):
        self.assertEqual(self.usage(64 * self.GIB, 40 * self.GIB),
                         {"used": 24 * 1024, "total": 64 * 1024})

    def test_none_when_either_side_is_unknown(self):
        self.assertIsNone(self.usage(0, 8 * self.GIB))
        self.assertIsNone(self.usage(64 * self.GIB, 0))

    def test_available_above_total_reads_as_nothing_used(self):
        self.assertEqual(self.usage(16 * self.GIB, 20 * self.GIB)["used"], 0)

    def test_this_machine(self):
        u = osplat.ram_usage()
        if u is not None:
            self.assertGreater(u["total"], 0)
            self.assertLessEqual(u["used"], u["total"])


if __name__ == "__main__":
    unittest.main()
