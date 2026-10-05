import unittest
from diagnostics.probe_getup_temporal_gate_r104 import enabled


class TemporalGateTests(unittest.TestCase):
    def test_early_boundary(self):
        self.assertTrue(enabled('early',129));self.assertFalse(enabled('early',130))

    def test_late_boundary(self):
        self.assertFalse(enabled('late',129));self.assertTrue(enabled('late',130))

    def test_local_boundary(self):
        self.assertFalse(enabled('local',79));self.assertTrue(enabled('local',80))
        self.assertFalse(enabled('local',130))

    def test_prefix_boundary(self):
        self.assertTrue(enabled('prefix',307));self.assertFalse(enabled('prefix',308))

    def test_no_home_residual(self):
        for name in ('early','late','local','prefix'):
            self.assertFalse(enabled(name,529));self.assertFalse(enabled(name,2278))


if __name__=='__main__':unittest.main()
