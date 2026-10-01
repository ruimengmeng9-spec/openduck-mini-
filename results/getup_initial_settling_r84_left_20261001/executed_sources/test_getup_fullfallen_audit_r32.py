import unittest
from diagnostics.audit_getup_fullfallen_r32 import summarize


class FailureClassificationTests(unittest.TestCase):
    def test_strict_boundaries_and_overlapping_causes(self):
        row = dict(pose='prone', valid=False, training_success=False,
                   peaks=dict(floor=.01, self=.004, joint=.08, force=3.23, slew=5.24),
                   final=dict(finite=True, up_z=.1))
        result = summarize([row])['prone']
        self.assertEqual(result['physical_failures'], 1)
        self.assertEqual(result['violation_counts'], dict(floor=1, self=1, joint=1))
        self.assertEqual(result['discovery_successes'], 0)

    def test_discovery_label_never_certifies_full_task(self):
        row = dict(pose='standing', valid=True, training_success=True,
                   peaks={}, final=dict(finite=True, up_z=1.))
        report = summarize([row])
        self.assertEqual(report['standing']['discovery_successes'], 1)
        self.assertEqual(report['prone']['episodes'], 0)
        self.assertNotIn('full_task_completed', report)


if __name__ == '__main__':
    unittest.main()
