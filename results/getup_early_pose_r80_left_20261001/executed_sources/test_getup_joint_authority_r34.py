import unittest

import numpy as np

from diagnostics.probe_getup_joint_authority_r34 import candidate_targets
from diagnostics.probe_getup_joint_pairs_r34 import pair_targets
from diagnostics.search_getup_joint_sequences_r34 import target_library, rank_key


class CandidateTargetTests(unittest.TestCase):
    def test_one_joint_at_a_time_and_exact_trial_count(self):
        home = np.zeros(14)
        lower = np.full(14, -0.4)
        upper = np.full(14, 0.4)
        rows = list(candidate_targets(home, lower, upper, 0.6))
        self.assertEqual(len(rows), 29)
        np.testing.assert_array_equal(rows[0][3], home)
        for name, joint, sign, target in rows[1:]:
            self.assertTrue(name.startswith(f"joint_{joint}_"))
            self.assertEqual(np.count_nonzero(target - home), 1)
            self.assertEqual(target[joint], sign * 0.4)

    def test_rejects_invalid_ranges_and_amplitude(self):
        zero = np.zeros(14)
        with self.assertRaises(ValueError):
            list(candidate_targets(zero, zero - 1, zero + 1, 0))
        with self.assertRaises(ValueError):
            list(candidate_targets(zero, zero + 1, zero + 2, 0.5))
        with self.assertRaises(ValueError):
            list(candidate_targets(np.zeros(13), zero - 1, zero + 1, 0.5))

    def test_pairs_use_two_distinct_joints_and_never_average_opposite_signs(self):
        home = np.zeros(14)
        ranked = [dict(name=f"j{j}_{sign}", joint=j, sign=sign)
                  for j, sign in [(0, 1), (0, -1), (1, -1), (2, 1)]]
        pairs = list(pair_targets(home, home - 0.4, home + 0.4,
                                  ranked, 0.6, top_n=4))
        self.assertEqual(len(pairs), 5)
        for _, joints, target in pairs:
            self.assertEqual(len(set(joints)), 2)
            self.assertEqual(np.count_nonzero(target), 2)
            self.assertLessEqual(np.max(np.abs(target)), 0.4)

    def test_stage_library_uses_exact_source_targets(self):
        home = np.zeros(14)
        paired = []
        for j in range(6):
            offsets = np.zeros(14)
            offsets[j] = 0.3
            paired.append(dict(name=f"pair_{j}", target_offsets_rad=offsets.tolist()))
        singles = [dict(name="single_a", joint=8, actual_target_offset_rad=-0.2),
                   dict(name="single_b", joint=9, actual_target_offset_rad=0.2)]
        firsts, seconds = target_library(
            dict(valid_ranked_by_max_up_gain=paired),
            dict(valid_ranked_by_max_up_gain=singles),
            home, home - 1, home + 1)
        self.assertEqual((len(firsts), len(seconds)), (4, 9))
        np.testing.assert_array_equal(firsts[0][1], paired[0]["target_offsets_rad"])
        np.testing.assert_array_equal(seconds[-1][1], home)

    def test_physical_validity_outweighs_search_shaping(self):
        valid = dict(physically_valid=True, longest_strict_standing_s=0,
                     best_progress=0.1, best_up_z=0.1)
        invalid = dict(physically_valid=False, longest_strict_standing_s=0,
                       best_progress=100, best_up_z=1)
        self.assertGreater(rank_key(valid), rank_key(invalid))


if __name__ == "__main__":
    unittest.main()
