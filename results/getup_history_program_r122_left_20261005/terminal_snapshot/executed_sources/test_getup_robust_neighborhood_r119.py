import unittest
import numpy as np
from diagnostics.search_getup_robust_neighborhood_r119 import (
    proposals, groups_from_pairs, rank_group, full_group_pass, recipe_key)


class RobustTeacherTest(unittest.TestCase):
    def test_fixed_seed_bounded_common_proposals(self):
        def state():
            return dict(profile=1, knots=np.zeros((6, 10)), mean=np.zeros((6, 10)),
                std=np.full((6, 10), .1), rng=np.random.default_rng(219))
        members = [(0, np.zeros((6, 10)))] * 3
        a, b = proposals(state(), members, 14), proposals(state(), members, 14)
        self.assertEqual(len(a), 14)
        for (p, k), (q, l) in zip(a, b):
            self.assertEqual(recipe_key(p, k), recipe_key(q, l))
            np.testing.assert_array_equal(k, l)
            self.assertLessEqual(np.abs(k).max(), 1.)

    def test_unseen_data_rejected_and_group_order_retained(self):
        pairs = [dict(source_case=s, target_case=t) for s, targets in
            [(None, [769002, 773005]), (769000, [3160029, 3160025]), (3160001, [773014, 773008])]
            for t in targets]
        self.assertEqual(groups_from_pairs(pairs)[0], [None, 769002, 773005])
        pairs[0]['target_case'] = 3200000
        with self.assertRaises(ValueError):
            groups_from_pairs(pairs)

    def test_short_success_cannot_pass_complete_gate(self):
        row = dict(valid=True, full_path=False, controls=629, entry_time_s=11., strict_tail_s=1.)
        self.assertFalse(full_group_pass([row.copy() for _ in range(3)]))
        row.update(full_path=True, controls=2279, strict_tail_s=30.)
        self.assertTrue(full_group_pass([row.copy() for _ in range(3)]))
        rows = [row.copy() for _ in range(3)]
        rows[0]['entry_time_s'] = 12.01
        self.assertFalse(full_group_pass(rows))
        rows[0]['entry_time_s'] = 11.
        rows[1]['valid'] = False
        self.assertFalse(full_group_pass(rows))

    def test_invalid_candidate_cannot_outrank_all_valid(self):
        good = [dict(valid=True, training_success=False, return_sum=-10.) for _ in range(3)]
        bad = [dict(valid=True, training_success=True, return_sum=100.) for _ in range(3)]
        bad[0]['valid'] = False
        self.assertGreater(rank_group(good), rank_group(bad))
        self.assertEqual(rank_group(bad), rank_group(list(reversed(bad))))


if __name__ == '__main__':
    unittest.main()
