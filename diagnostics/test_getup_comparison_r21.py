import unittest
from diagnostics.compare_getup_range_r21 import aggregate


class ComparisonTests(unittest.TestCase):
    def test_paired_gain_loss_and_survival_are_separate(self):
        rows=[dict(kind='home',seed=1,initial_hash='a',success=True,steps=200),
              dict(kind='home',seed=2,initial_hash='b',success=False,steps=200),
              dict(kind='model',seed=1,initial_hash='a',success=False,steps=200),
              dict(kind='model',seed=2,initial_hash='b',success=True,steps=200)]
        result=aggregate(rows,('home','model'))
        self.assertEqual(result['model']['successes'],1)
        self.assertEqual(result['model']['full_duration_runs'],2)
        self.assertEqual(result['model']['gained_vs_home'],1)
        self.assertEqual(result['model']['lost_vs_home'],1)

    def test_mismatched_start_rejected(self):
        rows=[dict(kind='home',seed=1,initial_hash='a'),dict(kind='model',seed=1,initial_hash='b')]
        with self.assertRaises(RuntimeError):aggregate(rows,('home','model'))


if __name__=='__main__':unittest.main()
