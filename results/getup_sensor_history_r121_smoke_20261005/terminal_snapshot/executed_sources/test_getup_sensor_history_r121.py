import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.probe_getup_sensor_history_r121 import record_preparation,KNOWN


class HistoryTest(unittest.TestCase):
    def fake(self):
        sim=SimpleNamespace(data=SimpleNamespace(time=0.),values=np.zeros(50,dtype=np.float32),calls=[])
        def step(target):
            sim.calls.append(target.copy());sim.data.time+=.02;sim.values[:]=len(sim.calls)
            return target
        sim.step_target=step
        return sim,step

    def test_post_step_causal_copies_and_unchanged_targets(self):
        sim,old=self.fake();target=np.arange(14,dtype=float)
        def prepare():
            for _ in range(40):sim.step_target(target)
            return 7
        result,frames,times=record_preparation(sim,prepare,lambda s:s.values)
        self.assertEqual(result,7);self.assertIs(sim.step_target,old)
        self.assertEqual(frames.shape,(40,50))
        np.testing.assert_array_equal(frames[:,0],np.arange(1,41))
        np.testing.assert_allclose(times,np.arange(1,41)*.02,rtol=0,atol=1e-12)
        for actual in sim.calls:np.testing.assert_array_equal(actual,target)

    def test_original_method_restored_after_exception(self):
        sim,old=self.fake()
        def prepare():raise RuntimeError('test')
        with self.assertRaises(RuntimeError):record_preparation(sim,prepare,lambda s:s.values)
        self.assertIs(sim.step_target,old)

    def test_incorrect_history_length_and_observation_rejected(self):
        sim,old=self.fake()
        with self.assertRaises(ValueError):record_preparation(sim,lambda:None,lambda s:s.values)
        self.assertIs(sim.step_target,old)
        with self.assertRaises(ValueError):
            record_preparation(sim,lambda:sim.step_target(np.zeros(14)),lambda s:np.zeros(51))
        self.assertIs(sim.step_target,old)

    def test_unseen_namespace_excluded(self):
        self.assertEqual(len(KNOWN),105)
        self.assertTrue(all(s not in KNOWN for s in range(3200000,3200040)))


if __name__=='__main__':unittest.main()
