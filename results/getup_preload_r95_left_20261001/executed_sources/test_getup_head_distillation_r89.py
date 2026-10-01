import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from diagnostics.train_getup_head_distillation_r89 import inputs,predict,head_offset,run,TRAIN,HELDOUT,CAP
from diagnostics.search_getup_reference_feedback_r64 import residual


class DistillationTest(unittest.TestCase):
    def test_no_root_state(self):
        sim=SimpleNamespace(data=SimpleNamespace(qpos=np.arange(21.),qvel=np.arange(20.)),
            qadr=np.arange(7,21),vadr=np.arange(6,20),home=np.zeros(14),prev=np.zeros(14))
        x=inputs(sim,np.arange(4.),1.,np.arange(10))
        sim.data.qpos[:7]=999;sim.data.qvel[:6]=999
        np.testing.assert_array_equal(x,inputs(sim,np.arange(4.),1.,np.arange(10)))
        self.assertEqual(x.shape,(35,))

    def test_cap_and_frozen_legs(self):
        e=np.ones(4);g=np.ones(8);out=head_offset(e,g,2.,[100.,-100.])
        np.testing.assert_array_equal(out[:8],residual(e,g)[:8])
        np.testing.assert_array_equal(out[8:],[CAP,-CAP])

    def test_inactive_window(self):
        e=np.ones(4);g=np.ones(8)
        for t in (0.,.5,3.5,4.):
            np.testing.assert_array_equal(head_offset(e,g,t,[1.,1.]),residual(e,g))

    def test_zero_strength_delegates(self):
        with patch('diagnostics.train_getup_head_distillation_r89.rollout',return_value=('base',[])) as fn:
            self.assertEqual(run(None,None,{},None,None,None,None,None,network={},strength=0.)[:2],('base',[]))
            fn.assert_called_once()

    def test_network_bounds(self):
        net={'center':np.zeros(35),'scale':np.ones(35),'w1':np.zeros((35,64)),
             'b1':np.zeros(64),'w2':np.zeros((64,2)),'b2':np.array([100.,-100.])}
        np.testing.assert_array_equal(predict(net,np.zeros(35)),[2.,-2.])

    def test_new_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        for old in (780000,781000,782000,784000,786000,787000):
            self.assertFalse(set(range(old,old+40))&set(HELDOUT))


if __name__=='__main__':unittest.main()
