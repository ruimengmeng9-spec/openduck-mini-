from types import SimpleNamespace
import unittest
import numpy as np
from unittest.mock import Mock
from diagnostics.train_getup_early_head_r86 import edited_head,rank,TRAIN,HELDOUT
from diagnostics.audit_getup_substep_contact_r85 import ContactObserver


class HeadCoordinationTest(unittest.TestCase):
    def make(self):
        ids={'neck_pitch':5,'head_pitch':6}
        sim=SimpleNamespace(model=SimpleNamespace(actuator=lambda n:SimpleNamespace(id=ids[n])),
                            lower=np.full(14,-2.),upper=np.full(14,2.))
        ck={'prefix_targets':np.zeros((450,14)),'prefix_phases':np.r_[np.zeros(308,dtype=int),np.ones(142,dtype=int)]}
        return sim,ck

    def test_identity(self):
        sim,ck=self.make();np.testing.assert_array_equal(edited_head(sim,ck,np.zeros(4))['prefix_targets'],ck['prefix_targets'])

    def test_only_head_inside_window_changes(self):
        sim,ck=self.make();t=edited_head(sim,ck,np.array([.1,-.1,.2,-.2]))['prefix_targets']
        np.testing.assert_array_equal(t[:,[i for i in range(14) if i not in (5,6)]],0.)
        np.testing.assert_array_equal(t[:26],0.);np.testing.assert_array_equal(t[175:],0.)
        self.assertAlmostEqual(t[75,5],.1);self.assertAlmostEqual(t[125,6],-.2)

    def test_bounds_rejected(self):
        sim,ck=self.make()
        for x in (np.zeros(3),np.full(4,.36),np.full(4,np.nan)):
            with self.assertRaises(ValueError):edited_head(sim,ck,x)

    def test_success_count_dominates_quality(self):
        self.assertGreater(rank(dict(nominal_success=True,successes=14,quality=-99)),
                           rank(dict(nominal_success=True,successes=13,quality=100)))

    def test_observer_delegates_once_and_returns_original_result(self):
        marker=object();original=Mock(return_value=marker);observer=ContactObserver(original)
        model=SimpleNamespace(geom=lambda n:SimpleNamespace(id=0));data=SimpleNamespace(contact=[])
        self.assertIs(observer(model,data),marker);original.assert_called_once_with(model,data)
        self.assertEqual(observer.calls,1);self.assertFalse(observer.events)

    def test_disjoint_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT));self.assertFalse(set(range(780000,785040))&set(HELDOUT))


if __name__=='__main__':unittest.main()
