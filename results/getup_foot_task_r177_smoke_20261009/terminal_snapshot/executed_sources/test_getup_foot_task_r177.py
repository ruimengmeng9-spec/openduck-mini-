import inspect
import unittest
import mujoco
import numpy as np
from diagnostics import train_getup_foot_task_r177 as run


class FootTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    def setUp(self):
        self.fk=run.FootKinematics(self.model);self.x=np.zeros(55,np.float32)
    def call(self,c=run.PROBE,x=None,n=None,x0=None,n0=None):
        return run.foot_feedback(c,self.x if x is None else x,self.x if n is None else n,self.x if x0 is None else x0,self.x if n0 is None else n0,self.fk)
    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(run.foot_feedback).parameters),['parameters','current','nominal','initial','initial_nominal','kinematics'])
    def test_numeric_jacobian_all_ten_leg_joints(self):
        rng=np.random.default_rng(277)
        for _ in range(10):
            q=rng.uniform(-.3,.3,14).astype(np.float32);_,jac=self.fk.measure(q,True)
            for leg,group in enumerate(self.fk.groups):
                for i,axis in enumerate(group):
                    a=q.copy();b=q.copy();a[axis]+=np.float32(.001);b[axis]-=np.float32(.001)
                    derivative=(self.fk.measure(a)[leg]-self.fk.measure(b)[leg])/float(a[axis]-b[axis])
                    np.testing.assert_allclose(derivative,jac[leg,:,i],atol=1e-7,rtol=3e-6)
    def test_leg_mapping_and_cross_leg_independence(self):
        original=self.fk.measure(np.zeros(14,np.float32))
        for leg,group in enumerate(self.fk.groups):
            q=np.zeros(14,np.float32);q[group[2]]=.1;fresh=self.fk.measure(q)
            self.assertGreater(np.linalg.norm(fresh[leg]-original[leg]),.001)
            np.testing.assert_array_equal(fresh[1-leg],original[1-leg])
    def test_actual_nominal_scalar_exact_zero_sequence(self):
        rng=np.random.default_rng(277)
        for _ in range(529):
            x=rng.normal(size=55).astype(np.float32)
            extra,error,_,dq=self.call(x=x,n=x)
            for value in (extra,error,dq):np.testing.assert_array_equal(value,np.zeros_like(value))
    def test_initial_nonzero_error_exact_zero(self):
        x=np.random.default_rng(277).normal(size=55).astype(np.float32)
        for value in (self.call(x=x,x0=x)[i] for i in (0,1,3)):np.testing.assert_array_equal(value,np.zeros_like(value))
    def test_zero_amplitude_preserves_merge_identity(self):
        x=self.x.copy();x[8]=.1;extra,_,_,_=self.call(c=np.zeros(2),x=x)
        np.testing.assert_array_equal(extra,np.zeros(14));target=np.zeros(14)
        self.assertIs(run.local.merge_target(target,target,extra,np.arange(14),-np.ones(14),np.ones(14)),target)
    def test_no_other_native_channel_dependency(self):
        x=np.random.default_rng(277).normal(size=55).astype(np.float32);x[6:20]=0
        for a,b in zip(self.call(x=x),self.call()):np.testing.assert_array_equal(a,b)
    def test_head_position_has_no_task_effect(self):
        x=self.x.copy();x[11:15]=[.1,-.2,.3,-.4]
        for a,b in zip(self.call(x=x),self.call()):np.testing.assert_array_equal(a,b)
    def test_fixed_dls_task_direction_nonpositive(self):
        rng=np.random.default_rng(277)
        for _ in range(100):
            x=self.x.copy();x[6:20]=rng.uniform(-.2,.2,14)
            _,e,j,dq=self.call(x=x)
            for ei,ji,di in zip(e,j,dq):self.assertLessEqual(float(ei@(ji@di)),1e-15)
    def test_private_root_fixed_and_model_not_mutated(self):
        before={name:np.asarray(getattr(self.model,name)).copy() for name in ('body_mass','body_pos','body_quat','jnt_range','geom_contype','actuator_forcerange')}
        self.call(x=np.random.default_rng(277).normal(size=55).astype(np.float32))
        np.testing.assert_array_equal(self.fk.data.qpos[:7],[0,0,0,1,0,0,0])
        self.assertEqual(self.fk.data.time,0)
        for name,value in before.items():np.testing.assert_array_equal(value,getattr(self.model,name))
    def test_output_and_original_combined_bounds(self):
        rng=np.random.default_rng(277)
        for _ in range(100):
            extra,_,_,_=self.call(c=np.full(2,run.BOUND),x=rng.normal(size=55).astype(np.float32))
            self.assertLessEqual(np.abs(extra).max(),.18)
            np.testing.assert_array_equal(extra[5:9],np.zeros(4))
            target=rng.uniform(-.18,.18,14);reference=np.zeros(14)
            adjusted=run.local.merge_target(target,reference,extra,np.arange(14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(adjusted-reference).max(),.18+1e-12)
    def test_invalid_nonfinite_and_amplitudes(self):
        for c in ([1],[-.001,0],[0,.05001],[np.nan,0]):
            with self.assertRaises(ValueError):self.call(c=c)
        with self.assertRaises(ValueError):self.call(x=np.full(55,np.nan))
        with self.assertRaises(ValueError):self.fk.measure(np.zeros(13))


if __name__=='__main__':unittest.main()
