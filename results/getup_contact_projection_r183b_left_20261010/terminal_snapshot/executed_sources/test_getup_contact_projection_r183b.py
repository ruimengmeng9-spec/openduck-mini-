"""Geometry and causal planner regressions, no dynamic episode or integration."""
import inspect
import unittest
import mujoco
import numpy as np
from diagnostics import probe_getup_contact_projection_r183b as run


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        cls.geometry=run.RelativeFootGeometry(cls.model)
        with np.load(run.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz',allow_pickle=False) as z:
            cls.nominal=z['applied'].copy();cls.contacts=z['observations'][:,48:50].copy()
        cls.q=cls.geometry.home.copy();cls.anchor=cls.geometry.anchor(cls.q,cls.q)

    def test_causal_interface(self):
        self.assertEqual(list(inspect.signature(run.project).parameters),['planned','nominal','contacts','anchor','lower','upper','geometry','enabled'])

    def test_fixed_budget(self):
        self.assertEqual(run.ITERATIONS,3);self.assertEqual(run.STEPS,(1.,.5,.25,.125,.0625))
        self.assertEqual(run.ROTATION_LENGTH_M,.05);self.assertEqual(len(run.CASES),25)

    def test_nominal_scalar_sequence(self):
        a=self.geometry.anchor(self.nominal[0],self.nominal[0])
        for q,contact in zip(self.nominal[:529],self.contacts[:529]):
            v,e,j,t,active=run.project(q,q,contact,a,q-.1,q+.1,self.geometry,True)
            self.assertIs(v,q);np.testing.assert_array_equal(e,np.zeros(6))

    def test_initial_anchor_residual(self):
        q=self.q.copy();q[0]+=.04;q[10]-=.02
        a=self.geometry.anchor(q,self.q)
        np.testing.assert_allclose(self.geometry.residual(q,self.q,a),0,atol=2e-16,rtol=0)
        # Runtime control zero stores anchor and returns the old target without
        # evaluating this roundoff-size residual; first target is exactly old.

    def test_zero_disabled_identity(self):
        q=self.q.copy();q[2]+=.04
        self.assertIs(run.project(q,self.q,[1,1],self.anchor,q-.1,q+.1,self.geometry,False)[0],q)

    def test_single_contact_identity(self):
        q=self.q.copy();q[2]+=.04
        for contacts in ([0,0],[1,0],[0,1]):
            self.assertIs(run.project(q,self.q,contacts,self.anchor,q-.1,q+.1,self.geometry,True)[0],q)

    def test_contact_flags(self):
        self.assertTrue(np.isin(self.contacts,[0,1]).all())
        print('R183b_NOMINAL_CONTACT_COUNTS',np.unique(self.contacts[:529],axis=0,return_counts=True),flush=True)

    def test_head_independence(self):
        a=self.q.copy();b=a.copy();b[5:9]+=.4
        for x,y in zip(self.geometry.pose(a),self.geometry.pose(b)):np.testing.assert_array_equal(x,y)

    def test_private_root_and_model(self):
        arrays=[self.model.body_pos.copy(),self.model.jnt_range.copy(),self.model.actuator_ctrlrange.copy()]
        self.geometry.pose(self.nominal[120]);np.testing.assert_array_equal(self.geometry.data.qpos[:7],[0,0,0,1,0,0,0])
        for a,b in zip(arrays,[self.model.body_pos,self.model.jnt_range,self.model.actuator_ctrlrange]):np.testing.assert_array_equal(a,b)

    def test_no_dynamic_functions(self):
        src=inspect.getsource(run.RelativeFootGeometry)
        for token in ('mj_forward','mj_step','mj_comPos','contact','qvel'):self.assertNotIn(token,src)

    def test_jacobian_finite_difference(self):
        q=self.q.copy();q[self.geometry.legs]+=np.linspace(-.1,.1,10)
        a=self.geometry.anchor(self.q,self.q);j=self.geometry.jacobian(q,self.q,a)
        direction=np.linspace(-.05,.08,10);h=1e-5
        p=q.copy();n=q.copy();p[self.geometry.legs]+=h*direction;n[self.geometry.legs]-=h*direction
        fd=(self.geometry.residual(p,self.q,a)-self.geometry.residual(n,self.q,a))/(2*h)
        np.testing.assert_allclose(j@direction,fd,atol=1e-9,rtol=1e-6)

    def test_two_leg_coupling(self):
        j=self.geometry.jacobian(self.q,self.q,self.anchor)
        self.assertGreater(np.linalg.norm(j[:,:5]),0);self.assertGreater(np.linalg.norm(j[:,5:]),0)
        self.assertGreater(np.linalg.norm(j[:,4]),0);self.assertGreater(np.linalg.norm(j[:,9]),0)

    def test_local_residual_decreases(self):
        q=self.q.copy();q[0]+=.02;q[10]-=.03
        v,e,j,t,active=run.project(q,self.q,[1,1],self.anchor,q-.1,q+.1,self.geometry,True)
        self.assertTrue(active);self.assertLess(np.linalg.norm(e),t[0,0]);self.assertTrue(np.all(np.diff(t[:,0])<=0))

    def test_bounds_and_head_unchanged(self):
        q=self.q.copy();q[2]+=.2
        lo=q-.0001;hi=q+.0001
        v,*_=run.project(q,self.q,[1,1],self.anchor,lo,hi,self.geometry,True)
        self.assertTrue(np.all(v>=lo) and np.all(v<=hi));np.testing.assert_array_equal(v[5:9],q[5:9])

    def test_rotation_units(self):
        r=np.array([[np.cos(.1),-np.sin(.1),0],[np.sin(.1),np.cos(.1),0],[0,0,1]])
        np.testing.assert_allclose(run.rotation_log(r,np.eye(3)),[0,0,.1],atol=1e-15,rtol=0)
        np.testing.assert_array_equal(run.rotation_log(r,r),np.zeros(3))

    def test_unused_head_infeasible_box(self):
        q=self.q.copy();q[2]+=.02;lo=q-.1;hi=q+.1
        lo[5:9]=q[5:9]+1;hi[5:9]=q[5:9]-1
        v,*_=run.project(q,self.q,[1,1],self.anchor,lo,hi,self.geometry,True)
        np.testing.assert_array_equal(v[5:9],q[5:9])

    def test_invalid_inputs(self):
        for contacts in ([np.nan,1],[-1,1],[2,1]):
            with self.assertRaises(ValueError):run.project(self.q,self.q,contacts,self.anchor,self.q-.1,self.q+.1,self.geometry,True)
        with self.assertRaises(ValueError):run.project(self.q,self.q,[1,1],self.anchor,self.q+1,self.q-1,self.geometry,True)


if __name__=='__main__':unittest.main(verbosity=2)
