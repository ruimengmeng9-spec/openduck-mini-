"""Regression of independent read-only reconstruction; no dynamic replay."""
import inspect
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from diagnostics import audit_getup_contact_projection_terminal_r184 as audit

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(audit.run.local.prior.program.SCENE))
        cls.geometry=audit.IndependentGeometry(cls.model)
        cls.old=audit.run.RelativeFootGeometry(cls.model)

    def test_limits_exact(self):
        rng=np.random.default_rng(184)
        for _ in range(100):
            q=rng.normal(size=14);prev=rng.normal(size=14);lo=-np.ones(14);hi=np.ones(14)
            np.testing.assert_array_equal(audit.limits(q,prev,lo,hi),audit.run.planner.apply_limits(q,prev,lo,hi))

    def test_geometry_anchor_residual_and_jacobian(self):
        rng=np.random.default_rng(184)
        for _ in range(5):
            q=self.geometry.home+rng.uniform(-.05,.05,14);nom=self.geometry.home.copy()
            a=self.geometry.anchor(q,nom)
            np.testing.assert_array_equal(a,self.old.anchor(q,nom))
            q=q+rng.uniform(-.01,.01,14)
            np.testing.assert_array_equal(self.geometry.residual(q,nom,a),self.old.residual(q,nom,a))
            np.testing.assert_array_equal(self.geometry.jacobian(q,nom,a),self.old.jacobian(q,nom,a))

    def test_solver_all_saved_outputs_are_independent(self):
        q=self.geometry.home.copy();nom=q.copy();a=self.geometry.anchor(q,nom)
        q[[1,2,10,11]]+=[.025,-.04,-.025,.04]
        lo=q-.1;hi=q+.1
        for c,enabled in [(np.ones(2),True),(np.ones(2),False),(np.array([0,1]),True)]:
            new=audit.reconstruct(q,nom,c,a,lo,hi,self.geometry,enabled)
            old=audit.run.project(q,nom,c,a,lo,hi,self.old,enabled)
            for x,y in zip(new,old):np.testing.assert_array_equal(x,y)
        self.assertNotIn('contact_projection_active',inspect.signature(audit.reconstruct).parameters)
        self.assertNotIn('projection_trace',inspect.signature(audit.reconstruct).parameters)

    def test_nominal_529_actual_planned_zero(self):
        with np.load(audit.run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:q=z['applied'][:529].copy()
        a=self.geometry.anchor(q[0],q[0])
        for target in q:
            np.testing.assert_array_equal(self.geometry.residual(target,target,a),np.zeros(6))
            result=audit.reconstruct(target,target,np.ones(2),a,target-.1,target+.1,self.geometry,True)
            self.assertIs(result[0],target)
            np.testing.assert_array_equal(result[1],np.zeros(6))

    def test_private_root_and_head_invariance(self):
        q=self.geometry.home.copy();nom=q.copy();a=self.geometry.anchor(q,nom)
        baseline=self.geometry.residual(q,nom,a)
        q[5:9]+=.5
        np.testing.assert_array_equal(self.geometry.residual(q,nom,a),baseline)
        np.testing.assert_array_equal(self.geometry.data.qpos[:7],[0,0,0,1,0,0,0])
        np.testing.assert_array_equal(self.geometry.data.qpos[self.geometry.qadr[5:9]],self.geometry.home[5:9])

    def test_no_dynamics_used(self):
        names=('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_geomDistance')
        mocks=[patch.object(mujoco,name,side_effect=AssertionError('Forbidden '+name)) for name in names]
        for mock in mocks:mock.start()
        try:
            q=self.geometry.home.copy();a=self.geometry.anchor(q,q)
            self.geometry.residual(q,q,a);self.geometry.jacobian(q,q,a)
        finally:
            for mock in reversed(mocks):mock.stop()

    def test_budget_and_nonzero_audit_smoke(self):
        self.assertEqual(len(audit.jobs()),62);self.assertEqual(len(audit.smoke_jobs()),6)
        self.assertEqual([audit.read(p/'result.json')['case_seed'] for p in audit.smoke_jobs()][3:],[773005,769006,773009])
        self.assertEqual(audit.ITERATIONS,3);self.assertEqual(audit.STEPS,(1.,.5,.25,.125,.0625))
        self.assertEqual(audit.digest(audit.run.__file__),audit.STORED_MAIN_SHA)

    def test_white_fields_and_scope(self):
        self.assertNotIn('qpos',audit.FIELDS);self.assertNotIn('qvel',audit.FIELDS)
        text=inspect.getsource(audit.scalar)
        self.assertNotIn("z.files",text);self.assertNotIn("HistoryReferenceEpisode(",text)
        self.assertNotIn("row['success']",inspect.getsource(audit.reconstruct))

    def test_bounds_and_head(self):
        q=self.geometry.home.copy();nom=q.copy();a=self.geometry.anchor(q,nom)
        q[[1,10]]+=[.05,-.05]
        result=audit.reconstruct(q,nom,np.ones(2),a,q-.01,q+.01,self.geometry,True)[0]
        np.testing.assert_array_equal(result[5:9],q[5:9])
        self.assertTrue(np.all(result[self.geometry.legs]>=q[self.geometry.legs]-.01))
        self.assertTrue(np.all(result[self.geometry.legs]<=q[self.geometry.legs]+.01))

if __name__=='__main__':unittest.main()

