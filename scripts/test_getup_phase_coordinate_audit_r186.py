"""Independent scalar/limits regression, no dynamic replay."""
import inspect
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from diagnostics import audit_getup_phase_coordinate_terminal_r186 as audit


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(audit.run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:cls.nom=z['observations'].copy()
        with np.load(audit.run.local.prior.program.REFERENCE,allow_pickle=False) as z:cls.ref=z['targets'].copy()
        cls.tangent=np.gradient(cls.nom[:529,6:20].astype(float),axis=0)

    def test_independent_phase_random_and_endpoints(self):
        rng=np.random.default_rng(186)
        for k in (0,1,2,527,528,529,2278):
            for enabled in (False,True):
                for _ in range(20):
                    values=[rng.normal(size=14).astype(np.float32) for _ in range(4)]
                    v=rng.normal(size=14);r=rng.normal(size=(2279,14))
                    a=audit.phase(*values,v,r,k,enabled);b=audit.run.local_phase(*values,v,r,k,enabled)
                    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_actual_nominal_scalar_zero(self):
        for k in range(529):
            extra,s,d=audit.phase(self.nom[k,6:20],self.nom[k,6:20],self.nom[0,6:20],self.nom[0,6:20],self.tangent[k],self.ref,k,True)
            np.testing.assert_array_equal(extra,np.zeros(14));self.assertEqual(s,0.);np.testing.assert_array_equal(d,np.zeros(14))

    def test_causal_initial_zero(self):
        x=self.nom[0,6:20]+np.float32(.031)
        extra,s,d=audit.phase(x,self.nom[0,6:20],x,self.nom[0,6:20],self.tangent[0],self.ref,0,True)
        np.testing.assert_array_equal(extra,np.zeros(14));self.assertEqual(s,0.);np.testing.assert_array_equal(d,np.zeros(14))

    def test_merge_limits_exact_and_identity(self):
        rng=np.random.default_rng(186)
        for _ in range(200):
            base=rng.normal(size=14);fixed=base+rng.uniform(-.18,.18,14);extra=rng.normal(size=14)
            lo=-np.ones(14)*2;hi=-lo;prev=rng.normal(size=14)
            a=audit.merge(fixed,base,extra,lo,hi);b=audit.run.local.merge_target(fixed,base,extra,np.arange(14),lo,hi)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(audit.limits(a,prev,lo,hi),audit.run.planner.apply_limits(a,prev,lo,hi))
            self.assertIs(audit.merge(fixed,base,np.zeros(14),lo,hi),fixed)

    def test_no_saved_decision_inputs_or_root(self):
        params=inspect.signature(audit.phase).parameters
        self.assertEqual(tuple(params),('current','nominal','initial','initial_nominal','tangent','reference','control','enabled'))
        for name in ('qpos','qvel','phase_offset_controls','phase_reference_shift_rad'):self.assertNotIn(name,params)
        self.assertNotIn('qpos',audit.FIELDS);self.assertNotIn('qvel',audit.FIELDS)
        scalar=inspect.getsource(audit.scalar)
        self.assertNotIn('z.files',scalar);self.assertNotIn('HistoryReferenceEpisode(',scalar)
        self.assertNotIn("data['phase_offset_controls']",scalar)
        self.assertNotIn("data['adjusted_double_pre_slew_target_rad'][k]",scalar)

    def test_no_environment_data_or_dynamics(self):
        names=('MjData','mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_kinematics','mj_geomDistance')
        mocks=[patch.object(mujoco,name,side_effect=AssertionError(name)) for name in names]
        for mock in mocks:mock.start()
        try:
            audit.phase(self.nom[1,6:20],self.nom[1,6:20],self.nom[0,6:20],self.nom[0,6:20],self.tangent[1],self.ref,1,True)
        finally:
            for mock in reversed(mocks):mock.stop()
        self.assertNotIn('mujoco.MjData(',inspect.getsource(audit))

    def test_budget_nonzero_smoke_and_source(self):
        self.assertEqual(len(audit.jobs()),62);self.assertEqual(len(audit.smoke_jobs()),6)
        rows=[audit.read(p/'result.json') for p in audit.smoke_jobs()]
        self.assertEqual([r['case_seed'] for r in rows][3:],[769002,773005,769006])
        self.assertTrue(all(r['maximum_phase_reference_shift_rad']>0 for r in rows[3:]))
        self.assertFalse(rows[-1]['valid']);self.assertEqual(rows[-1]['controls'],45)
        self.assertEqual(audit.digest(audit.run.__file__),audit.STORED_MAIN_SHA)

    def test_disabled_home_zero_and_invalid(self):
        x=self.nom[1,6:20]+np.float32(.01)
        for k,e in ((1,False),(529,True),(2278,True)):
            extra,s,d=audit.phase(x,self.nom[1,6:20],self.nom[0,6:20],self.nom[0,6:20],self.tangent[1],self.ref,k,e)
            np.testing.assert_array_equal(extra,np.zeros(14));self.assertEqual(s,0.);np.testing.assert_array_equal(d,np.zeros(14))
        with self.assertRaises(ValueError):audit.phase(x[:13],x,x,x,np.zeros(14),self.ref,1,True)
        with self.assertRaises(ValueError):audit.phase(x,x,x,x,np.full(14,np.nan),self.ref,1,True)
        with self.assertRaises(ValueError):audit.phase(x,x,x,x,np.zeros(14),self.ref,-1,True)


if __name__=='__main__':unittest.main()
