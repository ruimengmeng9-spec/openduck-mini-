"""Independent momentum/planning regression; no dynamic episodes."""
import inspect
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from diagnostics import audit_getup_centroidal_momentum_terminal_r189b as audit

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(audit.run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:cls.nom=z['observations'].copy()
        cls.model=audit.load_model();cls.g=audit.IndependentGeometry(cls.model)
        cls.original=audit.run.kernel.MomentumGeometry(cls.model)

    def test_independent_geometry_random(self):
        rng=np.random.default_rng(189)
        for _ in range(20):
            x=self.nom[0,:34].copy();x[:3]=rng.normal(size=3);x[6:34]+=rng.normal(0,.03,28).astype(np.float32)
            for a,b in zip(self.g.measure(x),self.original.measure(x)):np.testing.assert_array_equal(a,b)

    def test_native_gyro_units_and_weld(self):
        self.assertNotEqual(self.g.gyrobody,self.g.trunk)
        self.assertEqual(self.model.body_weldid[self.g.gyrobody],self.model.body_weldid[self.g.trunk])
        x=np.zeros(34,dtype=np.float32);x[:3]=[.3,-.2,.1];x[20:34]=np.arange(14)*np.float32(.001)
        h,A,M,v=self.g.measure(x)
        np.testing.assert_array_equal(v[self.g.vadr],x[20:34].astype(float)/.05)
        np.testing.assert_array_equal(v[self.g.rootlinear],np.zeros(3))
        np.testing.assert_allclose(M[:,self.g.rootlinear],np.zeros((3,3)),atol=1e-15,rtol=0.)
        d=self.g.data;jr=np.zeros((3,self.model.nv));mujoco.mj_jacSite(self.model,d,None,jr,self.g.site)
        np.testing.assert_allclose(jr@v,d.site_xmat[self.g.site].reshape(3,3)@x[:3].astype(float),atol=1e-15,rtol=3e-13)

    def test_root_translation_matrix_same_actual_c_implementation(self):
        x=np.zeros(34,dtype=np.float32)
        M=self.g.measure(x)[2];original=self.original.measure(x)[2]
        np.testing.assert_array_equal(M[:,self.g.rootlinear],original[:,self.g.rootlinear])
        self.assertGreater(float(np.abs(M[:,self.g.rootlinear]).max()),0.)
        self.assertLessEqual(float(np.abs(M[:,self.g.rootlinear]).max()),1e-15)

    def test_independent_requests_random(self):
        rng=np.random.default_rng(189)
        for k in (0,1,528,529,2278):
            for enabled in (False,True):
                xs=[(self.nom[0,:34]+rng.normal(0,.02,34)).astype(np.float32) for _ in range(4)]
                a=audit.momentum(*xs,self.g,k,enabled)
                b=audit.run.kernel.momentum_request(*xs,self.original,k,enabled)
                for x,y in zip(a[:4],b):np.testing.assert_array_equal(x,y)

    def test_actual_nominal_and_initial_zero(self):
        for k in range(529):
            a=audit.momentum(self.nom[k,:34],self.nom[k,:34],self.nom[0,:34],self.nom[0,:34],self.g,k,True)
            for item in (a[0],a[1],a[3]):np.testing.assert_array_equal(item,np.zeros_like(item))
        x=self.nom[0,:34]+np.float32(.01)
        a=audit.momentum(x,self.nom[0,:34],x,self.nom[0,:34],self.g,0,True)
        for item in (a[0],a[1],a[3]):np.testing.assert_array_equal(item,np.zeros_like(item))

    def test_unused_up_head_no_addition(self):
        x=self.nom[35,:34].copy();a=self.g.measure(x);x[3:6]=[3,4,5]
        for u,v in zip(a,self.g.measure(x)):np.testing.assert_array_equal(u,v)
        x[6+self.g.head[0]]+=np.float32(.12);x[20+self.g.head[0]]+=np.float32(.02)
        self.assertFalse(np.array_equal(a[0],self.g.measure(x)[0]))
        r=audit.momentum(x,self.nom[35,:34],self.nom[0,:34],self.nom[0,:34],self.g,35,True)[0]
        np.testing.assert_array_equal(r[self.g.head],np.zeros(4))

    def test_merge_limits_exact_identity_head(self):
        rng=np.random.default_rng(189)
        for _ in range(200):
            base=rng.normal(size=14);fixed=base+rng.uniform(-.18,.18,14);raw=rng.normal(size=14);raw[self.g.head]=0
            a=audit.merge(fixed,base,raw,self.g.legs,self.g.lower,self.g.upper)
            b=audit.run.merge_momentum(fixed,base,raw,self.original,self.g.lower,self.g.upper)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[self.g.head],fixed[self.g.head])
            prev=rng.normal(size=14)
            np.testing.assert_array_equal(audit.limits(a,prev,self.g.lower,self.g.upper),audit.run.planner.apply_limits(a,prev,self.g.lower,self.g.upper))
            self.assertIs(audit.merge(fixed,base,np.zeros(14),self.g.legs,self.g.lower,self.g.upper),fixed)

    def test_no_saved_decision_inputs(self):
        self.assertEqual(tuple(inspect.signature(audit.momentum).parameters),('current','nominal','initial','initial_nominal','geometry','control','enabled'))
        self.assertNotIn('qpos',audit.FIELDS);self.assertNotIn('qvel',audit.FIELDS)
        source=inspect.getsource(audit.scalar)
        for item in ("data['momentum_request_rad'][k]","data['momentum_allocation_kg_m2'][k]","data['adjusted_double_pre_slew_target_rad'][k]",'HistoryReferenceEpisode('):self.assertNotIn(item,source)
        self.assertNotIn('run.kernel.momentum_request',inspect.getsource(audit))
        self.assertNotIn('run.kernel.MomentumGeometry',inspect.getsource(audit))

    def test_no_private_dynamics_and_full_model_unchanged(self):
        before=audit.model_digest(self.model)
        names=('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision')
        mocks=[patch.object(mujoco,name,side_effect=AssertionError(name)) for name in names]
        for m in mocks:m.start()
        try:audit.momentum(self.nom[10,:34],self.nom[9,:34],self.nom[0,:34],self.nom[0,:34],self.g,10,True)
        finally:
            for m in reversed(mocks):m.stop()
        self.assertEqual(before,audit.model_digest(self.model));self.assertEqual(before,audit.COMPILED_SHA)

    def test_budget_nonzero_failed_invalid_smoke_source(self):
        self.assertEqual(len(audit.jobs()),62);self.assertEqual(len(audit.smoke_jobs()),6)
        rows=[audit.read(p/'result.json') for p in audit.smoke_jobs()]
        self.assertEqual([r['case_seed'] for r in rows][3:],[769002,773004,769001])
        self.assertTrue(rows[3]['success']);self.assertFalse(rows[4]['success']);self.assertTrue(rows[4]['valid'])
        self.assertFalse(rows[5]['valid']);self.assertEqual(rows[5]['controls'],44)
        self.assertTrue(all(r['maximum_raw_momentum_request_rad']>0 for r in rows[3:]))
        self.assertEqual(audit.digest(audit.run.__file__),audit.STORED_MAIN_SHA)

    def test_disabled_home_zero_and_invalid(self):
        x=self.nom[1,:34]+np.float32(.01)
        for k,e in ((1,False),(529,True),(2278,True)):
            for a in audit.momentum(x,self.nom[1,:34],self.nom[0,:34],self.nom[0,:34],self.g,k,e):np.testing.assert_array_equal(a,np.zeros_like(a))
        for xs,k in (([x[:33],x,x,x],1),([np.full(34,np.nan),x,x,x],1),([x,x,x,x],-1)):
            with self.assertRaises(ValueError):audit.momentum(*xs,self.g,k,True)

if __name__=='__main__':unittest.main()
