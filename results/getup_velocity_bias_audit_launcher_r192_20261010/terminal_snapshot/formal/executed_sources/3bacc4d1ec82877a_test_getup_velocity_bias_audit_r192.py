"""New independent bias/planning audit regressions; no dynamic episodes."""
import inspect
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import mujoco
import numpy as np
from diagnostics import audit_getup_velocity_bias_terminal_r192 as audit

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(audit.run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:cls.nom=z['observations'].copy()
        cls.model=audit.load_model();cls.g=audit.IndependentGeometry(cls.model)
        cls.original=audit.run.kernel.VelocityBias(cls.model)
        destination=os.environ.get('OPEN_DUCK_R192_TEST_CAPTURE')
        if destination:audit.write(Path(destination).parent/'test_source_manifest.json',audit.capture_sources(Path(destination)))

    def test_independent_measurement_same_native_arithmetic(self):
        rng=np.random.default_rng(292)
        for _ in range(20):
            x=self.nom[0,:34].copy();x[:3]=rng.normal(size=3);x[6:34]+=rng.normal(0,.03,28).astype(np.float32)
            for a,b in zip(self.g.measure(x),self.original.measure(x)):np.testing.assert_array_equal(a,b)

    def test_gyro_weld_native_units(self):
        self.assertNotEqual(self.g.gyrobody,self.g.trunk)
        self.assertEqual(self.model.body_weldid[self.g.gyrobody],self.model.body_weldid[self.g.trunk])
        x=np.zeros(34,np.float32);x[:3]=[.3,-.2,.1];x[20:34]=np.arange(14)*np.float32(.001)
        v=self.g.measure(x)[3]
        np.testing.assert_array_equal(v[self.g.vadr],x[20:34].astype(float)/.05)
        np.testing.assert_array_equal(v[self.g.rootlinear],np.zeros(3))
        jr=np.zeros((3,self.model.nv));mujoco.mj_jacSite(self.model,self.g.data,None,jr,self.g.site)
        np.testing.assert_allclose(jr@v,self.g.data.site_xmat[self.g.site].reshape(3,3)@x[:3].astype(float),atol=1e-15,rtol=3e-13)

    def test_requests_same_native_arithmetic(self):
        rng=np.random.default_rng(292)
        for k in (0,1,528,529,2278):
            for enabled in (False,True):
                xs=[(self.nom[0,:34]+rng.normal(0,.02,34)).astype(np.float32) for _ in range(4)]
                a=audit.bias(*xs,self.g,k,enabled)
                b=audit.run.kernel.velocity_bias_request(*xs,self.original,k,enabled)
                for x,y in zip(a[:3],b):np.testing.assert_array_equal(x,y)

    def test_actual_nominal_and_initial_zero(self):
        for k in range(529):
            a=audit.bias(self.nom[k,:34],self.nom[k,:34],self.nom[0,:34],self.nom[0,:34],self.g,k,True)
            for item in a[:2]:np.testing.assert_array_equal(item,np.zeros_like(item))
        x=self.nom[0,:34]+np.float32(.01)
        for item in audit.bias(x,self.nom[0,:34],x,self.nom[0,:34],self.g,0,True)[:2]:np.testing.assert_array_equal(item,np.zeros_like(item))

    def test_unused_up_head_branch(self):
        x=np.zeros(34,np.float32);x[20+self.g.head[0]]=.1
        a=self.g.measure(x);x[3:6]=[3,4,5]
        for u,v in zip(a,self.g.measure(x)):np.testing.assert_array_equal(u,v)
        np.testing.assert_array_equal(a[0][self.g.legs],np.zeros(10))
        self.assertGreater(float(np.abs(a[0][self.g.head]).max()),0.)
        raw=audit.bias(x,np.zeros(34),np.zeros(34),np.zeros(34),self.g,1,True)[0]
        np.testing.assert_array_equal(raw[self.g.head],np.zeros(4))

    def test_original_servo_positive_sign(self):
        np.testing.assert_array_equal(self.g.kp,np.full(14,13.37))
        x=self.nom[31,:34].copy();x[0]+=.2
        raw,e,_=audit.bias(x,self.nom[31,:34],self.nom[0,:34],self.nom[0,:34],self.g,31,True)[:3]
        np.testing.assert_array_equal(raw[self.g.legs],e[self.g.legs]/self.g.kp[self.g.legs])

    def test_merge_limits_identity_head(self):
        rng=np.random.default_rng(292)
        for _ in range(200):
            base=rng.normal(size=14);fixed=base+rng.uniform(-.18,.18,14);raw=rng.normal(size=14);raw[self.g.head]=0
            a=audit.merge(fixed,base,raw,self.g.legs,self.g.lower,self.g.upper)
            b=audit.run.merge_bias(fixed,base,raw,self.original,self.g.lower,self.g.upper)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[self.g.head],fixed[self.g.head])
            prev=rng.normal(size=14)
            np.testing.assert_array_equal(audit.limits(a,prev,self.g.lower,self.g.upper),audit.run.planner.apply_limits(a,prev,self.g.lower,self.g.upper))
            self.assertIs(audit.merge(fixed,base,np.zeros(14),self.g.legs,self.g.lower,self.g.upper),fixed)

    def test_no_saved_decision_inputs(self):
        self.assertEqual(tuple(inspect.signature(audit.bias).parameters),('current','nominal','initial','initial_nominal','geometry','control','enabled'))
        self.assertNotIn('qpos',audit.FIELDS);self.assertNotIn('qvel',audit.FIELDS)
        source=inspect.getsource(audit.scalar)
        for item in ("data['velocity_bias_request_rad'][k]","data['velocity_bias_four_nm'][k]","data['adjusted_double_pre_slew_target_rad'][k]",'HistoryReferenceEpisode('):self.assertNotIn(item,source)
        self.assertNotIn('run.kernel.velocity_bias_request',inspect.getsource(audit))
        self.assertNotIn('run.kernel.VelocityBias',inspect.getsource(audit))

    def test_no_private_dynamics_and_model_unchanged(self):
        before=audit.model_digest(self.model)
        with ExitStack() as stack:
            for name in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision','mj_rnePostConstraint','mj_passive','mj_fwdActuation'):
                stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError(name)))
            stack.enter_context(patch.object(audit.run.kernel,'velocity_bias_request',side_effect=AssertionError('old decision')))
            stack.enter_context(patch.object(audit.run.kernel.VelocityBias,'measure',side_effect=AssertionError('old measure')))
            audit.bias(self.nom[10,:34],self.nom[9,:34],self.nom[0,:34],self.nom[0,:34],self.g,10,True)
        self.assertEqual(before,audit.model_digest(self.model));self.assertEqual(before,audit.COMPILED_SHA)
        self.assertEqual(self.g.data.time,0.)

    def test_budget_smoke_rescue_degrade_failed_source(self):
        self.assertEqual(len(audit.jobs()),62);self.assertEqual(len(audit.smoke_jobs()),6)
        rows=[audit.read(p/'result.json') for p in audit.smoke_jobs()]
        self.assertEqual([r['case_seed'] for r in rows][3:],[773004,769000,769002])
        self.assertTrue(rows[3]['success']);self.assertFalse(rows[4]['success']);self.assertFalse(rows[5]['success'])
        self.assertTrue(all(r['valid'] for r in rows));self.assertTrue(all(r['maximum_raw_velocity_bias_request_rad']>0 for r in rows[3:]))
        self.assertEqual(audit.digest(audit.run.__file__),audit.STORED_MAIN_SHA)
        self.assertEqual(audit.digest(audit.run.kernel.__file__),audit.run.KERNEL_SHA)

    def test_disabled_home_zero_invalid(self):
        x=self.nom[1,:34]+np.float32(.01)
        for k,e in ((1,False),(529,True),(2278,True)):
            for a in audit.bias(x,self.nom[1,:34],self.nom[0,:34],self.nom[0,:34],self.g,k,e):np.testing.assert_array_equal(a,np.zeros_like(a))
        for xs,k,e in (([x[:33],x,x,x],1,True),([np.full(34,np.nan),x,x,x],1,True),([x,x,x,x],-1,True),([x,x,x,x],True,True),([x,x,x,x],1,2)):
            with self.assertRaises(ValueError):audit.bias(*xs,self.g,k,e)

if __name__=='__main__':unittest.main(verbosity=2)
