"""New R194 independent sensor/selection/planning tests; no dynamic episodes."""
import inspect,os
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import unittest
import mujoco
import numpy as np
from diagnostics import audit_getup_shared_horizon_teacher_terminal_r194 as audit

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(audit.run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:cls.nom=z['observations'].copy()
        cls.model=audit.load_model();cls.g=audit.IndependentMapping(cls.model)
        cls.c=audit.proposals()[0][1]
        destination=os.environ.get('OPEN_DUCK_R194_TEST_CAPTURE')
        if destination:audit.write(Path(destination).parent/'test_source_manifest.json',audit.capture_sources(Path(destination)))

    def test_independent_causal_arithmetic_equal(self):
        rng=np.random.default_rng(294)
        for k in (0,1,51,528,529,2278):
            for c in audit.proposals()[0]:
                xs=[(self.nom[0,:34]+rng.normal(0,.02,34)).astype(np.float32) for _ in range(4)]
                a=audit.teacher(*xs,c,self.g,k)
                b=audit.run.kernel.teacher_request(*xs,c,self.g,k)
                for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_actual_nominal_and_control_zero(self):
        for k in range(529):
            raw,delta,wave,a=audit.teacher(self.nom[k,:34],self.nom[k,:34],self.nom[0,:34],self.nom[0,:34],self.c,self.g,k)
            np.testing.assert_array_equal(raw,np.zeros(14));np.testing.assert_array_equal(delta,np.zeros(34));self.assertEqual(a,0)
        x=self.nom[0,:34]+np.float32(.1)
        for v in audit.teacher(x,self.nom[0,:34],x,self.nom[0,:34],self.c,self.g,0):np.testing.assert_array_equal(v,np.zeros_like(v))

    def test_fixed_harmonics_scale_units(self):
        np.testing.assert_array_equal(audit.BASIS,audit.run.kernel.BASIS)
        np.testing.assert_array_equal(audit.SCALE,audit.run.kernel.SCALE)
        self.assertFalse(audit.BASIS.flags.writeable)
        raw,delta,wave,a=audit.teacher(np.ones(34),np.zeros(34),np.zeros(34),np.zeros(34),np.full((10,4),1e-4),self.g,10)
        np.testing.assert_array_equal(delta,1/audit.SCALE)
        np.testing.assert_array_equal(raw[self.g.legs],a*wave)
        self.assertLessEqual(np.abs(raw).max(),4e-4);np.testing.assert_array_equal(raw[self.g.head],np.zeros(4))

    def test_zero_coefficients_and_home(self):
        for k in (1,100,528):
            raw,delta,wave,a=audit.teacher(np.ones(34),np.zeros(34),np.zeros(34),np.zeros(34),np.zeros((10,4)),self.g,k)
            np.testing.assert_array_equal(raw,np.zeros(14));np.testing.assert_array_equal(wave,np.zeros(10));self.assertGreater(a,0)
        for k in (529,2278):
            for v in audit.teacher(np.ones(34),np.zeros(34),np.zeros(34),np.zeros(34),self.c,self.g,k):np.testing.assert_array_equal(v,np.zeros_like(v))

    def test_merge_limits_head_zero_object(self):
        rng=np.random.default_rng(294)
        for _ in range(200):
            ref=rng.uniform(-.2,.2,14);fixed=ref+rng.uniform(-.18,.18,14);raw=rng.normal(0,1e-5,14);raw[self.g.head]=0
            x=audit.merge(fixed,ref,raw,self.g.legs,self.g.lower,self.g.upper)
            y=audit.run.merge_teacher(fixed,ref,raw,self.g,self.g.lower,self.g.upper)
            np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(x[self.g.head],fixed[self.g.head])
            previous=rng.normal(size=14)
            np.testing.assert_array_equal(audit.limits(x,previous,self.g.lower,self.g.upper),audit.run.planner.apply_limits(x,previous,self.g.lower,self.g.upper))
            self.assertIs(audit.merge(fixed,ref,np.zeros(14),self.g.legs,self.g.lower,self.g.upper),fixed)

    def test_scalar_white_list_and_saved_compare_only(self):
        self.assertEqual(tuple(inspect.signature(audit.teacher).parameters),('current','nominal','initial','initial_nominal','coefficients','mapping','control'))
        self.assertNotIn('qpos',audit.FIELDS);self.assertNotIn('qvel',audit.FIELDS)
        source=inspect.getsource(audit.scalar)
        for key in audit.SIGNALS:
            self.assertNotIn("data['"+key+"'][k]",source)
        self.assertNotIn('run.kernel.teacher_request',inspect.getsource(audit))
        self.assertNotIn('run.kernel.choose_teacher',inspect.getsource(audit))
        self.assertNotIn('run.kernel.retention',inspect.getsource(audit))
        self.assertNotIn('HistoryReferenceEpisode(',source)

    def test_no_MjData_dynamics_or_original_decisions(self):
        before=audit.model_digest(self.model)
        with ExitStack() as stack:
            for name in ('MjData','mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_kinematics','mj_comPos','mj_comVel','mj_rne','mj_geomDistance','mj_contactForce','mj_collision','mj_passive','mj_fwdActuation'):
                stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError(name)))
            stack.enter_context(patch.object(audit.run.kernel,'teacher_request',side_effect=AssertionError('old decision')))
            mapping=audit.IndependentMapping(self.model)
            audit.teacher(self.nom[10,:34],self.nom[9,:34],self.nom[0,:34],self.nom[0,:34],self.c,mapping,10)
        self.assertEqual(before,audit.model_digest(self.model));self.assertEqual(before,audit.COMPILED_SHA)

    def test_proposals_RNG_pairs_and_bound(self):
        p,state=audit.proposals();q,other=audit.run.kernel.make_proposals()
        for x,y in zip(p,q):np.testing.assert_array_equal(x,y)
        self.assertEqual(state,other);self.assertEqual(len(p),5)
        np.testing.assert_array_equal(p[1],-p[2]);np.testing.assert_array_equal(p[3],-p[4])

    def test_common_selection_entire_history(self):
        v=audit.common_selection()
        self.assertEqual(v['selected_program'],0);self.assertEqual(v['closed_programs'],5)
        self.assertFalse(v['original_development_gate'])
        self.assertTrue(all(not e['eligible'] for e in v['retention']))

    def test_retention_no_union_invalid_and_rank(self):
        def report(successes,invalid=0,nominal=True,returns=None):
            rows=[dict(case_seed=i,initial_hash=str(i),success=i in successes,return_sum=float((returns or [0,1,2,3])[i])) for i in range(4)]
            return dict(rows=rows,successes=len(successes),physical_failures=invalid,nominal_success=nominal)
        base=report({0,1});regressed=report({1,2,3});rescue=report({0,1,2});invalid=report({0,1,2,3},1)
        selected,e=audit.choose([base,regressed,rescue,invalid]);self.assertEqual(selected,2);self.assertFalse(e[1]['eligible']);self.assertFalse(e[3]['eligible'])
        self.assertEqual(audit.choose([base,regressed,invalid])[0],0)
        self.assertEqual(audit.choose([base,rescue,report({0,1,2},returns=[1,2,3,4])])[0],2)
        rescue['rows'][0]['initial_hash']='other'
        with self.assertRaises(ValueError):audit.retention(rescue,base)

    def test_full187_and_smoke_nonzero_recover_degrade_invalid(self):
        self.assertEqual(len(audit.jobs()),187);self.assertEqual(len(audit.smoke_jobs()),6)
        self.assertEqual(len({audit.identity(p) for p in audit.jobs()}),187)
        rows=[audit.read(p/'result.json') for p in audit.smoke_jobs()]
        self.assertTrue(rows[3]['success']);self.assertTrue(rows[3]['valid'])
        self.assertFalse(rows[4]['success']);self.assertTrue(rows[4]['valid'])
        self.assertFalse(rows[5]['valid']);self.assertEqual(rows[5]['controls'],45)
        self.assertTrue(all(r['maximum_raw_teacher_request_rad']>0 for r in rows[3:]))

    def test_invalid_inputs_and_source_pins(self):
        self.assertEqual(audit.digest(audit.run.__file__),audit.STORED_MAIN_SHA)
        self.assertEqual(audit.digest(audit.run.kernel.__file__),audit.run.KERNEL_SHA)
        z=np.zeros(34);c=np.zeros((10,4))
        for bad in (z[:33],np.full(34,np.nan)):
            with self.assertRaises(ValueError):audit.teacher(bad,z,z,z,c,self.g,1)
        for bad in (np.zeros((10,5)),np.full((10,4),.001)):
            with self.assertRaises(ValueError):audit.teacher(z,z,z,z,bad,self.g,1)
        for k in (-1,1.2):
            with self.assertRaises(ValueError):audit.teacher(z,z,z,z,c,self.g,k)

if __name__=='__main__':unittest.main(verbosity=2)
