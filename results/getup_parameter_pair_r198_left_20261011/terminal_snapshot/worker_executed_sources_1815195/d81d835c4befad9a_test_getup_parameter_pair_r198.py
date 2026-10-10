"""Offline full causal-policy/interface/parameter-learning regressions; no episodes."""
import ast
import os
from pathlib import Path
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_parameter_pair_r198 as run
from diagnostics import parameter_pair_kernel_r198 as k

class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights();self.x=k.features(np.arange(50)/50,np.zeros(50),17)
        self.p,self.eps,self.before,self.after=k.proposal_set(self.w)

    def test_01_fresh_hidden_initial_output_zero(self):
        self.assertTrue(np.any(self.w['actor_w0']))
        np.testing.assert_array_equal(k.network(self.w,self.x),np.zeros(10))
        self.assertEqual(sum(v.size for v in self.w.values()),1802)
        self.assertEqual(set(self.w),set(k.BLOCKS))

    def test_02_features_layout(self):
        np.testing.assert_array_equal(self.x[:50],np.tanh(np.arange(50)/50))
        np.testing.assert_array_equal(self.x[50:100],np.zeros(50))
        self.assertEqual(self.x.shape,(101,));self.assertEqual(self.x[100],17/528)

    def test_03_only_causal_initial_history(self):
        y=k.features(np.arange(50)/50,np.ones(50),17)
        np.testing.assert_array_equal(y[:50],self.x[:50])
        self.assertTrue(np.any(y[50:100]!=self.x[50:100]))

    def test_04_reject_metadata_extended_sensor_and_clock(self):
        for sensor in (np.zeros(49),np.zeros(55),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(sensor,np.zeros(50),1)
        for control in (-1,2280,1.5):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),control)

    def test_05_progress_boundaries(self):
        for control,expected in ((0,0.),(528,1.),(529,1.),(2279,1.)):
            self.assertEqual(k.features(np.zeros(50),np.zeros(50),control)[-1],expected)

    def test_06_pack_unpack_all_blocks_exact(self):
        weights=k.unflatten(k.flatten(self.w))
        for key in k.BLOCKS:np.testing.assert_array_equal(weights[key],self.w[key])
        weights['actor_w0'][0,0]+=1
        self.assertNotEqual(weights['actor_w0'][0,0],self.w['actor_w0'][0,0])

    def test_07_reject_wrong_parameter_keys_shape_nonfinite(self):
        for theta in (np.zeros(1801),np.full(1802,np.inf)):
            with self.assertRaises(ValueError):k.unflatten(theta)
        wrong={**self.w,'case_seed':np.array(1.)}
        with self.assertRaises(ValueError):k.flatten(wrong)
        wrong={**self.w,'actor_w0':np.zeros((101,15))}
        with self.assertRaises(ValueError):k.flatten(wrong)

    def test_08_direction_rng_independently_reconstructed(self):
        rng=np.random.default_rng(29801)
        self.assertEqual(rng.bit_generator.state,self.before)
        np.testing.assert_array_equal(rng.normal(size=(2,1802)),self.eps)
        self.assertEqual(rng.bit_generator.state,self.after)

    def test_09_exact_antithetic_full_parameters(self):
        theta=np.r_[self.w['actor_w0'].ravel(),self.w['actor_b0'],self.w['actor_w1'].ravel(),self.w['actor_b1']]
        for i in range(4):
            expected=theta+(1 if i%2==0 else -1)*1e-5*self.eps[i//2]
            np.testing.assert_array_equal(k.flatten(self.p[i]),expected)

    def test_10_every_block_perturbed_not_readout_only(self):
        for weights in self.p:
            for key in k.BLOCKS:self.assertGreater(np.linalg.norm(weights[key]-self.w[key]),0.)
        self.assertGreater(np.linalg.norm(k.network(self.p[0],self.x)),0.)

    def test_11_fixed_policy_no_action_rng_or_recurrence(self):
        a=k.network(self.p[0],self.x)
        for _ in range(10):np.testing.assert_array_equal(k.network(self.p[0],self.x),a)
        self.assertFalse(hasattr(k,'sample_action'));self.assertFalse(hasattr(k,'Learner'))

    def test_12_nonlinear_hidden_current_input_effect(self):
        changed={n:v.copy() for n,v in self.p[0].items()};changed['actor_w0']+=.2
        self.assertTrue(np.any(k.network(changed,self.x)!=k.network(self.p[0],self.x)))
        self.assertTrue(np.any(k.network(self.p[0],self.x+.1)!=k.network(self.p[0],self.x)))

    def test_13_numpy_batch_same_scalar_arithmetic_tolerance(self):
        x=np.stack([self.x,self.x+.1])
        np.testing.assert_allclose(k.network(self.p[0],x),np.stack([k.network(self.p[0],y) for y in x]),atol=1e-14,rtol=0)
        self.assertLessEqual(np.max(np.abs(k.network(self.p[0],x))),1.)

    def test_14_original_all25_returns_including_negative(self):
        returns=np.arange(25,dtype=float);returns[0]=-1000.
        self.assertEqual(k.complete_score(returns),float(sum(returns)/25))
        for r in (np.ones(24),np.full(25,np.nan)):
            with self.assertRaises(ValueError):k.complete_score(r)

    def test_15_update_independent_explicit_formula(self):
        scores=np.array([[11.,10.],[5.,7.]])
        weights,out=k.update(self.w,self.eps,scores)
        coefficients=(scores[:,0]-scores[:,1])/(2e-5)
        g=(coefficients[0]*self.eps[0]+coefficients[1]*self.eps[1])/2
        np.testing.assert_array_equal(g,out['gradient'])
        norm=float(np.sqrt(np.sum(g*g)))
        self.assertEqual(out['gradient_norm'],float(np.linalg.norm(g)))
        np.testing.assert_allclose(k.flatten(weights),k.flatten(self.w)+1e-4*g/max(1.,norm),atol=1e-15,rtol=0)
        for key in k.BLOCKS:self.assertGreater(out['changes'][key],0.)

    def test_16_gradient_norm_bound_and_small_gradient(self):
        weights,out=k.update(self.w,self.eps,np.array([[11.,10.],[5.,7.]]))
        self.assertLessEqual(np.linalg.norm(k.flatten(weights)-k.flatten(self.w)),1e-4+1e-14)
        _,small=k.update(self.w,np.ones((2,1802))*1e-10,np.array([[1.,0.],[0.,0.]]))
        np.testing.assert_array_equal(small['gradient'],small['bounded_gradient'])
        self.assertLess(small['gradient_norm'],1.)

    def test_17_equal_scores_exact_zero_update_no_implicit_optimizer(self):
        weights,out=k.update(self.w,self.eps,np.ones((2,2)))
        for key in k.BLOCKS:np.testing.assert_array_equal(weights[key],self.w[key])
        np.testing.assert_array_equal(out['gradient'],np.zeros(1802));self.assertEqual(out['gradient_norm'],0.)

    def test_18_update_shape_and_nonfinite_rejected(self):
        for eps,scores in ((np.ones((3,1802)),np.ones((2,2))),(self.eps,np.ones((2,3))),(self.eps,np.full((2,2),np.nan))):
            with self.assertRaises(ValueError):k.update(self.w,eps,scores)

    def test_19_inactive_original_object(self):
        fixed=np.arange(14)/100.
        self.assertIs(k.replace_legs(fixed,None,None,None,None,None,False),fixed)

    def test_20_replace_not_add_no_extra_tanh_head_joint(self):
        fixed=np.full(14,.12);legs=np.arange(10)
        out=k.replace_legs(fixed,np.zeros(14),np.ones(10),legs,-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],np.full(10,.18));np.testing.assert_array_equal(out[10:],fixed[10:])
        out=k.replace_legs(fixed,np.zeros(14),np.zeros(10),legs,-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],np.zeros(10))
        out=k.replace_legs(fixed,np.zeros(14),np.ones(10),legs,np.full(14,-.1),np.full(14,.1),True)
        np.testing.assert_array_equal(out[:10],np.full(10,.1))
        with self.assertRaises(ValueError):k.replace_legs(fixed,np.zeros(14),np.ones(10)*1.01,legs,-np.ones(14),np.ones(14),True)

    def test_21_actual_static_model_actuator_mapping_and_limits(self):
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        self.assertEqual(run.utility.model_digest(model),'53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3')
        mapping=run.utility.kernel.Mapping(model)
        joints=model.actuator_trnid[:,0];lo,hi=model.jnt_range[joints].T
        reference=(lo+hi)/2;fixed=reference.copy();fixed[mapping.head]+=.01
        out=k.replace_legs(fixed,reference,np.ones(10),mapping.legs,lo,hi,True)
        np.testing.assert_array_equal(out[mapping.head],fixed[mapping.head])
        np.testing.assert_array_equal(out[mapping.legs],np.clip(reference[mapping.legs]+.18,lo[mapping.legs],hi[mapping.legs]))
        previous=reference
        planned=run.planner.apply_limits(out,previous,lo,hi)
        self.assertLessEqual(np.max(np.abs(planned-previous)),5.24*.02+1e-12)

    def test_22_smoke_null_standard_invalid_block(self):
        reports={name:dict(physical_failures=0,nominal_success=True if None in cases else None,rows=[dict(controls=2279,complete_R157_parity={'a':True}) for c in cases]) for name,mode,cases,parity in run.SMOKE_SPEC}
        self.assertTrue(run.smoke_gate(reports))
        reports['initial_probe']['physical_failures']=1;self.assertFalse(run.smoke_gate(reports))

    def test_23_budget_selection_no_success_union(self):
        self.assertEqual(6+6+25+100+25+25,187)
        base=dict(successes=18,rows=[dict(case_seed=i,success=i<18) for i in range(24)])
        report=dict(successes=19,nominal_success=True,physical_failures=0,rows=[dict(case_seed=i,success=i<19) for i in range(24)])
        self.assertTrue(run.eligible(report,base))
        report['rows'][0]['success']=False;self.assertFalse(run.eligible(report,base))
        report['rows'][0]['success']=True;report['physical_failures']=1;self.assertFalse(run.eligible(report,base))

    def test_24_kernel_no_simulator_and_original_worker_boundary(self):
        source=Path(k.__file__).read_text();tree=ast.parse(source)
        self.assertNotIn('mujoco',source);self.assertNotIn('case_seed',source)
        self.assertEqual([n.names[0].name for n in tree.body if isinstance(n,ast.Import)],['numpy'])
        runner=Path(run.__file__).read_text()
        self.assertIn("active=mode!='baseline' and 0<k<529",runner)
        self.assertIn("if not active:assert adjusted is fixed",runner)
        self.assertIn("local.NOMINAL.shape==(2279,55)",runner)
        self.assertIn("np.testing.assert_array_equal(applied,wanted)",runner)
        self.assertIn("np.testing.assert_array_equal(sim.prev,wanted)",runner)
        self.assertNotIn('utility.init_worker(',runner)
        self.assertIn("OUTPUT,SMOKE,LAUNCH,PREFLIGHT",runner)

if __name__=='__main__':
    where=os.environ.get('OPEN_DUCK_R198_TEST_CAPTURE')
    if where:
        destination=Path(where);manifest=run.capture_sources(destination)
        run.local.write_json(destination.parent/'source_manifest.json',manifest)
    unittest.main()

