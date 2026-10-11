"""Synthetic numerical checks only: no simulator, episode, old data or physics."""
import copy
import unittest
from pathlib import Path
import numpy as np
import total_target_trpo_kernel_r201 as k

OUTPUT = None
EVIDENCE = {}


def independent_jacobian(weights, x):
    h0=np.tanh(x@weights['actor_w0']+weights['actor_b0'])
    h1=np.tanh(h0@weights['actor_w1']+weights['actor_b1'])
    rows=[]
    names=sorted(n for n in weights if n.startswith('actor_'))
    for s in range(len(x)):
        for j in range(10):
            d2=np.eye(10)[j]
            d1=(d2@weights['actor_w2'].T)*(1-h1[s]**2)
            d0=(d1@weights['actor_w1'].T)*(1-h0[s]**2)
            blocks=dict(actor_b0=d0,actor_b1=d1,actor_b2=d2,
                        actor_w0=np.outer(x[s],d0),actor_w1=np.outer(h0[s],d1),
                        actor_w2=np.outer(h1[s],d2))
            rows.append(np.concatenate([blocks[n].reshape(-1) for n in names]))
    return np.asarray(rows).reshape(len(x),10,-1)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng=np.random.default_rng(30177)
        cls.x=cls.rng.normal(size=(8,101))
        cls.z=1e-4*cls.rng.normal(size=(8,10))
        cls.adv=cls.rng.normal(size=8)
        cls.w=k.initial_weights()
        cls.w_nonzero=copy.deepcopy(cls.w)
        cls.w_nonzero['actor_w2']=cls.rng.normal(scale=1e-4,size=(16,10))
        cls.w_nonzero['actor_b2']=cls.rng.normal(scale=1e-4,size=10)
        cls.learn=k.Learner(cls.w_nonzero)

    def test_01_causal_layout(self):
        a=np.arange(50)/100.;b=-a
        np.testing.assert_array_equal(k.features(a,b,264),np.r_[np.tanh(a),np.tanh(b),.5])

    def test_02_original_clock(self):
        self.assertEqual(k.features(np.zeros(50),np.zeros(50),2279)[-1],1.)
        for bad in (-1,2280,1.2):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),bad)

    def test_03_no_extra_sensor_or_nonfinite(self):
        with self.assertRaises(ValueError):k.features(np.zeros(51),np.zeros(50),0)
        with self.assertRaises(ValueError):k.features(np.full(50,np.nan),np.zeros(50),0)

    def test_04_fresh_initialization(self):
        another=k.initial_weights()
        for n,v in self.w.items():np.testing.assert_array_equal(v,another[n])
        np.testing.assert_array_equal(k.network(self.w,self.x,'actor'),np.zeros((8,10)))
        self.assertGreater(np.linalg.norm(self.w['actor_w0']),0)
        self.assertGreater(np.linalg.norm(self.w['value_w2']),0)
        self.assertEqual(set(self.w),{f'{n}_{b}{i}' for n in ('actor','value') for b in ('w','b') for i in range(3)})

    def test_05_latent_density_formula(self):
        mu=k.network(self.w_nonzero,self.x,'actor')
        ref=np.sum(-.5*((self.z-mu)/1e-4)**2-np.log(1e-4)-.5*np.log(2*np.pi),axis=1)
        np.testing.assert_allclose(k.log_probability(self.z,mu),ref,atol=1e-12,rtol=1e-14)

    def test_06_exact_gaussian_kl(self):
        a=np.zeros((2,10));b=np.ones((2,10))*1e-6
        self.assertAlmostEqual(k.exact_mean_kl(a,b),.0005,places=15)
        self.assertEqual(k.exact_mean_kl(a,a),0)
        self.assertEqual(k.exact_mean_kl(a,b),k.exact_mean_kl(b,a))

    def test_07_tanh_jacobian_ratio_cancels(self):
        mu=k.network(self.w_nonzero,self.x,'actor');new=mu+1e-6
        jac=np.sum(np.log(1-np.tanh(self.z)**2),axis=1)
        np.testing.assert_allclose((k.log_probability(self.z,new)-jac)-
                    (k.log_probability(self.z,mu)-jac),
                    k.log_probability(self.z,new)-k.log_probability(self.z,mu),atol=2e-14,rtol=0)

    def test_08_replace_not_add_head_preserved(self):
        fixed=np.arange(14)/100.;ref=fixed-.02;legs=np.arange(10);head=np.arange(10,14)
        out=k.replace_legs(fixed,ref,np.zeros(10),legs,-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[legs],ref[legs])
        np.testing.assert_array_equal(out[head],fixed[head])
        np.testing.assert_array_equal(fixed,np.arange(14)/100.)

    def test_09_inactive_original_object(self):
        fixed=np.zeros(14)
        self.assertIs(k.replace_legs(fixed,None,None,None,None,None,False),fixed)

    def test_10_joint_and_reference_bounds(self):
        ref=np.zeros(14);fixed=np.zeros(14);legs=np.arange(10)
        out=k.replace_legs(fixed,ref,np.ones(10)*100,legs,-np.ones(14)*.1,np.ones(14)*.1,True)
        np.testing.assert_array_equal(out[:10],np.ones(10)*.1)
        self.assertLessEqual(np.abs(out-ref).max(),.18)

    def test_11_complete_home_tail_and_invalid_terminal(self):
        r=np.array([1.,-2.,3.,4.]);ref=np.array([sum(r[j:]*k.GAMMA**np.arange(len(r)-j)) for j in range(len(r))])
        np.testing.assert_allclose(k.complete_returns(r),ref,atol=2e-15,rtol=0)
        self.assertEqual(k.complete_returns([-3.])[0],-3.)

    def test_12_independent_backprop_gradient(self):
        l=self.learn;mu=k.network(self.w_nonzero,self.x,'actor')
        jac=independent_jacobian(self.w_nonzero,self.x)
        expected=np.einsum('nip,ni,n->p',jac,(self.z-mu)/k.SIGMA**2,self.adv)/len(self.x)
        jp=l.jp;oldlog=k.log_probability(jp.asarray(self.z),l.flat_mean(l.theta,jp.asarray(self.x)),jp)
        actual=np.asarray(l.policy_gradient(l.theta,jp.asarray(self.x),jp.asarray(self.z),oldlog,jp.asarray(self.adv)))
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=3e-12)
        EVIDENCE['independent_gradient_max_error']=float(np.abs(actual-expected).max())

    def test_13_independent_fisher_product(self):
        l=self.learn;jac=independent_jacobian(self.w_nonzero,self.x)
        v=np.sin(np.arange(jac.shape[-1]));v/=np.linalg.norm(v)
        expected=np.einsum('nip,ni->p',jac,np.einsum('nip,p->ni',jac,v))/(len(self.x)*k.SIGMA**2)
        jp=l.jp;x=jp.asarray(self.x);mu=l.flat_mean(l.theta,x)
        actual=np.asarray(l.fisher(l.theta,jp.asarray(v),x,mu))
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=3e-11)
        EVIDENCE['independent_fisher_max_error']=float(np.abs(actual-expected).max())

    def test_14_directional_finite_difference(self):
        l=self.learn;jp=l.jp;x=jp.asarray(self.x);z=jp.asarray(self.z);a=jp.asarray(self.adv)
        oldlog=k.log_probability(z,l.flat_mean(l.theta,x),jp)
        v=np.cos(np.arange(len(l.theta)));v/=np.linalg.norm(v);eps=1e-8
        finite=(float(l.surrogate(l.theta+eps*v,x,z,oldlog,a))-
                float(l.surrogate(l.theta-eps*v,x,z,oldlog,a)))/(2*eps)
        analytic=float(l.policy_gradient(l.theta,x,z,oldlog,a)@v)
        np.testing.assert_allclose(finite,analytic,rtol=1e-4,atol=1e-5)
        EVIDENCE['directional_finite_difference_error']=abs(finite-analytic)

    def test_15_cg_against_dense_solve(self):
        a=np.diag([1.,2.,3.,4.,5.])+.01*np.ones((5,5));g=np.arange(1.,6.)
        d,trace=k.conjugate_gradient(lambda v:a@v,g)
        np.testing.assert_allclose(d,np.linalg.solve(a,g),atol=1e-13,rtol=0)
        self.assertLessEqual(len(trace),10)
        for t in trace:
            np.testing.assert_array_equal(t['product'],a@t['search'])
            self.assertEqual(t['curvature'],float(t['search']@t['product']))

    def test_16_backtrack_first_exact_kl_pass(self):
        theta=np.zeros(1);step=np.array([.01]);g=np.ones(1)
        chosen,e=k.choose_backtrack(theta,step,g,lambda t:(float(t[0]),float(100*t[0]**2)))
        self.assertEqual(e['accepted_index'],2);self.assertEqual(len(e['trials']),3)
        np.testing.assert_array_equal(chosen,step*.25)

    def test_17_no_acceptable_trial_and_nonfinite(self):
        theta=np.zeros(1);step=np.ones(1);g=np.ones(1)
        def evaluation(t):
            return (0.,0.) if t[0]==0 else (np.inf,k.MAX_KL+1)
        chosen,e=k.choose_backtrack(theta,step,g,evaluation)
        self.assertIsNone(e['accepted_index']);self.assertEqual(len(e['trials']),10)
        np.testing.assert_array_equal(chosen,theta)

    def test_18_zero_gradient_no_actor_change(self):
        l=k.Learner(self.w);before=l.numpy_weights()
        e=l.actor_update(self.x,self.z,np.zeros(8))
        self.assertEqual(len(e['cg_trace']),0);self.assertEqual(len(e['line_search']['trials']),0)
        self.assertIsNone(e['line_search']['accepted_index'])
        for n in before:np.testing.assert_array_equal(before[n],l.numpy_weights()[n])

    def test_19_real_synthetic_step_trace_and_determinism(self):
        a=k.Learner(self.w);b=k.Learner(self.w)
        ea=a.actor_update(self.x,self.z,self.adv);eb=b.actor_update(self.x,self.z,self.adv)
        for n in a.numpy_weights():np.testing.assert_array_equal(a.numpy_weights()[n],b.numpy_weights()[n])
        self.assertEqual(a.actor_updates,1);self.assertEqual(a.value_updates,0)
        self.assertLessEqual(len(ea['cg_trace']),10);self.assertLessEqual(len(ea['line_search']['trials']),10)
        if ea['line_search']['accepted_index'] is not None:
            trial=ea['line_search']['trials'][-1]
            self.assertLessEqual(trial['mean_kl'],k.MAX_KL);self.assertGreater(trial['actual_gain'],0)
        if OUTPUT is not None:k.Learner.save_actor_evidence(OUTPUT/'synthetic_actor_trace.npz',ea)
        EVIDENCE['synthetic_actor']={key:ea['line_search'][key] for key in
                                ('baseline_surrogate','baseline_mean_kl','predicted_gain','accepted_index')}
        EVIDENCE['synthetic_actor']['cg_iterations']=len(ea['cg_trace'])

    def test_20_value_only_64_indices_rng_and_msgpack(self):
        a=k.Learner(self.w);rng=np.random.default_rng(30188);ref=np.random.default_rng(30188)
        before=a.numpy_weights();ret=np.arange(8)/10
        u=a.value_update(self.x,ret,rng)
        expected=np.stack([ref.integers(0,8,size=512) for _ in range(64)])
        np.testing.assert_array_equal(u['indices'],expected)
        self.assertEqual(rng.bit_generator.state,ref.bit_generator.state)
        self.assertEqual(a.actor_updates,0);self.assertEqual(a.value_updates,64)
        for n in before:
            if n.startswith('actor_'):np.testing.assert_array_equal(before[n],a.numpy_weights()[n])
        self.assertTrue(all(np.isfinite(u['losses'])))
        state=dict(actor=a.actor,value=a.value,state=a.state,actor_updates=0,value_updates=64)
        restored=a.serialization.from_bytes(state,a.state_bytes())
        self.assertEqual(a.serialization.to_bytes(restored),a.state_bytes())
        if OUTPUT is not None:
            a.checkpoint(OUTPUT/'synthetic_value_checkpoint')
            np.savez_compressed(OUTPUT/'synthetic_value_updates.npz',indices=u['indices'],
                            losses=u['losses'],norms=u['norms'],
                            **{'last_gradient_'+n:v for n,v in u['last_gradient'].items()})
        EVIDENCE['synthetic_value']={'updates':64,'first_loss':float(u['losses'][0]),
                                     'last_loss':float(u['losses'][-1]),'rng':rng.bit_generator.state}

    def test_21_fixed_budget_constants(self):
        self.assertEqual((k.SEED,k.CG_CAP,k.BACKTRACK_CAP,k.VALUE_UPDATES),(301,10,10,64))
        self.assertEqual((k.SIGMA,k.MAX_KL,k.DAMPING,k.VALUE_LR),(1e-4,.001,.01,1e-4))

    def test_22_empty_batch_rejected(self):
        with self.assertRaises(ValueError):self.learn.actor_update(np.empty((0,101)),np.empty((0,10)),np.empty(0))
        with self.assertRaises(ValueError):self.learn.value_update(np.empty((0,101)),np.empty(0),np.random.default_rng(1))
