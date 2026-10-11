"""Synthetic-only independent V-MPO structure checks; no old data or simulation."""
import copy
from pathlib import Path
import unittest
import numpy as np
import total_target_vmpo_kernel_r202 as k
OUTPUT=None
EVIDENCE={}

def independent_jacobian(w,x):
    a=np.tanh(x@w['actor_w0']+w['actor_b0'])
    b=np.tanh(a@w['actor_w1']+w['actor_b1'])
    rows=[]
    for i in range(len(x)):
        sub=[]
        for j in range(10):
            d2=np.eye(10)[j]
            d1=(d2@w['actor_w2'].T)*(1-b[i]**2)
            d0=(d1@w['actor_w1'].T)*(1-a[i]**2)
            blocks=dict(actor_b0=d0,actor_b1=d1,actor_b2=d2,
                actor_w0=np.outer(x[i],d0),actor_w1=np.outer(a[i],d1),actor_w2=np.outer(b[i],d2))
            sub.append(np.concatenate([blocks[n].ravel() for n in sorted(blocks)]))
        rows.append(sub)
    return np.asarray(rows)

class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng=np.random.default_rng(30277)
        cls.x=cls.rng.normal(size=(8,101));cls.z=1e-4*cls.rng.normal(size=(8,10))
        cls.a=cls.rng.normal(size=8);cls.w=k.initial_weights()
        cls.nonzero=copy.deepcopy(cls.w)
        cls.nonzero['actor_w2']=cls.rng.normal(scale=1e-4,size=(16,10))
        cls.nonzero['actor_b2']=cls.rng.normal(scale=1e-4,size=10)
    def test_01_features(self):
        a=np.arange(50)/100
        np.testing.assert_array_equal(k.features(a,-a,264),np.r_[np.tanh(a),np.tanh(-a),.5])
    def test_02_clocks(self):
        self.assertEqual(k.features(np.zeros(50),np.zeros(50),2279)[-1],1)
        for t in (-1,2280,1.1):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),t)
    def test_03_finite_and_shape(self):
        for a in (np.zeros(51),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(a,np.zeros(50),0)
    def test_04_fresh_seed(self):
        for n,v in self.w.items():np.testing.assert_array_equal(v,k.initial_weights()[n])
        np.testing.assert_array_equal(k.network(self.w,self.x,'actor'),np.zeros((8,10)))
        self.assertGreater(np.linalg.norm(self.w['actor_w0']),0)
        self.assertGreater(np.linalg.norm(self.w['value_w2']),0)
    def test_05_density(self):
        mu=k.network(self.nonzero,self.x,'actor')
        ref=np.sum(-.5*((self.z-mu)/1e-4)**2-np.log(1e-4)-.5*np.log(2*np.pi),axis=1)
        np.testing.assert_allclose(k.log_probability(self.z,mu),ref,atol=1e-12,rtol=1e-14)
    def test_06_kl(self):
        a=np.zeros((2,10));b=a+1e-6
        self.assertAlmostEqual(k.exact_mean_kl(a,b),.0005,places=15)
        self.assertEqual(k.exact_mean_kl(a,a),0)
    def test_07_replace_not_add(self):
        fixed=np.arange(14)/100;ref=fixed-.02
        out=k.replace_legs(fixed,ref,np.zeros(10),np.arange(10),-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],ref[:10]);np.testing.assert_array_equal(out[10:],fixed[10:])
    def test_08_inactive_object(self):
        fixed=np.zeros(14);self.assertIs(k.replace_legs(fixed,None,None,None,None,None,False),fixed)
    def test_09_bounds(self):
        out=k.replace_legs(np.zeros(14),np.zeros(14),np.ones(10)*100,np.arange(10),-np.ones(14)*.1,np.ones(14)*.1,True)
        np.testing.assert_array_equal(out[:10],np.ones(10)*.1)
    def test_10_MC_tail(self):
        r=np.array([1.,-2.,3.,4.])
        ref=np.array([sum(r[i:]*k.GAMMA**np.arange(len(r)-i)) for i in range(len(r))])
        np.testing.assert_allclose(k.complete_returns(r),ref,atol=2e-15,rtol=0)
        self.assertEqual(k.complete_returns([-3])[0],-3)
    def test_11_top_half_stable(self):
        e=k.e_step([2.,2.,2.,1.,0.])
        np.testing.assert_array_equal(e['indices'],[0,1,2])
        self.assertEqual(len(e['trace']),80);self.assertAlmostEqual(e['weights'].sum(),1)
    def test_12_flat_advantages(self):
        e=k.e_step(np.zeros(8))
        np.testing.assert_array_equal(e['indices'],np.arange(4))
        np.testing.assert_array_equal(e['weights'][:4],np.full(4,.25))
        self.assertEqual(e['top_kl'],0)
    def test_13_E_constraint_independent(self):
        e=k.e_step(self.a);top=self.a[e['indices']]
        u=np.exp((top-top.max())/e['eta']);q=u/u.sum()
        np.testing.assert_array_equal(q,e['weights'][e['indices']])
        kl=sum(q*np.log(q*len(q)))
        self.assertLessEqual(kl,.1+1e-12);self.assertAlmostEqual(kl,.1,places=10)
        self.assertTrue(np.all(e['weights'][np.setdiff1d(np.arange(8),e['indices'])]==0))
        EVIDENCE['E_eta']=e['eta'];EVIDENCE['E_KL']=e['top_kl']
    def test_14_shift_invariance(self):
        a=k.e_step(self.a);b=k.e_step(self.a+3)
        np.testing.assert_allclose(a['weights'],b['weights'],atol=1e-15,rtol=0)
    def test_15_dual_derivative(self):
        e=k.e_step(self.a);v=self.a[e['indices']];eta=e['eta'];eps=eta*1e-5
        def dual(t):
            w=(v-v.max())/t
            return t*(.1+v.max()/t+np.log(np.exp(w).mean()))
        deriv=(dual(eta+eps)-dual(eta-eps))/(2*eps)
        self.assertAlmostEqual(deriv,.1-e['top_kl'],places=8)
    def test_16_independent_actor_gradient(self):
        l=k.Learner(self.nonzero);q=k.e_step(self.a)['weights']
        mu=k.network(self.nonzero,self.x,'actor');old=mu+1e-6
        jac=independent_jacobian(self.nonzero,self.x)
        dm=q[:,None]*(mu-self.z)/k.SIGMA**2 + .7*(mu-old)/(len(self.x)*k.SIGMA**2)
        expected=np.einsum('nip,ni->p',jac,dm)
        (_,aux),grad=l.actor_gradient(l.actor,l.jp.asarray(self.x),l.jp.asarray(self.z),l.jp.asarray(q),l.jp.asarray(old),l.jp.asarray(.7))
        actual=np.concatenate([np.asarray(grad[n]).ravel() for n in sorted(grad)])
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=3e-12)
        EVIDENCE['gradient_max_error']=float(np.abs(actual-expected).max())
    def test_17_two_updates_nonzero_baseline(self):
        l=k.Learner(self.nonzero)
        for _ in range(2):
            e=l.actor_update(self.x,self.z,self.a)
            self.assertEqual(e['gradients'].shape[0],64);self.assertEqual(e['scalars'].shape,(64,8))
            if e['accepted']:
                self.assertLessEqual(e['chosen_mean_kl'],.001);self.assertLess(e['proposal_mle'],e['baseline_mle'])
            else:self.assertEqual(e['chosen_mean_kl'],0)
        self.assertEqual(l.actor_fit_attempts,128);self.assertEqual(l.actor_batches,2)
    def test_18_deterministic_full_trace(self):
        a=k.Learner(self.w);b=k.Learner(self.w)
        ea=a.actor_update(self.x,self.z,self.a);eb=b.actor_update(self.x,self.z,self.a)
        np.testing.assert_array_equal(ea['parameters'],eb['parameters'])
        np.testing.assert_array_equal(ea['gradients'],eb['gradients'])
        self.assertEqual(a.state_bytes(),b.state_bytes())
        if OUTPUT is not None:
            k.Learner.save_actor_evidence(OUTPUT/'synthetic_actor_trace.npz',ea)
            a.checkpoint(OUTPUT/'synthetic_actor_checkpoint')
        EVIDENCE['synthetic_actor']={n:ea[n] for n in ('accepted','proposal_mean_kl','chosen_mean_kl','actor_fit_attempts')}
    def test_19_rejected_proposal_restores(self):
        l=k.Learner(self.w);before=l.state_bytes();old=l.numpy_weights()
        e=l.actor_update(self.x,np.ones((8,10))*100,self.a)
        self.assertFalse(e['accepted'])
        for n,v in old.items():np.testing.assert_array_equal(v,l.numpy_weights()[n])
        self.assertEqual(l.alpha,1)
        self.assertEqual(l.serialization.to_bytes(l.actor_state),l.serialization.to_bytes(k.Learner(self.w).actor_state))
    def test_20_value_indices_and_rng(self):
        l=k.Learner(self.w);rng=np.random.default_rng(30288);ref=np.random.default_rng(30288)
        e=l.value_update(self.x,np.arange(8)/10,rng)
        np.testing.assert_array_equal(e['indices'],np.stack([ref.integers(0,8,size=512) for _ in range(64)]))
        self.assertEqual(rng.bit_generator.state,ref.bit_generator.state)
        self.assertEqual(l.value_updates,64);self.assertEqual(l.actor_fit_attempts,0)
        if OUTPUT is not None:
            np.savez_compressed(OUTPUT/'synthetic_value_updates.npz',indices=e['indices'],losses=e['losses'],norms=e['norms'],gradients=e['gradients'])
            l.checkpoint(OUTPUT/'synthetic_value_checkpoint')
    def test_21_checkpoint_full_roundtrip(self):
        l=k.Learner(self.w)
        template=dict(actor=l.actor,value=l.value,actor_state=l.actor_state,value_state=l.value_state,alpha=l.alpha,actor_fit_attempts=0,actor_batches=0,value_updates=0)
        restored=l.serialization.from_bytes(template,l.state_bytes())
        self.assertEqual(l.serialization.to_bytes(restored),l.state_bytes())
    def test_22_invalid_batches(self):
        l=k.Learner(self.w)
        with self.assertRaises(ValueError):l.actor_update(np.empty((0,101)),np.empty((0,10)),np.empty(0))
        with self.assertRaises(ValueError):k.e_step([np.nan])
        with self.assertRaises(ValueError):l.value_update(np.empty((0,101)),np.empty(0),np.random.default_rng(1))
    def test_23_fixed_budget(self):
        self.assertEqual((k.SEED,k.ACTOR_STEPS,k.VALUE_UPDATES,k.ETA_STEPS),(302,64,64,80))
        self.assertEqual((k.SIGMA,k.MAX_KL,k.E_KL,k.ACTOR_LR,k.VALUE_LR),(1e-4,.001,.1,1e-7,1e-4))
    def test_24_no_old_learning_or_simulator(self):
        import ast
        source=Path(k.__file__).read_text();tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('diagnostics') for n in ast.walk(tree)))
        self.assertNotIn('mujoco',source)
        for term in ('conjugate_gradient','choose_backtrack','policy_gradient','replay'):
            self.assertNotIn(term,source)
