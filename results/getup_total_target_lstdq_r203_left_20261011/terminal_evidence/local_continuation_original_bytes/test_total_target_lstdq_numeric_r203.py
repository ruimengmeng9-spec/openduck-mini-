"""Independent synthetic LSTD-Q algebra checks, no robot data or simulation."""
from pathlib import Path
import unittest
import numpy as np
import total_target_lstdq_kernel_r203 as k
OUTPUT=None
EVIDENCE={}

def reference_system(theta,x,a,nx,r,b):
    n,d=x.shape;m=a.shape[1];p=d*(m+1)
    cross=theta[d:].reshape(d,m)
    na=np.minimum(1.,np.maximum(-1.,nx@cross))
    na[b==0]=0.
    A=np.zeros((p,p));v=np.zeros(p)
    for i in range(n):
        psi=np.r_[x[i],np.outer(x[i],a[i]).ravel()]
        npsi=np.r_[nx[i],np.outer(nx[i],na[i]).ravel()]
        c=-.5*sum(a[i]**2);nc=-.5*sum(na[i]**2)
        A+=np.outer(psi,psi-k.GAMMA*b[i]*npsi)
        v+=psi*(r[i]+k.GAMMA*b[i]*nc-c)
    return A/n+k.RIDGE*np.eye(p),v/n,na

def independent_elimination(A,b):
    a=np.c_[A.copy(),b.copy()]
    for j in range(len(b)):
        pivot=j+np.argmax(np.abs(a[j:,j]))
        a[[j,pivot]]=a[[pivot,j]]
        a[j]/=a[j,j]
        for i in range(len(b)):
            if i!=j:a[i]-=a[i,j]*a[j]
    return a[:,-1]

class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng=np.random.default_rng(30377)
        cls.x=cls.rng.normal(size=(31,3));cls.a=cls.rng.uniform(-1,1,size=(31,2))
        cls.nx=cls.rng.normal(size=(31,3));cls.b=np.ones(31);cls.b[::4]=0;cls.nx[::4]=0
        cls.r=cls.rng.normal(size=31);cls.t=cls.rng.normal(scale=.05,size=9)
    def test_01_causal_features(self):
        a=np.arange(50)/100
        np.testing.assert_array_equal(k.features(a,-a,264),np.r_[np.tanh(a),np.tanh(-a),.5,1.])
    def test_02_clock_errors(self):
        for c in (-1,2280,1.1,True):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),c)
        self.assertEqual(k.features(np.zeros(50),np.zeros(50),2279)[-2],1)
    def test_03_finite_shape(self):
        for a in (np.zeros(51),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(a,np.zeros(50),0)
    def test_04_fresh_zero(self):
        w=k.initial_weights();self.assertEqual(w['theta'].shape,(1122,))
        np.testing.assert_array_equal(w['theta'],np.zeros(1122))
        np.testing.assert_array_equal(k.greedy(w['theta'],np.ones(102)),np.zeros(10))
    def test_05_basis_order(self):
        x=np.array([[1.,2.]]);a=np.array([[.1,-.2]])
        np.testing.assert_array_equal(k.basis(x,a),[[1.,2.,.1,-.2,.2,-.4]])
    def test_06_Q_independent(self):
        expected=[]
        for x,a in zip(self.x,self.a):
            expected.append(x@self.t[:3]+sum((x@self.t[3:].reshape(3,2))*a)-.5*sum(a*a))
        np.testing.assert_allclose(k.q_value(self.t,self.x,self.a),expected,atol=2e-16,rtol=1e-14)
    def test_07_greedy_box_optimum(self):
        from itertools import product
        grid=np.array(list(product(np.linspace(-1,1,17),repeat=2)))
        for x in self.x[:8]:
            g=k.greedy(self.t,x,2);q=k.q_value(self.t,x,g)
            self.assertGreaterEqual(q+1e-14,float(k.q_value(self.t,np.tile(x,(len(grid),1)),grid).max()))
    def test_08_saturated_greedy(self):
        t=np.r_[np.zeros(2),np.array([[3.,-4.],[0.,0.]]).ravel()]
        np.testing.assert_array_equal(k.greedy(t,[1.,0.],2),[1.,-1.])
    def test_09_sample_exact_rng(self):
        rng=np.random.default_rng(303);ref=np.random.default_rng(303)
        mean,noise,proposal,request=k.sample(np.zeros(1122),np.ones(102),rng)
        np.testing.assert_array_equal(noise,ref.normal(size=10))
        np.testing.assert_array_equal(proposal,1e-4*noise)
        np.testing.assert_array_equal(request,np.clip(proposal,-1,1))
        self.assertEqual(rng.bit_generator.state,ref.bit_generator.state)
    def test_10_deterministic_no_rng(self):
        rng=np.random.default_rng(303);before=str(rng.bit_generator.state)
        k.greedy(np.zeros(1122),np.ones(102))
        self.assertEqual(str(rng.bit_generator.state),before)
    def test_11_sample_boundary_atoms_not_density(self):
        theta=np.zeros(1122);theta[102:112]=100.
        mean,noise,proposal,request=k.sample(theta,np.r_[1.,np.zeros(101)],np.random.default_rng(9))
        np.testing.assert_array_equal(mean,np.ones(10))
        self.assertTrue(np.any(proposal>1));self.assertTrue(np.any(request==1))
    def test_12_replace_not_add(self):
        original=np.arange(14)/100;ref=original-.02
        out=k.replace_legs(original,ref,np.zeros(10),np.arange(10),-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],ref[:10]);np.testing.assert_array_equal(out[10:],original[10:])
    def test_13_inactive_identity(self):
        obj=np.zeros(14)
        self.assertIs(k.replace_legs(obj,None,None,None,None,None,False),obj)
    def test_14_joint_and_request_bounds(self):
        out=k.replace_legs(np.zeros(14),np.zeros(14),np.ones(10),np.arange(10),-np.ones(14)*.1,np.ones(14)*.1,True)
        np.testing.assert_array_equal(out[:10],np.ones(10)*.1)
        with self.assertRaises(ValueError):k.basis([[1.]],[[1.01]])
        with self.assertRaises(ValueError):k.replace_legs(np.zeros(14),np.zeros(14),np.ones(10)*1.01,np.arange(10),-np.ones(14),np.ones(14),True)
    def test_15_full_returns(self):
        r=np.array([1.,-2.,3.,4.])
        expected=[sum(r[i:]*k.GAMMA**np.arange(len(r)-i)) for i in range(4)]
        np.testing.assert_allclose(k.complete_returns(r),expected,atol=2e-15,rtol=0)
    def test_16_home_backup_full_tail(self):
        n=532;r=np.arange(n)/1000;obs=np.zeros((n,50));a=np.zeros((n,10))
        e=k.episode_transitions(obs,r,a,(np.arange(n)>0)&(np.arange(n)<529))
        self.assertEqual(len(e['phi']),528)
        self.assertEqual(e['bootstrap'][-1],0);self.assertTrue(np.all(e['next_phi'][-1]==0))
        expected=sum(r[528:]*k.GAMMA**np.arange(n-528))
        self.assertAlmostEqual(e['reward_backup'][-1],expected,places=14)
        np.testing.assert_array_equal(e['reward_backup'][:-1],r[1:528])
    def test_17_invalid_terminal_no_cross_reset(self):
        n=304;r=np.ones(n);obs=np.zeros((n,50));a=np.zeros((n,10))
        e=k.episode_transitions(obs,r,a,np.arange(n)>0)
        self.assertEqual(len(e['phi']),303);self.assertEqual(e['bootstrap'][-1],0)
        self.assertEqual(e['reward_backup'][-1],1);self.assertEqual(e['control'][-1],303)
    def test_18_independent_nonsymmetric_equation(self):
        A,b,na=k.lstd_system(self.t,self.x,self.a,self.nx,self.r,self.b)
        RA,Rb,Rna=reference_system(self.t,self.x,self.a,self.nx,self.r,self.b)
        np.testing.assert_allclose(A,RA,atol=4e-16,rtol=2e-14)
        np.testing.assert_allclose(b,Rb,atol=2e-16,rtol=2e-14)
        np.testing.assert_array_equal(na,Rna)
        self.assertGreater(np.max(np.abs(A-A.T)),1e-3)
        EVIDENCE['independent_system_max_error']=float(np.max(np.abs(A-RA)))
    def test_19_independent_pivoted_solve(self):
        e=k.solve_lstd(self.t,self.x,self.a,self.nx,self.r,self.b)
        A,b,_=reference_system(self.t,self.x,self.a,self.nx,self.r,self.b)
        expected=independent_elimination(A,b)
        np.testing.assert_allclose(e['theta'],expected,atol=2e-14,rtol=2e-13)
        self.assertLessEqual(e['relative_residual'],1e-10)
        EVIDENCE['independent_solve_max_error']=float(np.max(np.abs(e['theta']-expected)))
        if OUTPUT is not None:np.savez_compressed(OUTPUT/'synthetic_small_system.npz',**{n:e[n] for n in ('theta','matrix','rhs','residual','next_action','action_before','action_after')})
    def test_20_production_dimension_two_solve_budget(self):
        rng=np.random.default_rng(30388);x=rng.normal(size=(19,102));a=rng.uniform(-1e-4,1e-4,size=(19,10))
        nx=rng.normal(size=(19,102));boot=np.ones(19);boot[-1]=0;nx[-1]=0;r=rng.normal(size=19)
        l=k.Learner()
        for j in range(2):
            e=l.update(x,a,nx,r,boot);self.assertEqual(e['solve_count'],j+1)
            self.assertEqual(e['matrix'].shape,(1122,1122))
            if OUTPUT is not None:
                np.savez_compressed(OUTPUT/f'synthetic_full_solve_{j+1}.npz',phi=x,action=a,next_phi=nx,reward=r,bootstrap=boot,**{n:e[n] for n in ('theta','matrix','rhs','residual','next_action','action_before','action_after')})
        with self.assertRaises(RuntimeError):l.update(x,a,nx,r,boot)
        if OUTPUT is not None:l.checkpoint(OUTPUT/'synthetic_checkpoint')
        EVIDENCE['synthetic_two_solves']=dict(count=l.solves,relative_residual=e['relative_residual'],not_robot_learning=True)
    def test_21_checkpoint_state_exact(self):
        l=k.Learner();l.theta=np.arange(1122)/1000;l.solves=2
        before=l.state_bytes();v=np.frombuffer(before[20:],dtype='<f8')
        np.testing.assert_array_equal(v,l.theta)
        self.assertEqual(len(before),20+1122*8)
        self.assertEqual(k.Learner({'theta':l.theta}).theta.tobytes(),l.theta.tobytes())
    def test_22_no_leakage_or_bad_terminal(self):
        with self.assertRaises(ValueError):k.lstd_system(self.t,self.x,self.a,self.nx,self.r,self.b+.1)
        nx=self.nx.copy();nx[0]=1
        with self.assertRaises(ValueError):k.lstd_system(self.t,self.x,self.a,nx,self.r,self.b)
        with self.assertRaises(ValueError):k.episode_transitions(np.zeros((4,50)),np.ones(4),np.zeros((4,10)),np.ones(4,dtype=bool))
    def test_23_offset_not_reward_penalty(self):
        A,b,na=k.lstd_system(self.t,self.x,self.a,self.nx,self.r,self.b)
        psi=k.basis(self.x,self.a);npsi=k.basis(self.nx,na)
        expected=psi.T@(self.r+self.b*k.GAMMA*k.offset(na)-k.offset(self.a))/len(self.x)
        np.testing.assert_allclose(b,expected,atol=2e-16,rtol=2e-14)
        self.assertEqual(k.CURVATURE,1.)
    def test_24_fixed_structure_and_no_old_learning(self):
        import ast
        tree=ast.parse(Path(k.__file__).read_text())
        imported=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n and (n.startswith('diagnostics') or n.startswith('jax') or n.startswith('optax') or n.startswith('mujoco')) for n in imported))
        self.assertEqual((k.SEED,k.PARAM_DIM,k.MAX_SOLVES,k.RIDGE,k.SIGMA),(303,1122,2,.001,.0001))
        EVIDENCE['no_simulation_or_old_learning_data']=True

