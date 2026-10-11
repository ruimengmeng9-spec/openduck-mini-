"""24 synthetic NAF regressions, no old data or simulator."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import total_target_naf_kernel_r204 as k
OUTPUT=None
EVIDENCE={}
LEARNER=None

def episode(n,seed=1):
    rng=np.random.default_rng(seed)
    obs=rng.normal(size=(n,50));a=np.clip(rng.normal(0,.02,(n,10)),-1,1);r=rng.normal(size=n)
    mask=(np.arange(n)>0)&(np.arange(n)<529)
    return obs,a,r,mask

def independent_heads(p,x,jp):
    z=jp.tanh(x@p['q_w0']+p['q_b0'])
    z=jp.tanh(z@p['q_w1']+p['q_b1'])
    o=z@p['q_w2']+p['q_b2'];raw=o[:,11:]
    lower=jp.zeros((x.shape[0],10,10),dtype=jp.float64);index=0
    for row in range(10):
        for col in range(row+1):
            v=raw[:,index]
            if row==col:v=jp.logaddexp(v,0.)+.001
            lower=lower.at[:,row,col].set(v);index+=1
    return o[:,0],jp.tanh(o[:,1:11]),lower

def independent_objective(p,x,a,y):
    import jax.numpy as jp
    value,mean,L=independent_heads(p,x,jp)
    d=a-mean
    transformed=jp.matmul(jp.swapaxes(L,1,2),d[:,:,None])[:,:,0]
    prediction=value-.5*jp.sum(transformed**2,axis=1)
    return .5*jp.mean((prediction-y)**2)

def audit_trace(learner,replay,trace,start_count,start_state):
    jax,jp,optax=learner.jax,learner.jp,learner.optax
    p={n:jp.asarray(v) for n,v in trace['before'].items()}
    target={n:jp.asarray(v) for n,v in trace['target_before'].items()}
    state=start_state
    gradient=jax.jit(jax.grad(independent_objective))
    max_error=0.;max_target=0.
    for step,ix in enumerate(trace['replay_indices']):
        x,a,r,nx,boot,tail=[jp.asarray(replay[n][ix]) for n in ('features','action','reward','next_features','bootstrap','home_tail')]
        value=independent_heads(target,nx,jp)[0]
        expected=r+k.GAMMA*(boot*value+tail)
        y=jp.asarray(trace['targets'][step])
        max_target=max(max_target,float(jp.max(jp.abs(expected-y))))
        g=gradient(p,x,a,y);stored={n:jp.asarray(v[step]) for n,v in trace['all_gradients'].items()}
        for n in g:max_error=max(max_error,float(jp.max(jp.abs(g[n]-stored[n]))))
        changes,state=learner.optimizer.update(stored,state,p)
        p=optax.apply_updates(p,changes)
        target=jax.tree.map(lambda t,v:(1-k.TAU)*t+k.TAU*v,target,p)
        assert int(trace['statistics'][step,4])==start_count+step+1
    assert max_target<1e-12 and max_error<1e-12,(max_target,max_error)
    for n in p:
        assert np.array_equal(np.asarray(p[n]),trace['after'][n])
        assert np.array_equal(np.asarray(target[n]),trace['target_after'][n])
    data=dict(params=p,target=target,optimizer_state=state,count=start_count+64)
    assert learner.serialization.to_bytes(data)==trace['state_after']
    return dict(independent_gradient_max_absolute_error=max_error,independent_target_max_absolute_error=max_target,
        stored_gradient_Adam_Polyak_and_msgpack_exact=True,independent_gradient_not_cross_implementation_bitwise_claim=True)

def save_trace(name,replay,trace,learner):
    if OUTPUT is None:return
    payload={**{'replay_'+n:v for n,v in replay.items()},
        **{n:trace[n] for n in ('replay_indices','targets','predictions','td_errors','statistics')},
        **{'gradient_'+n:v for n,v in trace['all_gradients'].items()}}
    for group in ('before','after','target_before','target_after'):payload.update({group+'_'+n:v for n,v in trace[group].items()})
    np.savez_compressed(OUTPUT/(name+'_full_trace.npz'),**payload)
    (OUTPUT/(name+'_state_before.msgpack')).write_bytes(trace['state_before'])
    (OUTPUT/(name+'_state_after.msgpack')).write_bytes(trace['state_after'])
    with (OUTPUT/(name+'_rng.json')).open('x') as f:json.dump(dict(before=trace['rng_before'],after=trace['rng_after']),f,indent=2);f.write('\n')
    learner.checkpoint(OUTPUT/name)

class NumericalTests(unittest.TestCase):
    def setUp(self):self.w=k.initial_weights();self.rng=np.random.default_rng(41)
    def test_01_causal_features(self):
        c=np.arange(50)/20;i=-c
        np.testing.assert_array_equal(k.features(c,i,528),np.r_[np.tanh(c),np.tanh(i),1.])
    def test_02_bad_inputs(self):
        for c,i,s in [(np.zeros(49),np.zeros(50),0),(np.ones(50)*np.nan,np.zeros(50),0),(np.zeros(50),np.zeros(50),.5),(np.zeros(50),np.zeros(50),2280)]:
            with self.assertRaises(ValueError):k.features(c,i,s)
    def test_03_fresh_seed(self):
        for n in self.w:np.testing.assert_array_equal(self.w[n],k.initial_weights()[n])
        self.assertEqual(set(self.w),{'q_w0','q_b0','q_w1','q_b1','q_w2','q_b2'})
    def test_04_initial_exact_zero_mean(self):
        _,mu,_,raw=k.heads(self.w,self.rng.normal(size=(17,101)))
        np.testing.assert_array_equal(mu,np.zeros((17,10)));np.testing.assert_array_equal(raw,np.zeros((17,10)))
    def test_05_positive_lower(self):
        _,_,L,_=k.heads(self.w,self.rng.normal(size=(7,101)))
        self.assertTrue(np.all(np.diagonal(L,axis1=1,axis2=2)>0))
        np.testing.assert_allclose(L,np.broadcast_to(np.eye(10),L.shape),atol=2e-16,rtol=0)
    def test_06_state_dependent_positive_curvature(self):
        self.w['q_w2'][:,11:]=self.rng.normal(0,.03,(16,55))
        _,_,L,_=k.heads(self.w,self.rng.normal(size=(7,101)))
        P=L@L.swapaxes(1,2)
        for p in P:self.assertGreater(np.linalg.eigvalsh(p).min(),0.)
        self.assertFalse(np.array_equal(P[0],P[1]))
    def test_07_independent_quadratic(self):
        self.w['q_w2'][:,1:]=self.rng.normal(0,.01,(16,65))
        x=self.rng.normal(size=(9,101));a=self.rng.normal(0,.1,(9,10));v,m,L,_=k.heads(self.w,x)
        expected=np.array([v[i]-.5*(a[i]-m[i])@(L[i]@L[i].T)@(a[i]-m[i]) for i in range(9)])
        np.testing.assert_allclose(k.q_value(self.w,x,a),expected,atol=1e-12,rtol=1e-12)
    def test_08_analytic_box_greedy(self):
        self.w['q_w2'][:,1:11]=self.rng.normal(0,.1,(16,10))
        x=self.rng.normal(size=(12,101));v,m,_,_=k.heads(self.w,x)
        np.testing.assert_array_equal(k.q_value(self.w,x,m),v)
        self.assertTrue(np.all(k.q_value(self.w,x,self.rng.uniform(-1,1,(12,10)))<=v))
    def test_09_bounded_mean(self):
        self.w['q_b2'][1:11]=np.arange(10)*100.-500.
        _,m,_,_=k.heads(self.w,np.zeros(101));self.assertLessEqual(np.abs(m).max(),1.)
    def test_10_exact_sample_rng(self):
        a=np.random.default_rng(51);b=np.random.default_rng(51);z=b.normal(size=10)
        out=k.request(self.w,np.zeros(101),a)
        np.testing.assert_array_equal(out['noise'],z);np.testing.assert_array_equal(out['action'],np.clip(1e-4*z,-1,1))
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
    def test_11_deterministic_no_rng(self):
        state=copy.deepcopy(self.rng.bit_generator.state);out=k.request(self.w,np.zeros(101))
        np.testing.assert_array_equal(out['noise'],np.zeros(10));self.assertEqual(state,self.rng.bit_generator.state)
    def test_12_clipping_atoms(self):
        self.w['q_b2'][1:11]=1000.
        out=k.request(self.w,np.zeros(101),np.random.default_rng(42))
        self.assertTrue(np.any(out['proposal']>1));self.assertTrue(np.any(out['action']==1));self.assertLessEqual(out['action'].max(),1.)
    def test_13_inactive_object_identity(self):
        obj=np.zeros(14);self.assertIs(k.replace_legs(obj,None,None,None,None,None,False),obj)
    def test_14_replace_no_old_leg_sum(self):
        fixed=np.arange(14)/10;ref=np.zeros(14);legs=np.arange(10);a=np.ones(10)
        out=k.replace_legs(fixed,ref,a,legs,-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],np.ones(10)*.18);np.testing.assert_array_equal(out[10:],fixed[10:])
        with self.assertRaises(ValueError):k.replace_legs(fixed,ref,a*2,legs,-np.ones(14),np.ones(14),True)
    def test_15_full_returns(self):
        r=np.array([1.,-2.,3.]);np.testing.assert_array_equal(k.complete_returns(r),np.array([1.+k.GAMMA*(-2.+k.GAMMA*3),-2.+k.GAMMA*3,3.]))
    def test_16_full_home_tail(self):
        data=episode(2279);e=k.episode_replay(*data);self.assertEqual(len(e['reward']),528)
        self.assertEqual(e['control'][-1],528);self.assertEqual(e['bootstrap'][-1],0)
        total=0.
        for r in data[2][529:][::-1]:total=float(r)+k.GAMMA*total
        self.assertEqual(e['home_tail'][-1],total);self.assertEqual(e['home_tail'][:-1].sum(),0)
        np.testing.assert_array_equal(e['features'][0],k.features(data[0][1],data[0][0],1))
    def test_17_original_invalid_terminal(self):
        e=k.episode_replay(*episode(304))
        self.assertEqual(len(e['reward']),303);self.assertTrue(e['terminal'][-1]);self.assertEqual(e['bootstrap'][-1],0);self.assertEqual(e['home_tail'].sum(),0)
    def test_18_no_reset_crossing(self):
        e1=k.episode_replay(*episode(4,2));e2=k.episode_replay(*episode(6,3))
        merged=k.concatenate_replays([e1,e2])
        np.testing.assert_array_equal(merged['episode_offsets'],[0,3,8])
        self.assertEqual(merged['bootstrap'][2],0);self.assertEqual(merged['bootstrap'][7],0)
    def test_19_bad_mask_rejected(self):
        obs,a,r,mask=episode(10);mask[0]=True
        with self.assertRaises(ValueError):k.episode_replay(obs,a,r,mask)
        a[0,0]=np.inf
        with self.assertRaises(ValueError):k.episode_replay(obs,a,r,mask)
    def test_20_bellman_original_reward(self):
        np.testing.assert_array_equal(k.bellman(np.array([1.,2.,3.]),np.array([1.,0.,0.]),np.array([0.,0.,5.]),np.array([7.,8.,9.])),[1+k.GAMMA*7,2,3+k.GAMMA*5])
    def test_21_joint_gradient_finite_difference(self):
        learner=k.Learner(self.w);jp=learner.jp
        x=self.rng.normal(size=(3,101));a=self.rng.normal(0,.1,(3,10));y=np.array([1.,2.,3.])
        (_,aux),g=learner.gradient(learner.params,jp.asarray(x),jp.asarray(a),jp.asarray(y));errors=[]
        for name,index in [('q_w0',(2,3)),('q_b2',(1,)),('q_b2',(11,)),('q_b2',(13,)),('q_b2',(0,))]:
            plus={n:v.copy() for n,v in self.w.items()};minus={n:v.copy() for n,v in self.w.items()}
            plus[name][index]+=1e-6;minus[name][index]-=1e-6
            loss=lambda w:.5*np.mean((k.q_value(w,x,a)-y)**2)
            numerical=(loss(plus)-loss(minus))/2e-6;errors.append(abs(numerical-float(g[name][index])))
        self.assertLess(max(errors),2e-8);EVIDENCE['finite_difference_max_error']=max(errors)
    def test_22_target_stopped(self):
        learner=k.Learner(self.w);jax,jp=learner.jax,learner.jp;x=jp.ones((2,101))
        g=jax.grad(lambda p:jp.sum(jax.lax.stop_gradient(k.heads(p,x,jp)[0])))(learner.target)
        self.assertTrue(all(np.array_equal(np.asarray(v),np.zeros_like(v)) for v in g.values()))
    def test_23_first_joint_Q_update_trace(self):
        global LEARNER
        LEARNER=k.Learner(self.w)
        replay=k.concatenate_replays([k.episode_replay(*episode(8,61)),k.episode_replay(*episode(10,62))])
        state=LEARNER.state;trace=LEARNER.update(replay,np.random.default_rng(64))
        self.assertEqual(LEARNER.count,64);self.assertEqual(trace['separate_actor_updates'],0)
        audit=audit_trace(LEARNER,replay,trace,0,state);save_trace('synthetic_round1',replay,trace,LEARNER)
        self.assertEqual(trace['targets'].shape,(64,512));self.assertTrue(any(x>0 for x in trace['changes'].values()))
        EVIDENCE['synthetic_round1']=dict(audit,updates=64,changes=trace['changes'])
    def test_24_second_fresh_round_and_full_state(self):
        self.assertIsNotNone(LEARNER)
        replay=k.concatenate_replays([k.episode_replay(*episode(9,71)),k.episode_replay(*episode(11,72))])
        state=LEARNER.state;trace=LEARNER.update(replay,np.random.default_rng(74))
        self.assertEqual(LEARNER.count,128)
        audit=audit_trace(LEARNER,replay,trace,64,state);save_trace('synthetic_round2',replay,trace,LEARNER)
        self.assertEqual(trace['all_gradients']['q_w2'].shape,(64,16,66))
        EVIDENCE['synthetic_round2']=dict(audit,updates=64,changes=trace['changes'])

