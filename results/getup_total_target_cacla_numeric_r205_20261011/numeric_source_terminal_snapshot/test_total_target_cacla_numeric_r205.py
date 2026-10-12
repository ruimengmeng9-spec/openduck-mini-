"""24 independent finite numerical checks; synthetic observations only, no dynamics."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import total_target_cacla_kernel_r205 as K
OUTPUT=None
EVIDENCE={}

def independent_forward(w,x,group):
    a=np.tanh(np.dot(x,w[group+'_w1'])+w[group+'_b1'])
    b=np.tanh(np.dot(a,w[group+'_w2'])+w[group+'_b2'])
    raw=np.dot(b,w[group+'_w3'])+w[group+'_b3']
    return (np.tanh(raw) if group=='actor' else raw),raw

def nonzero_weights():
    w=K.initial_weights();r=np.random.default_rng(803)
    w['actor_w3']=r.normal(size=(16,10))*.03
    w['actor_b3']=r.normal(size=10)*.01
    return w

def batch(n=7,all_negative=False):
    r=np.random.default_rng(804)
    x=r.normal(size=(n,101))*.2;nx=r.normal(size=(n,101))*.2
    a=np.clip(r.normal(size=(n,10))*.1,-1,1)
    rewards=r.normal(size=n)*.3
    if all_negative:rewards.fill(-10.)
    return dict(phi=x,nextphi=nx,actions=a,rewards=rewards,bootstrap=(np.arange(n)%2).astype(np.float64),
        home_tail=np.linspace(0.,.1,n))

def independent_grad(w,b,group,y,positive):
    grads={group+'_'+n:np.zeros_like(w[group+'_'+n]) for n in K.NAMES}
    for i,x in enumerate(b['phi']):
        h1=np.tanh(x@w[group+'_w1']+w[group+'_b1'])
        h2=np.tanh(h1@w[group+'_w2']+w[group+'_b2'])
        raw=h2@w[group+'_w3']+w[group+'_b3']
        if group=='actor':
            out=np.tanh(raw);d=(out-b['actions'][i])*float(positive[i])*(1.-out*out)/len(b['phi'])
        else:d=(raw-y[i])/len(b['phi'])
        grads[group+'_w3']+=np.outer(h2,d);grads[group+'_b3']+=d
        d2=(w[group+'_w3']@d)*(1.-h2*h2)
        grads[group+'_w2']+=np.outer(h1,d2);grads[group+'_b2']+=d2
        d1=(w[group+'_w2']@d2)*(1.-h1*h1)
        grads[group+'_w1']+=np.outer(x,d1);grads[group+'_b1']+=d1
    return grads

def frozen_loss(w,b,group,y,mask):
    z,_=independent_forward(w,b['phi'],group)
    if group=='value':return .5*np.mean((z[:,0]-y)**2)
    return .5*np.mean(np.sum((z-b['actions'])**2*mask[:,None],axis=1))

class NumericalTests(unittest.TestCase):
    def assert_arrays(self,a,b):
        self.assertEqual(a.keys(),b.keys())
        for k in a:self.assertTrue(np.array_equal(a[k],b[k]),k)
    def test_01_shapes_and_counts(self):
        w=K.initial_weights();K.validate_weights(w)
        self.assertEqual(sum(v.size for k,v in w.items() if k.startswith('actor')),3962)
        self.assertEqual(sum(v.size for k,v in w.items() if k.startswith('value')),3809)
        self.assertTrue(all(v.dtype==np.float64 for v in w.values()))
        self.assertTrue(np.all(w['actor_w3']==0.) and np.all(w['actor_b3']==0.))
    def test_02_independent_initialization(self):
        r=np.random.default_rng(305);w={}
        for g,last in (('actor',10),('value',1)):
            ds=(101,32,16,last)
            for i in range(1,4):
                w[g+'_w'+str(i)]=(np.zeros((ds[i-1],ds[i])) if g=='actor' and i==3
                    else r.normal(size=(ds[i-1],ds[i]))/np.sqrt(float(ds[i-1])))
                w[g+'_b'+str(i)]=np.zeros(ds[i])
        self.assert_arrays(K.initial_weights(),w)
        EVIDENCE['fresh_seed305_independent_initial_arrays_exact']=True
    def test_03_causal_features(self):
        c=np.linspace(-2.,2.,50);initial=c[::-1].copy()
        p=K.features(c,initial,269)
        self.assertTrue(np.array_equal(p,np.r_[np.tanh(c),np.tanh(initial),269/528.]))
        self.assertEqual(K.features(c,initial,700)[-1],1.)
        with self.assertRaises(AssertionError):K.features(c[:-1],initial,1)
        with self.assertRaises(AssertionError):K.features(c,initial,-1)
    def test_04_input_signature(self):
        self.assertEqual(tuple(inspect.signature(K.features).parameters),('current','initial','k'))
        self.assertEqual(tuple(inspect.signature(K.request).parameters),('weights','phi','rng'))
        self.assertFalse(any(s in inspect.getsource(K).lower() for s in ('import mujoco','import jax','import optax')))
    def test_05_det_no_rng(self):
        w=K.initial_weights();p=np.zeros(101);r=np.random.default_rng(5);before=copy.deepcopy(r.bit_generator.state)
        t=K.request(w,p)
        self.assertEqual(r.bit_generator.state,before);self.assertIsNone(t['rng_before'])
        self.assertTrue(np.array_equal(t['boxrequest'],np.zeros(10)))
    def test_06_iid_normal_and_rng(self):
        w=nonzero_weights();p=np.linspace(-.2,.3,101);a=np.random.default_rng(306);b=np.random.default_rng(306)
        before=copy.deepcopy(a.bit_generator.state);t=K.request(w,p,a)
        mean,raw=independent_forward(w,p[None,:],'actor');noise=1e-4*b.normal(size=10)
        for key,val in (('mean',mean[0]),('rawmean',raw[0]),('noise',noise),
                        ('proposal',mean[0]+noise),('boxrequest',np.clip(mean[0]+noise,-1,1))):
            self.assertTrue(np.array_equal(t[key],val))
        self.assertEqual(t['rng_before'],before);self.assertEqual(t['rng_after'],b.bit_generator.state)
    def test_07_box_boundary_not_second_tanh(self):
        class R:
            class B:state={'synthetic':True}
            bit_generator=B()
            def normal(self,size):return np.full(size,30000.)
        t=K.request(K.initial_weights(),np.zeros(101),R())
        self.assertTrue(np.array_equal(t['proposal'],np.full(10,3.)))
        self.assertTrue(np.array_equal(t['boxrequest'],np.ones(10)))
    def test_08_independent_forward(self):
        w=nonzero_weights();b=batch()
        for g in ('actor','value'):
            a,c=K.forward(w,b['phi'],g);v,raw=independent_forward(w,b['phi'],g)
            self.assertTrue(np.array_equal(a,v));self.assertTrue(np.array_equal(c[3],raw))
    def test_09_strict_positive_only(self):
        w=K.initial_weights();x=np.zeros((3,101));v=K.forward(w,x,'value')[0][:,0]
        b=dict(phi=x,nextphi=x,actions=np.ones((3,10))*.2,rewards=v+np.array([-1.,0.,1.]),
               bootstrap=np.zeros(3),home_tail=np.zeros(3))
        t=K.gradients(w,**b)
        self.assertTrue(np.array_equal(t['positive'],[False,False,True]))
        self.assertAlmostEqual(t['actor_loss'],.5*10*.04/3)
        self.assertTrue(np.array_equal(t['gradients']['actor_b3'],np.full(10,-.2/3)))
    def test_10_target_before_value_semi_gradient(self):
        w=nonzero_weights();b=batch();t=K.gradients(w,**b)
        pv=independent_forward(w,b['phi'],'value')[0][:,0];nv=independent_forward(w,b['nextphi'],'value')[0][:,0]
        y=b['rewards']+(.99**.2)*(b['bootstrap']*nv+b['home_tail'])
        self.assertTrue(np.array_equal(t['y'],y));self.assertTrue(np.array_equal(t['delta'],y-pv))
        self.assertTrue(np.array_equal(t['positive'],y>pv))
    def test_11_independent_per_sample_backprop(self):
        w=nonzero_weights();b=batch();t=K.gradients(w,**b);maximum=0.
        for g in ('actor','value'):
            ig=independent_grad(w,b,g,t['y'],t['positive'])
            for k,v in ig.items():maximum=max(maximum,float(np.max(np.abs(v-t['gradients'][k]))))
        self.assertLess(maximum,1e-13);EVIDENCE['independent_scalar_gradient_max_error']=maximum
    def coordinate_diff(self,group):
        w=nonzero_weights();b=batch();t=K.gradients(w,**b);maximum=0.;eps=1e-6
        for k in w:
            if not k.startswith(group):continue
            for flat in sorted(set((0,w[k].size//2,w[k].size-1))):
                plus={n:v.copy() for n,v in w.items()};minus={n:v.copy() for n,v in w.items()}
                plus[k].flat[flat]+=eps;minus[k].flat[flat]-=eps
                fd=(frozen_loss(plus,b,group,t['y'],t['positive'])-
                    frozen_loss(minus,b,group,t['y'],t['positive']))/(2*eps)
                maximum=max(maximum,abs(fd-t['gradients'][k].flat[flat]))
        self.assertLess(maximum,1e-8);EVIDENCE[group+'_coordinate_finite_difference_max_error']=maximum
    def test_12_value_finite_difference(self):self.coordinate_diff('value')
    def test_13_actor_finite_difference(self):self.coordinate_diff('actor')
    def test_14_full_directional_finite_difference(self):
        w=nonzero_weights();b=batch();t=K.gradients(w,**b);r=np.random.default_rng(811);eps=1e-6;maximum=0.
        for group in ('actor','value'):
            d={k:r.normal(size=v.shape) if k.startswith(group) else np.zeros_like(v) for k,v in w.items()}
            norm=np.sqrt(sum(np.sum(v*v) for v in d.values()));d={k:v/norm for k,v in d.items()}
            plus={k:v+eps*d[k] for k,v in w.items()};minus={k:v-eps*d[k] for k,v in w.items()}
            fd=(frozen_loss(plus,b,group,t['y'],t['positive'])-frozen_loss(minus,b,group,t['y'],t['positive']))/(2*eps)
            exact=sum(float(np.sum(t['gradients'][k]*d[k])) for k in d if k.startswith(group))
            maximum=max(maximum,abs(fd-exact))
        self.assertLess(maximum,1e-8);EVIDENCE['full_directional_difference_max_error']=maximum
    def test_15_separate_globalnorm(self):
        g={'one':np.array([3.,4.]),'two':np.zeros(2)};c,n,s=K.clip_gradients(g)
        self.assertEqual(n,5.);self.assertEqual(s,.2);self.assertTrue(np.array_equal(c['one'],g['one']*.2))
        z,n,s=K.clip_gradients({'zero':np.zeros(3)});self.assertEqual(n,0.);self.assertEqual(s,1.)
        with self.assertRaises(AssertionError):K.clip_gradients({'bad':np.array([np.inf])})
    def test_16_plain_sgd_exact(self):
        l=K.Learner(nonzero_weights());b=batch(20);t=l.update(b)
        for k in l.weights:self.assertTrue(np.array_equal(l.weights[k],t['before'][k]-1e-4*t['clipped_gradients'][k]))
        self.assertEqual(l.value_count,1);self.assertEqual(l.actor_count,1)
        self.assertFalse(l.state()['adam_or_target_network'])
    def test_17_no_positive_no_actor_change(self):
        l=K.Learner(nonzero_weights());t=l.update(batch(10,True))
        self.assertFalse(t['actor_updated']);self.assertEqual(l.actor_count,0);self.assertEqual(l.value_count,1)
        for k in l.weights:
            if k.startswith('actor'):self.assertTrue(np.array_equal(t['before'][k],t['after'][k]))
        self.assertEqual(t['actor_loss'],0.)
    def test_18_mask_frozen_before_both_updates(self):
        l=K.Learner(nonzero_weights());b=batch(14);t=l.update(b);idx=t['indices']
        bb={k:v[idx] for k,v in b.items()};prior=K.gradients(t['before'],**bb)
        self.assertTrue(np.array_equal(t['positive'],prior['positive']))
        self.assertTrue(np.array_equal(t['delta'],prior['delta']))
        self.assertEqual(t['state_before']['value_count'],0);self.assertEqual(t['state_after']['value_count'],1)
    def test_19_complete_home_tail_and_exponents(self):
        r=np.random.default_rng(12);phi=r.normal(size=(2279,101));a=np.zeros((2279,10));reward=r.normal(size=2279)
        t=K.episode_transitions(phi,a,reward,False)
        tail=sum(float(reward[k])*(.99**.2)**(k-529) for k in range(529,2279))
        self.assertEqual(len(t['controls']),528);self.assertEqual(t['bootstrap'][-1],0.)
        self.assertLess(abs(t['home_tail'][-1]-tail),1e-12)
        self.assertTrue(np.array_equal(t['full_rewards'],reward))
        self.assertTrue(np.array_equal(t['rewards'],reward[1:529]))
        self.assertTrue(np.all(t['home_tail'][:-1]==0.))
    def test_20_invalid_terminal_no_bootstrap(self):
        n=46;phi=np.arange(n*101,dtype=np.float64).reshape(n,101);a=np.zeros((n,10));r=np.arange(n,dtype=np.float64)
        t=K.episode_transitions(phi,a,r,True)
        self.assertEqual(len(t['controls']),45);self.assertEqual(t['controls'][-1],45)
        self.assertEqual(t['bootstrap'][-1],0.);self.assertTrue(np.all(t['nextphi'][-1]==0.))
        self.assertTrue(np.all(t['home_tail']==0.));self.assertEqual(t['rewards'][-1],45.)
    def test_21_episode_offsets_no_cross_reset(self):
        es=[]
        for n in (4,6):
            es.append(K.episode_transitions(np.ones((n,101))*n,np.zeros((n,10)),np.arange(n,dtype=np.float64),True))
        t=K.merge_episodes(es)
        self.assertTrue(np.array_equal(t['episode_offsets'],[0,3,8]))
        self.assertTrue(np.array_equal(t['full_reward_offsets'],[0,4,10]))
        self.assertTrue(np.array_equal(t['bootstrap'],[1.,1.,0.,1.,1.,1.,1.,0.]))
    def test_22_checkpoint_raw_arrays_counts_rng(self):
        l=K.Learner();l.update(batch())
        with tempfile.TemporaryDirectory(prefix='cacla_r205_numeric_',dir=OUTPUT) as d:
            stem=Path(d)/'state';l.checkpoint(stem)
            with np.load(str(stem)+'.npz',allow_pickle=False) as z:self.assert_arrays(dict(z),l.weights)
            self.assertEqual(json.loads(Path(str(stem)+'.json').read_text()),l.state())
            with self.assertRaises(AssertionError):l.checkpoint(stem)
    def test_23_two_full_synthetic_rounds_saved_decisions(self):
        l=K.Learner();r=np.random.default_rng(812);maximum=0.;checked=0
        for round_no in range(2):
            es=[]
            for episode in range(25):
                n=24
                phi=r.normal(size=(n,101))*.15
                a=np.clip(r.normal(size=(n,10))*.05,-1.,1.)
                rewards=r.normal(size=n)*.2
                es.append(K.episode_transitions(phi,a,rewards,True))
            replay=K.merge_episodes(es)
            dest=None if OUTPUT is None else OUTPUT/('synthetic_round%d'%(round_no+1))
            ts=l.train_round(replay,dest)
            self.assertEqual(len(ts),64)
            for t in ts:
                idx=t['indices'];b={k:replay[k][idx] for k in ('phi','nextphi','actions','rewards','bootstrap','home_tail')}
                nv=independent_forward(t['before'],b['nextphi'],'value')[0][:,0]
                pv=independent_forward(t['before'],b['phi'],'value')[0][:,0]
                y=b['rewards']+(.99**.2)*(b['bootstrap']*nv+b['home_tail'])
                self.assertTrue(np.array_equal(t['y'],y))
                self.assertTrue(np.array_equal(t['positive'],(y-pv)>0.))
                for group in ('actor','value'):
                    ig=independent_grad(t['before'],b,group,y,(y-pv)>0.)
                    for k,v in ig.items():
                        maximum=max(maximum,float(np.max(np.abs(v-t['raw_gradients'][k]))))
                for k in l.weights:
                    expected=t['before'][k]
                    if k.startswith('value') or t['actor_updated']:expected=expected-1e-4*t['clipped_gradients'][k]
                    self.assertTrue(np.array_equal(expected,t['after'][k]));checked+=1
                reference=np.random.default_rng();reference.bit_generator.state=copy.deepcopy(t['state_before']['learning_rng'])
                self.assertTrue(np.array_equal(reference.integers(0,len(replay['phi']),size=512,dtype=np.int64),idx))
                self.assertEqual(reference.bit_generator.state,t['state_after']['learning_rng'])
                if dest is not None:
                    step=t['state_before']['value_count']%64
                    with np.load(dest/('update_%03d.npz'%step),allow_pickle=False) as z:
                        for name in ('indices','y','pred','delta','positive','mean','rawmean','action_error'):
                            self.assertTrue(np.array_equal(z[name],t[name]))
                        for p in ('before','after','raw_gradients','clipped_gradients'):
                            for k,v in t[p].items():self.assertTrue(np.array_equal(z[p+'__'+k],v))
                    state=json.loads((dest/('update_%03d.json'%step)).read_text())
                    self.assertEqual(state['state_before'],t['state_before']);self.assertEqual(state['state_after'],t['state_after'])
            self.assertEqual(l.round_count,round_no+1)
        self.assertEqual(l.value_count,128);self.assertLessEqual(l.actor_count,128)
        self.assertLess(maximum,1e-12)
        if OUTPUT is not None:l.checkpoint(OUTPUT/'synthetic_final')
        EVIDENCE['synthetic_two_rounds']=dict(value_updates=l.value_count,actor_updates=l.actor_count,
            independent_all128_scalar_gradient_max_error=maximum,
            checked_before_after_weight_blocks=checked,stored_gradient_sgd_rng_raw_array_exact=True,
            original_production_trace=False,new_dynamic_attempts=0)
    def test_24_round_budget_and_current_only(self):
        l=K.Learner();b=batch();orig={k:v.copy() for k,v in b.items()}
        l.round_count=2
        with self.assertRaises(AssertionError):l.train_round(b)
        with self.assertRaises(AssertionError):l.update(b)
        l.round_count=0;l.value_count=128
        with self.assertRaises(AssertionError):l.update(b)
        self.assert_arrays(b,orig)
        self.assertFalse(hasattr(l,'replay'));self.assertFalse(hasattr(l,'target'))
        self.assertFalse(hasattr(l,'adam'))
if __name__=='__main__':unittest.main()

