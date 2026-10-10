"""Fresh causal total-target policy and one finite antithetic parameter update.
No simulator, Q/critic, replay, per-control innovations, optimizer or old weights.
"""
import numpy as np

SEED=298
DIRECTION_SEED=29801
SIGMA=1e-5
RATE=1e-4
DIRECTIONS=2
PARAMETERS=1802
BLOCKS=('actor_w0','actor_b0','actor_w1','actor_b1')
SHAPES=((101,16),(16,),(16,10),(10,))

def finite(value,shape=None):
    a=np.asarray(value,dtype=np.float64)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('Nonfinite or invalid causal policy input')
    return a

def features(current,initial,control):
    current=finite(current,(50,));initial=finite(initial,(50,))
    if not isinstance(control,(int,np.integer)) or not 0<=control<=2279:raise ValueError('Invalid control')
    return np.r_[np.tanh(current),np.tanh(initial),min(control/528.,1.)]

def initial_weights():
    rng=np.random.default_rng(SEED)
    return dict(actor_w0=rng.normal(size=(101,16))/np.sqrt(101.),
                actor_b0=np.zeros(16),actor_w1=np.zeros((16,10)),actor_b1=np.zeros(10))

def flatten(weights):
    if set(weights)!=set(BLOCKS):raise ValueError('Unexpected parameter keys')
    return np.concatenate([finite(weights[k],s).reshape(-1) for k,s in zip(BLOCKS,SHAPES)])

def unflatten(theta):
    theta=finite(theta,(PARAMETERS,));offset=0;weights={}
    for key,shape in zip(BLOCKS,SHAPES):
        size=int(np.prod(shape));weights[key]=theta[offset:offset+size].reshape(shape).copy();offset+=size
    assert offset==PARAMETERS
    return weights

def network(weights,x,name='actor'):
    if name!='actor':raise ValueError('No critic or other network')
    flatten(weights);x=finite(x)
    if x.ndim not in (1,2) or x.shape[-1]!=101:raise ValueError('Invalid features')
    h=np.tanh(x@weights['actor_w0']+weights['actor_b0'])
    return np.tanh(h@weights['actor_w1']+weights['actor_b1'])

def proposal_set(weights):
    theta=flatten(weights);rng=np.random.default_rng(DIRECTION_SEED)
    before=rng.bit_generator.state
    eps=rng.normal(size=(DIRECTIONS,PARAMETERS))
    proposals=[unflatten(theta+sign*SIGMA*eps[i]) for i in range(DIRECTIONS) for sign in (1.,-1.)]
    return proposals,eps,before,rng.bit_generator.state

def update(weights,directions,scores):
    theta=flatten(weights);eps=finite(directions,(DIRECTIONS,PARAMETERS))
    scores=finite(scores,(DIRECTIONS,2))
    coefficients=(scores[:,0]-scores[:,1])/(2.*SIGMA)
    gradient=np.mean(coefficients[:,None]*eps,axis=0)
    finite(gradient,(PARAMETERS,));norm=float(np.linalg.norm(gradient))
    if not np.isfinite(norm):raise ValueError('Nonfinite update norm')
    bounded=gradient/max(1.,norm)
    result=unflatten(theta+RATE*bounded)
    changes={k:float(np.linalg.norm(result[k]-weights[k])) for k in BLOCKS}
    return result,dict(scores=scores.copy(),coefficients=coefficients,gradient=gradient,
                       gradient_norm=norm,bounded_gradient=bounded,changes=changes)

def complete_score(returns):
    # Full original undiscounted episode sums, INCLUDING failed/invalid rows.
    return float(np.mean(finite(returns,(25,))))

def replace_legs(original,reference,action,legs,lower,upper,active):
    if not active:return original
    original=finite(original,(14,));reference=finite(reference,(14,))
    action=finite(action,(10,));legs=np.asarray(legs)
    lower=finite(lower,(14,));upper=finite(upper,(14,))
    if legs.shape!=(10,) or not np.issubdtype(legs.dtype,np.integer) or len(set(legs.tolist()))!=10 or np.any(legs<0) or np.any(legs>=14):
        raise ValueError('Invalid leg mapping')
    if np.any(lower>upper) or np.any(np.abs(action)>1.):raise ValueError('Invalid target bounds')
    out=original.copy()
    out[legs]=np.clip(reference[legs]+.18*action,lower[legs],upper[legs])
    return out

