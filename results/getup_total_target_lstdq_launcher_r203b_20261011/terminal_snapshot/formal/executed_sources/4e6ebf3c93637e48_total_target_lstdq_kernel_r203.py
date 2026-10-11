"""Finite affine-offset LSTD-Q policy iteration, no dynamics model or actor fitting."""
from pathlib import Path
import struct
import numpy as np

SEED=303
STATE_DIM=102
ACTION_DIM=10
PARAM_DIM=STATE_DIM*(1+ACTION_DIM)
BOUND=.18
SIGMA=1e-4
GAMMA=.99**.2
CURVATURE=1.
RIDGE=1e-3
MAX_SOLVES=2
BLOCK=256

def finite(a,shape=None):
    a=np.asarray(a,dtype=np.float64)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('Finite exact-shape array required')
    return a

def features(current,initial,control):
    if isinstance(control,bool) or int(control)!=control or not 0<=control<=2279:
        raise ValueError('Original integer control clock required')
    return np.r_[np.tanh(finite(current,(50,))),np.tanh(finite(initial,(50,))),
                 min(control/528.,1.),1.]

def initial_weights():
    # Fresh exact-zero critic; no old weights, optimizer, data, or RNG.
    return {'theta':np.zeros(PARAM_DIM,dtype=np.float64)}

def split(theta,d,m):
    theta=finite(theta,(d*(1+m),))
    return theta[:d],theta[d:].reshape(d,m)

def greedy(theta,phi,m=ACTION_DIM):
    phi=finite(phi)
    if phi.ndim not in (1,2) or phi.shape[-1]<1:raise ValueError('State feature rows required')
    _,cross=split(theta,phi.shape[-1],m)
    return np.clip(phi@cross/CURVATURE,-1.,1.)

def sample(theta,phi,rng):
    mean=greedy(theta,phi)
    if mean.shape!=(ACTION_DIM,):raise ValueError('One online state required')
    noise=rng.normal(size=ACTION_DIM)
    proposal=mean+SIGMA*noise
    request=np.clip(proposal,-1.,1.)
    # Clipping creates boundary atoms: do not claim a Gaussian executed density.
    return mean,noise,proposal,request

def basis(phi,action):
    phi=finite(phi);action=finite(action)
    if phi.ndim!=2 or action.ndim!=2 or len(phi)!=len(action) or not len(phi):
        raise ValueError('Nonempty paired rows required')
    if np.any(np.abs(action)>1.):raise ValueError('Normalized action box required')
    return np.concatenate((phi,(phi[:,:,None]*action[:,None,:]).reshape(len(phi),-1)),axis=1)

def offset(action):
    return -.5*CURVATURE*np.sum(np.asarray(action,dtype=np.float64)**2,axis=-1)

def q_value(theta,phi,action):
    phi=finite(phi);action=finite(action)
    v,b=split(theta,phi.shape[-1],action.shape[-1])
    return phi@v+np.sum((phi@b)*action,axis=-1)+offset(action)

def replace_legs(original,reference,request,legs,lower,upper,active):
    if not active:return original
    out=finite(original,(14,)).copy();ref=finite(reference,(14,))
    a=finite(request,(10,))
    if np.any(np.abs(a)>1.):raise ValueError('Box action required')
    ids=np.asarray(legs)
    if ids.shape!=(10,) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=10 or np.any((ids<0)|(ids>=14)):
        raise ValueError('Ten distinct original leg indices required')
    out[ids]=np.clip(ref[ids]+BOUND*a,finite(lower,(14,))[ids],finite(upper,(14,))[ids])
    return out

def complete_returns(reward):
    r=finite(reward)
    if r.ndim!=1 or not len(r):raise ValueError('Nonempty original episode reward required')
    out=np.empty_like(r);tail=0.
    for i in range(len(r)-1,-1,-1):tail=float(r[i])+GAMMA*tail;out[i]=tail
    return out

def episode_transitions(observations,rewards,requests,active):
    obs=finite(observations)
    if obs.ndim!=2 or obs.shape[1]!=50 or not 1<=len(obs)<=2279:
        raise ValueError('Complete original causal native50 episode required')
    n=len(obs);r=finite(rewards,(n,));a=finite(requests,(n,10));mask=np.asarray(active)
    expected=(np.arange(n)>0)&(np.arange(n)<529)
    if mask.shape!=(n,) or mask.dtype!=bool or not np.array_equal(mask,expected) or not np.any(mask):
        raise ValueError('Original active controls1..528 required')
    if np.any(np.abs(a)>1.):raise ValueError('Original box requests required')
    ids=np.flatnonzero(mask)
    x=np.stack([features(obs[i],obs[0],int(i)) for i in ids])
    nx=np.zeros_like(x);bootstrap=np.zeros(len(ids));backup=r[ids].copy()
    returns=complete_returns(r)
    for j,i in enumerate(ids):
        if i+1<n and mask[i+1]:
            nx[j]=features(obs[i+1],obs[0],int(i+1));bootstrap[j]=1.
        elif i+1<n:
            # Fixed original home tail, never another reset or learned home action.
            backup[j]+=GAMMA*returns[i+1]
    return dict(phi=x,next_phi=nx,action=a[ids].copy(),reward_backup=backup,
                bootstrap=bootstrap,control=ids,original_reward=r.copy(),
                original_full_returns=returns)

def lstd_system(theta,phi,action,next_phi,reward,bootstrap):
    phi=finite(phi);action=finite(action);n=len(phi)
    if phi.ndim!=2 or action.ndim!=2 or not n or len(action)!=n:raise ValueError('Training rows required')
    d=phi.shape[1];m=action.shape[1];split(theta,d,m)
    nx=finite(next_phi,phi.shape);r=finite(reward,(n,));b=finite(bootstrap,(n,))
    if np.any((b!=0)&(b!=1)):raise ValueError('Exact bootstrap0/1 required')
    if np.any(nx[b==0]!=0):raise ValueError('Terminal next features must be zero')
    next_action=greedy(theta,nx,m)
    next_action[b==0]=0.
    size=d*(1+m);matrix=np.zeros((size,size));rhs=np.zeros(size)
    for start in range(0,n,BLOCK):
        sl=slice(start,min(start+BLOCK,n))
        psi=basis(phi[sl],action[sl]);npsi=basis(nx[sl],next_action[sl])
        delta=psi-GAMMA*b[sl,None]*npsi
        corrected=r[sl]+GAMMA*b[sl]*offset(next_action[sl])-offset(action[sl])
        matrix+=psi.T@delta;rhs+=psi.T@corrected
    matrix/=n;rhs/=n
    # This is the nonsymmetric projected Bellman equation, NOT OLS TD regression.
    matrix.flat[::size+1]+=RIDGE
    return matrix,rhs,next_action

def solve_lstd(theta,phi,action,next_phi,reward,bootstrap):
    matrix,rhs,next_action=lstd_system(theta,phi,action,next_phi,reward,bootstrap)
    solution=np.linalg.solve(matrix,rhs)
    residual=matrix@solution-rhs
    scale=max(1.,float(np.linalg.norm(matrix,np.inf)*np.linalg.norm(solution,np.inf)),
              float(np.linalg.norm(rhs,np.inf)))
    relative=float(np.linalg.norm(residual,np.inf)/scale)
    if not np.isfinite(solution).all() or relative>1e-10:
        raise ArithmeticError('Finite LSTD solve with original numerical residual bound required')
    return dict(theta=solution,matrix=matrix,rhs=rhs,residual=residual,
                relative_residual=relative,next_action=next_action,
                action_before=greedy(theta,phi,action.shape[1]),
                action_after=greedy(solution,phi,action.shape[1]),
                fixed_curvature_not_reward_penalty=True,projected_equation_not_convergence_or_safety=True)

class Learner:
    def __init__(self,weights=None):
        self.theta=finite((initial_weights() if weights is None else weights)['theta'],(PARAM_DIM,)).copy()
        self.solves=0
    def numpy_weights(self):return {'theta':self.theta.copy()}
    def update(self,phi,action,next_phi,reward,bootstrap):
        if self.solves>=MAX_SOLVES:raise RuntimeError('Fixed two-solve budget closed')
        phi=finite(phi)
        if phi.ndim!=2 or phi.shape[1]!=STATE_DIM:raise ValueError('Production causal102 required')
        action=finite(action,(len(phi),ACTION_DIM))
        evidence=solve_lstd(self.theta,phi,action,next_phi,reward,bootstrap)
        self.theta=evidence['theta'].copy();self.solves+=1
        evidence['solve_count']=self.solves
        return evidence
    def state_bytes(self):
        return b'R203LSTDQv1\0'+struct.pack('<Q',self.solves)+self.theta.astype('<f8').tobytes()
    def checkpoint(self,stem):
        stem=Path(stem)
        np.savez_compressed(str(stem)+'.npz',theta=self.theta,solve_count=np.asarray(self.solves))
        Path(str(stem)+'.state').write_bytes(self.state_bytes())

