"""Causal online identification and fixed three-step sensor MPC, no simulator."""
import numpy as np

HORIZON=529
CALIBRATION=64
BOUND=1e-4
EXCITATION=1e-5
FORGET=.995
PREDICTION=3
PENALTY=.001
SCALE=np.r_[np.ones(3),np.full(31,.05)]
SCALE.setflags(write=False)

class Mapping:
    def __init__(self,model):
        if model.nu!=14:raise ValueError('Original actuator mapping required')
        self.legs=np.array([model.actuator(side+'_'+part).id for side in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
        self.head=np.array([i for i in range(14) if i not in self.legs])

def finite(value,shape):
    x=np.asarray(value,dtype=float)
    if x.shape!=shape or not np.isfinite(x).all():raise ValueError('Finite causal arrays required')
    return x

def delta(current,nominal,initial,initial_nominal):
    a=[finite(np.asarray(v,dtype=np.float32),(34,)) for v in (current,nominal,initial,initial_nominal)]
    return ((a[0]-a[1])-(a[2]-a[3]))/SCALE

def innovations():
    rng=np.random.default_rng(295)
    noise=rng.normal(0,EXCITATION,(HORIZON,10))
    probe=rng.normal(size=10);probe/=np.linalg.norm(probe)
    noise[0]=0
    return noise,probe,rng.bit_generator.state

class OnlineModel:
    def __init__(self):
        self.theta=np.zeros((34,78));self.p=100*np.eye(78)
        self.previous=None;self.previous_change=np.zeros(34);self.pending=None
        self.count=0;self.error=np.zeros(34)
    def observe(self,x):
        x=finite(x,(34,))
        change=np.zeros(34) if self.previous is None else x-self.previous
        self.error=np.zeros(34)
        if self.pending is not None:
            f=self.pending;pf=self.p@f;den=FORGET+f@pf
            if not np.isfinite(den) or den<=0:raise ArithmeticError('Invalid covariance')
            self.error=change-self.theta@f
            self.theta+=np.outer(self.error,pf/den)
            self.p=(self.p-np.outer(pf,pf)/den)/FORGET
            self.p=(self.p+self.p.T)*.5
            finite(self.theta,(34,78));finite(self.p,(78,78))
            if np.abs(self.p).max()>1e8:raise ArithmeticError('Covariance bound')
            self.count+=1
        self.previous=x.copy();self.previous_change=change.copy();self.pending=None
        return change
    def commit(self,executed_difference):
        if self.previous is None or self.pending is not None:raise RuntimeError('Observe then commit once')
        u=finite(executed_difference,(10,))/BOUND
        self.pending=np.r_[self.previous,self.previous_change,u]
    def effect(self,executed_difference):
        return self.theta[:,68:]@(finite(executed_difference,(10,))/BOUND)
    def predict(self,x,change,u):
        return finite(x,(34,))+self.theta@np.r_[x,finite(change,(34,)),finite(u,(10,))]
    def mpc(self,x,change):
        # Constant ten-leg normalized action over three mathematical sensor steps.
        x=finite(x,(34,));change=finite(change,(34,))
        state=np.r_[x,change]
        a=self.theta[:,:34];c=self.theta[:,34:68];b=self.theta[:,68:]
        transition=np.block([[np.eye(34)+a,c],[a,c]])
        input_matrix=np.vstack((b,b));response=np.zeros((68,10))
        lhs=PENALTY*np.eye(10);rhs=np.zeros(10)
        for _ in range(PREDICTION):
            state=transition@state;response=transition@response+input_matrix
            lhs+=response[:34].T@response[:34]/34
            rhs+=response[:34].T@state[:34]/34
        action=np.clip(-np.linalg.solve(lhs,rhs),-1,1)
        finite(action,(10,))
        return BOUND*action

class Controller:
    def __init__(self,mode):
        if mode not in ('zero','positive','negative','candidate'):raise ValueError('Fixed experiment mode')
        self.mode=mode;self.model=OnlineModel();self.noise,self.probe,self.rng=innovations()
        self.fallbacks=[];self.x=np.zeros(34);self.change=np.zeros(34);self.active=False
    def request(self,current,nominal,initial,initial_nominal,control):
        if not isinstance(control,(int,np.integer)) or control<0:raise ValueError('Control index')
        self.active=False;raw=np.zeros(10);self.x=np.zeros(34);self.change=np.zeros(34)
        if self.mode=='zero' or control>=HORIZON:return raw
        self.x=delta(current,nominal,initial,initial_nominal)
        if control==0:self.x[:]=0
        try:
            self.change=self.model.observe(self.x);self.active=True
            if control==0:return raw
            activation=float(np.tanh(np.sqrt(np.mean(self.x*self.x))))
            if self.mode=='candidate' and control>=CALIBRATION:
                raw=self.model.mpc(self.x,self.change)
            else:
                raw=activation*self.noise[control]
                if control==CALIBRATION:
                    raw+=activation*EXCITATION*self.probe*(1 if self.mode=='positive' else -1)
            return np.clip(raw,-BOUND,BOUND)
        except (ArithmeticError,np.linalg.LinAlgError):
            self.fallbacks.append(control);self.model.pending=None;self.active=False
            return np.zeros(10)
    def commit(self,executed_difference):
        if self.active:self.model.commit(executed_difference)

