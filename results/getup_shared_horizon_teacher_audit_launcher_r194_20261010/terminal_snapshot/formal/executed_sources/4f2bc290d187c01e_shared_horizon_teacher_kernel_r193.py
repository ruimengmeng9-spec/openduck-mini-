"""Shared full-horizon teacher; causal deviation scale, no metadata or oracle."""
import numpy as np
HORIZON=529
MODES=4
BOUND=1e-4
SCALE=np.r_[np.ones(3),np.full(3,.05),np.full(14,.05),np.full(14,.05)]
BASIS=np.sin(np.pi*np.arange(HORIZON)[:,None]/(HORIZON-1)*np.arange(1,MODES+1)[None,:])
BASIS[[0,-1]]=0.
BASIS.setflags(write=False);SCALE.setflags(write=False)

class Mapping:
    def __init__(self,model):
        if model.nu!=14:raise ValueError('Original 14 actuators required')
        self.legs=np.array([model.actuator(side+'_'+part).id for side in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
        self.head=np.array([i for i in range(14) if i not in self.legs])

def validate_coefficients(coefficients):
    c=np.asarray(coefficients,dtype=float)
    if c.shape!=(10,MODES) or not np.isfinite(c).all() or np.abs(c).max()>BOUND:raise ValueError('Finite bounded ten-leg global teacher coefficients required')
    return c

def teacher_request(current,nominal,initial,initial_nominal,coefficients,mapping,control):
    c=validate_coefficients(coefficients)
    xs=[np.asarray(x,dtype=np.float32) for x in (current,nominal,initial,initial_nominal)]
    if any(x.shape!=(34,) or not np.isfinite(x).all() for x in xs):raise ValueError('Finite native34 only')
    if not isinstance(control,(int,np.integer)) or control<0:raise ValueError('Nonnegative integer control required')
    delta=np.zeros(34);wave=np.zeros(10);request=np.zeros(14);activation=0.
    if 0<control<HORIZON:
        delta=((xs[0].astype(float)-xs[1].astype(float))-(xs[2].astype(float)-xs[3].astype(float)))/SCALE
        activation=float(np.tanh(np.sqrt(np.mean(delta*delta))))
        wave=c@BASIS[control]
        request[mapping.legs]=activation*wave
    return request,delta,wave,activation

def make_proposals(seed=293):
    rng=np.random.default_rng(seed)
    directions=np.clip(rng.normal(0,1e-5,size=(2,10,MODES)),-BOUND,BOUND)
    proposals=[np.zeros((10,MODES)),directions[0],-directions[0],directions[1],-directions[1]]
    return proposals,rng.bit_generator.state

def retention(report,baseline):
    if len(report['rows'])!=len(baseline['rows']):raise ValueError('Complete paired rows required')
    preserved=[];rescued=[];regressed=[]
    for row,base in zip(report['rows'],baseline['rows']):
        if row['case_seed']!=base['case_seed'] or row['initial_hash']!=base['initial_hash']:raise ValueError('Paired original starts required')
        if base['success'] and row['success']:preserved.append(row['case_seed'])
        if not base['success'] and row['success']:rescued.append(row['case_seed'])
        if base['success'] and not row['success']:regressed.append(row['case_seed'])
    eligible=bool(report['nominal_success'] and not report['physical_failures'] and not regressed and report['successes']>baseline['successes'])
    return dict(preserved=preserved,rescued=rescued,regressed=regressed,eligible=eligible)

def choose_teacher(reports):
    baseline=reports[0]
    evidence=[retention(r,baseline) for r in reports]
    eligible=[i for i,e in enumerate(evidence) if i and e['eligible']]
    def rank(i):
        r=reports[i];returns=[x['return_sum'] for x in r['rows']]
        return r['successes'],min(returns),sum(returns),-i
    return (max(eligible,key=rank) if eligible else 0),evidence

