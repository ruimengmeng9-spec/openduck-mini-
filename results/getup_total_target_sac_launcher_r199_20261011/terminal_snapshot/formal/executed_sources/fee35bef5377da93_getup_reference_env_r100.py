"""R100 complete-fall reference-residual environment, original 50Hz physics.

No intermediate-state library. Every reset prepares a real left-side fall.
Only the combined bounded target correction is learned; the reference path,
stand phase and StrictSim constraints remain intact.
"""
import os
import traceback
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT
from diagnostics.getup_fullfallen_env_r32 import observation as native_observation
from diagnostics.getup_fullfallen_contract_r32 import potential
from diagnostics.search_getup_reference_feedback_r64 import features,residual
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


CAP=.18
GAMMA=.99**.2  # Same time discount as R98, at 50Hz instead of 10Hz.
TRAIN=tuple(range(769000,769008))+tuple(range(773000,773016))


def checked_residual(action):
    action=np.asarray(action,dtype=float)
    if action.shape!=(10,) or not np.isfinite(action).all() or np.abs(action).max()>1.+1e-6:
        raise ValueError('Ten finite normalized residuals required')
    return CAP*np.clip(action,-1.,1.)


class ReferenceEpisode:
    def __init__(self,scene,stand,path_file,seed,qualification=False):
        self.sim=StrictSim(scene,stand)
        with np.load(path_file,allow_pickle=False) as data:
            self.targets=data['targets'].copy();self.phases=data['phases'].copy()
            self.reference=data['reference'].copy();self.gains=data['gains'].copy()
        self.recovery_controls=int(np.count_nonzero(self.phases!=7))
        if not 0<self.recovery_controls<600:
            raise ValueError('Reference recovery must fit the frozen 12s entry deadline')
        self.maximum_controls=len(self.targets) if qualification else self.recovery_controls+100
        self.ids=np.array([self.sim.model.actuator(name).id for name in JOINTS])
        self.rng=np.random.default_rng(seed)
        self.reset()

    def observe(self):
        k=min(self.controls,len(self.targets)-1)
        error=features(self.sim)-self.reference[k]
        return np.concatenate([native_observation(self.sim),error,
            [min(self.controls/self.recovery_controls,1.)]]).astype(np.float32)

    def phi(self,metric):
        body=self.sim.model.body('trunk_assembly').id
        return potential(metric,float(self.sim.data.xmat[body].reshape(3,3)[2,0]))

    def reset(self,seed_override='sample'):
        self.case_seed=(None if self.rng.random()<.1 else int(self.rng.choice(TRAIN))) if seed_override=='sample' else seed_override
        self.sim.prepare('left_side',0 if self.case_seed is None else self.case_seed,self.case_seed is not None)
        self.initial=self.sim.measure();self.initial_hash=state_hash(self.sim)
        if not (self.initial['up_z']<.5 and self.initial['torso_contact']):
            raise RuntimeError('Actual complete fallen start required')
        self.sim.clear_audit()
        self.controls=self.tail=0;self.return_sum=0.;self.entry_time=None
        self.previous_phi=self.phi(self.initial)
        return self.observe()

    def step(self,action,auto_reset=True):
        extra=checked_residual(action)
        k=self.controls;phase=int(self.phases[k]);sim=self.sim
        error=features(sim)-self.reference[k]
        correction=residual(error,self.gains[phase])
        if phase!=7:
            correction=np.clip(correction+extra,-CAP,CAP)
        # Phase 7 is literally the existing home hold and legacy feedback.
        target=self.targets[k].copy()
        target[self.ids]=np.clip(target[self.ids]+correction,sim.lower[self.ids],sim.upper[self.ids])
        sim.step_target(target);self.controls+=1
        metric=sim.measure();valid=sim.physical_valid()
        self.tail=self.tail+1 if valid and metric['stable'] else 0
        if self.tail>=10 and self.entry_time is None:self.entry_time=self.controls*DT
        phi=self.phi(metric) if valid else self.previous_phi
        reward=(GAMMA*phi-self.previous_phi+.04*bool(metric['stable'])-.002) if valid else -5.
        self.previous_phi=phi
        terminated=not valid;done=terminated or self.controls>=self.maximum_controls
        success=bool(valid and self.tail*DT>=1.-1e-8 and self.entry_time is not None and self.entry_time<=12.)
        if done and success:reward+=10.
        self.return_sum+=reward
        bootstrap=self.observe() if valid else np.zeros(55,dtype=np.float32)
        info=None
        if done:
            info=dict(pose='left_side',case_seed=self.case_seed,initial_hash=self.initial_hash,
                      actual_fallen_start=True,valid=bool(valid),training_success=success,
                      controls=self.controls,strict_tail_s=self.tail*DT,entry_time_s=self.entry_time,
                      return_sum=float(self.return_sum),final=metric,peaks=sim.peaks.copy(),
                      prefix_start=False,root_edits_after_initialization=0)
        obs=self.reset() if done and auto_reset else bootstrap
        return obs,float(reward),bool(done),bool(terminated),bootstrap,info,1


def pipe_worker(connection,scene,stand,number,seed,path_file):
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        envs=[ReferenceEpisode(scene,stand,path_file,seed+i) for i in range(number)]
        connection.send(('ready',np.stack([e.observe() for e in envs])))
        while True:
            op,actions=connection.recv()
            if op=='close':break
            if op!='step':raise ValueError(op)
            connection.send(('ok',[e.step(a) for e,a in zip(envs,actions)]))
    except Exception as exc:
        connection.send(('error',type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally:connection.close()


def numpy_action(weights,obs):
    x=np.asarray(obs,dtype=np.float32)
    for layer in ('hidden0','hidden1','mean'):
        x=np.tanh(x@weights[layer+'_kernel']+weights[layer+'_bias'])
    return x


def full_trial(scene,stand,path_file,weights,seed,trace_file):
    env=ReferenceEpisode(scene,stand,path_file,200,qualification=True)
    env.reset(seed)
    initial=env.initial.copy();trace=[]
    while True:
        action=np.zeros(10) if weights is None else numpy_action(weights,env.observe())
        row=env.step(action,auto_reset=False)
        trace.append((float(env.sim.data.time),env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),
                      env.sim.prev.copy(),env.tail>0))
        if row[2]:break
    result=row[5]
    result.update(initial=initial,
                  success=bool(result['valid'] and result['controls']==len(env.targets)
                    and result['entry_time_s'] is not None and result['entry_time_s']<=12.
                    and result['strict_tail_s']>=30.-1e-8))
    if trace_file:
        np.savez_compressed(trace_file,time=[r[0] for r in trace],qpos=[r[1] for r in trace],
            qvel=[r[2] for r in trace],applied=[r[3] for r in trace],strict=[r[4] for r in trace])
    return result
