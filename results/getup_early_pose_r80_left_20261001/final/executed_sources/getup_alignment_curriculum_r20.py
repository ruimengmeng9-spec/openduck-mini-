"""R20 reward ablation: preserve physical dynamics and strict success gates."""
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode,at_goal,physical_safe,reward_terms
from diagnostics.getup_macro_curriculum_r15 import advance_macro
from diagnostics.getup_independent_native import DT,SLEW


def training_reward_terms(metric,previous_height,home_error,mode):
    terms=reward_terms(metric,previous_height,home_error)
    if mode=='base':return terms
    if mode!='height':raise ValueError('unknown reward mode')
    height=float(metric['height_m']);up=float(metric['up_z'])
    upright=np.exp(-((np.arccos(np.clip(up,-1.,1.))/.4)**2))
    # Change reward only, never the 0.155--0.185 m success-height interval.
    tight_height=np.exp(-((height-.165)/.01)**2)
    pose=np.exp(-(home_error/.6)**2)
    terms['goal']=4.*tight_height*upright*pose*metric['foot_load_fraction']
    terms['height_error']=-80.*abs(height-.165)
    return terms


class AlignmentEpisode(NativeEpisode):
    def __init__(self,*args,reward_mode='base',**kwargs):
        if reward_mode not in ('base','height'):raise ValueError('unknown reward mode')
        self.reward_mode=reward_mode
        super().__init__(*args,**kwargs)

    def step(self,normalized):
        action=np.asarray(normalized,dtype=float)
        if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('bad recovery action')
        target=(self.sim.lower+self.sim.upper)/2+np.clip(action,-1.,1.)*(self.sim.upper-self.sim.lower)/2
        before=self.sim.prev.copy();applied=self.sim.step_target(target)
        if np.max(np.abs(applied-before))>SLEW*DT+1e-8:raise RuntimeError('slew contract violated')
        self.steps+=1;m=self.sim.measure();safe=physical_safe(self.sim,m)
        self.episode_safe=self.episode_safe and safe
        terms=training_reward_terms(m,self.previous_height,m['joint_home_error_mean_rad'],self.reward_mode)
        self.previous_height=m['height_m'];reward=sum(terms.values()) if safe else -20.
        self.stable_tail=self.stable_tail+1 if at_goal(m) else 0
        self.episode_return+=reward
        terminated=(not safe) or m['up_z']<.45;done=terminated or self.steps>=200
        bootstrap=self.sim.observation();info=None
        if done:
            info=dict(success=bool(self.stable_tail*DT>=2. and self.episode_safe),
                      return_sum=self.episode_return,steps=self.steps,initial_height_m=self.initial['height_m'],
                      final_height_m=m['height_m'],final_up_z=m['up_z'],foot_load_fraction=m['foot_load_fraction'],
                      safe=self.episode_safe,reset_rejections=self.reset_rejections,reward_mode=self.reward_mode)
            obs=self.reset()
        else:obs=bootstrap
        return obs,float(reward),bool(done),bool(terminated),bootstrap,info


def pipe_worker(connection,scene,stand,starts,number,seed,tilt_max,repeat,reward_mode):
    import os
    os.environ['JAX_PLATFORMS']='cpu'
    try:
        envs=[AlignmentEpisode(scene,stand,starts,seed+i,tilt_max,reward_mode=reward_mode) for i in range(number)]
        connection.send(('ready',np.stack([e.reset() for e in envs])))
        while True:
            operation,payload=connection.recv()
            if operation=='close':break
            if operation!='step':raise ValueError('unknown macro operation')
            connection.send(('ok',[advance_macro(e,a,repeat,.99**(1/repeat)) for e,a in zip(envs,payload)]))
    except Exception as exc:
        import traceback
        connection.send(('error',type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally:connection.close()
