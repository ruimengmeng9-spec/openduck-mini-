"""Actual four-orientation fallen PPO environments, with substep audits.

Policy decisions are held for five 50Hz motor updates. Unlike the old mild-tilt
task, being down NEVER terminates an episode. Only physical violations or the
declared episode time limit do. Reset modifies state only between episodes.
"""
import os
import traceback
import numpy as np

from diagnostics.getup_independent_native import DT, POSES
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim
from diagnostics.getup_fullfallen_contract_r32 import DECISION_CONTROLS, GAMMA, potential, decode_action, bootstrap_observation


def observation(sim):
    # Retain the old native 50D observation layout for checkpoint compatibility.
    feet,_,_=sim.contacts()
    return np.concatenate([sim.sensor('gyro'),sim.sensor('upvector'),
        sim.data.qpos[sim.qadr]-sim.home,sim.data.qvel[sim.vadr]*.05,
        sim.prev-sim.home,np.asarray(feet,dtype=float)]).astype(np.float32)


class FullFallEpisode:
    def __init__(self,scene,stand,seed,hold_controls=DECISION_CONTROLS):
        self.sim=StrictSim(scene,stand)
        self.rng=np.random.default_rng(seed)
        self.hold_controls=hold_controls
        self.reset()

    def phi(self,m):
        body=self.sim.model.body('trunk_assembly').id
        gravity_x=self.sim.data.xmat[body].reshape(3,3)[2,0]
        return potential(m,float(gravity_x))

    def reset(self,pose_override=None):
        # Each actual fallen orientation is practiced from the first update.
        # 20% standing retains the hold skill; no near-standing substitute.
        self.pose=pose_override or ('standing',*POSES)[int(self.rng.integers(5))]
        # Held-out tests use >=1,300,000, disjoint from this training namespace.
        self.reset_seed=int(self.rng.integers(1000000))
        self.sim.prepare(self.pose,self.reset_seed,True)
        self.initial=self.sim.measure()
        self.initial_hash=state_hash(self.sim)
        if self.pose!='standing' and not (self.initial['up_z']<.5 and self.initial['torso_contact']):
            raise RuntimeError('Label does not describe an actual fallen start')
        self.sim.clear_audit()
        self.controls=0; self.tail_controls=0; self.return_sum=0.
        self.previous_phi=self.phi(self.initial)
        return observation(self.sim)

    def step(self,normalized):
        q=decode_action(normalized,self.sim.lower,self.sim.upper)
        valid=True; goals=0; count=0
        for _ in range(self.hold_controls):
            self.sim.step_target(q)
            self.controls+=1; count+=1
            m=self.sim.measure()
            valid=self.sim.physical_valid()
            goal=bool(valid and m['stable'])
            self.tail_controls=self.tail_controls+1 if goal else 0
            goals+=goal
            if not valid or self.controls>=300:
                break
        # Invalid physics is a terminal penalty, not a potential computation.
        phi=self.phi(m) if valid else self.previous_phi
        reward=(GAMMA*phi-self.previous_phi+.2*goals/count-.01) if valid else -5.
        self.previous_phi=phi; self.return_sum+=reward
        # NOT m['up_z'] < .45: a fallen policy must be allowed to recover.
        terminated=not valid
        done=terminated or self.controls>=300
        bootstrap=bootstrap_observation(observation(self.sim),terminated)
        info=None
        if done:
            info=dict(pose=self.pose,initial_hash=self.initial_hash,reset_seed=self.reset_seed,
                actual_fallen_start=self.pose!='standing',decision_steps=int(np.ceil(self.controls/self.hold_controls)),
                controls=self.controls,return_sum=float(self.return_sum),valid=valid,
                strict_tail_s=self.tail_controls*DT,
                training_success=bool(valid and self.tail_controls*DT>=1.),
                final=m,peaks=self.sim.peaks.copy())
            obs=self.reset()
        else:
            obs=bootstrap
        # Training success is only a 1s discovery label, NOT final 30s acceptance.
        return obs,float(reward),bool(done),bool(terminated),bootstrap,info,count


def pipe_worker(connection,scene,stand,number,seed):
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        envs=[FullFallEpisode(scene,stand,seed+i) for i in range(number)]
        connection.send(('ready',np.stack([observation(e.sim) for e in envs])))
        while True:
            operation,payload=connection.recv()
            if operation=='close':
                break
            if operation!='step':
                raise ValueError('Unknown worker operation')
            connection.send(('ok',[e.step(a) for e,a in zip(envs,payload)]))
    except Exception as exc:
        connection.send(('error',type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally:
        connection.close()
