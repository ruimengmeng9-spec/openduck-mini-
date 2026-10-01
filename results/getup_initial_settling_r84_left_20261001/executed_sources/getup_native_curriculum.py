"""Native recovery curriculum: control-rate limits and loaded-foot success.

Curriculum resets modify state only before an episode. This is not a claim
that mild tilted/low starts are equivalent to full fallen recovery.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.audit_getup_load_support import LoadSupportSim
from diagnostics.getup_independent_native import DT, SLEW, digest


class CurriculumSim(LoadSupportSim):
    def prepare_joint_start(self, target, seed=0, tilt=0., perturb=False, settle_steps=50):
        target=np.asarray(target,dtype=float)
        if target.shape!=self.home.shape or not np.isfinite(target).all():
            raise ValueError('bad curriculum start target')
        target=np.clip(target,self.lower,self.upper)
        m,d=self.model,self.data
        mujoco.mj_resetDataKeyframe(m,d,m.keyframe('home').id)
        rng=np.random.default_rng(seed)
        d.qpos[self.qadr]=target
        axis=rng.uniform(-1.,1.,2)
        axis/=max(np.linalg.norm(axis),1e-9)
        d.qpos[3:7]=[np.cos(tilt/2),*(np.sin(tilt/2)*np.r_[axis,0.])]
        d.qpos[2]=0.
        mujoco.mj_forward(m,d)
        d.qpos[2]+=.005-self.lowest_geom()
        d.qvel[:]=rng.uniform(-.01,.01,m.nv) if perturb else 0.
        self.prev=target.copy()
        self.history[:]=0.
        mujoco.mj_forward(m,d)
        for _ in range(settle_steps):
            self.step_target(target)
        return self.measure()

    def observation(self):
        feet,_,_=self.contacts()
        return np.concatenate([self.sensor('gyro'),self.sensor('upvector'),
                               self.data.qpos[self.qadr]-self.home,
                               self.data.qvel[self.vadr]*.05,self.prev-self.home,
                               np.asarray(feet,dtype=float)]).astype(np.float32)


def physical_safe(sim,m):
    q=sim.data.qpos[sim.qadr]
    return bool(m['finite'] and m['floor_penetration_m']<.01 and m['self_penetration_m']<.004
                and np.maximum(sim.lower-q,q-sim.upper).max()<.08)


def reward_terms(m,previous_height,home_error):
    height=float(m['height_m']); up=float(m['up_z'])
    # Goal composite is gated by actual loaded feet, not contact flags alone.
    upright=np.exp(-((np.arccos(np.clip(up,-1.,1.))/.4)**2))
    height_match=np.exp(-((height-.165)/.04)**2)
    pose=np.exp(-(home_error/.6)**2)
    goal=height_match*upright*pose*m['foot_load_fraction']
    progress=max(min(height-previous_height,.03),-.03)/DT
    # Upward progress is not rewarded once above the measured stand height.
    rise=progress if height<.17 and up>.6 else 0.
    return dict(goal=4.*goal,height_error=-8.*abs(height-.165),
                upright=.3*up,loaded_feet=.3*m['foot_load_fraction']*max(up,0.),
                rise=.15*rise,stable=1. if at_goal(m) else 0.)


def at_goal(m):
    return bool(m['stable'] and .155<m['height_m']<.185
                and m['joint_home_error_mean_rad']<.2 and m['joint_home_error_max_rad']<.45)


_envs=[]


def init_rollout_worker(scene,stand,starts,number,seed,tilt_max):
    global _envs
    _envs=[NativeEpisode(scene,stand,starts,seed+i,tilt_max) for i in range(number)]


class NativeEpisode:
    def __init__(self,scene,stand,starts,seed,tilt_max):
        self.sim=CurriculumSim(scene,stand)
        self.starts=np.asarray(starts)
        self.rng=np.random.default_rng(seed)
        self.tilt_max=tilt_max
        self.reset()

    def reset(self):
        self.steps=0; self.stable_tail=0; self.episode_return=0.; self.episode_safe=True
        # Rejection is initialization-only, explicitly counted and bounded.
        for attempt in range(20):
            target=self.starts[self.rng.integers(len(self.starts))]
            tilt=self.rng.uniform(0.,self.tilt_max)
            m=self.sim.prepare_joint_start(target,int(self.rng.integers(2**30)),tilt,True,settle_steps=5)
            if physical_safe(self.sim,m) and m['up_z']>.65:
                self.initial=m
                self.previous_height=m['height_m']
                self.reset_rejections=attempt
                return self.sim.observation()
        raise RuntimeError('curriculum reset could not produce safe near-standing state')

    def step(self,normalized):
        action=np.asarray(normalized,dtype=float)
        if action.shape!=(14,) or not np.isfinite(action).all():
            raise ValueError('invalid normalized recovery action')
        target=(self.sim.lower+self.sim.upper)/2+np.clip(action,-1.,1.)*(self.sim.upper-self.sim.lower)/2
        before=self.sim.prev.copy()
        applied=self.sim.step_target(target)
        if np.max(np.abs(applied-before))>SLEW*DT+1e-8:
            raise RuntimeError('slew contract violated')
        self.steps+=1
        m=self.sim.measure()
        safe=physical_safe(self.sim,m)
        self.episode_safe=self.episode_safe and safe
        terms=reward_terms(m,self.previous_height,m['joint_home_error_mean_rad'])
        self.previous_height=m['height_m']
        reward=sum(terms.values()) if safe else -20.
        self.stable_tail=self.stable_tail+1 if at_goal(m) else 0
        self.episode_return+=reward
        # A fall is failure in stage 1 only. Full fallen recovery is NOT trained here.
        terminated=(not safe) or m['up_z']<.45
        done=terminated or self.steps>=200
        bootstrap=self.sim.observation()
        info=None
        if done:
            info=dict(success=bool(self.stable_tail*DT>=2. and self.episode_safe),
                      return_sum=self.episode_return,steps=self.steps,initial_height_m=self.initial['height_m'],
                      final_height_m=m['height_m'],final_up_z=m['up_z'],foot_load_fraction=m['foot_load_fraction'],
                      safe=self.episode_safe,reset_rejections=self.reset_rejections)
            obs=self.reset()
        else:
            obs=bootstrap
        return obs,float(reward),bool(done),bool(terminated),bootstrap,info


def worker_reset():
    return np.stack([e.reset() for e in _envs])


def worker_step(actions):
    return [e.step(a) for e,a in zip(_envs,actions)]


def pipe_worker(connection,scene,stand,starts,number,seed,tilt_max):
    import os
    os.environ['JAX_PLATFORMS']='cpu'
    try:
        init_rollout_worker(scene,stand,starts,number,seed,tilt_max)
        connection.send(('ready',worker_reset()))
        while True:
            operation,payload=connection.recv()
            if operation=='close':
                break
            if operation=='step':
                connection.send(('ok',worker_step(payload)))
            else:
                raise ValueError('unknown native rollout operation')
    except Exception as exc:
        import traceback
        connection.send(('error',type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally:
        connection.close()


def qualify_job(job):
    scene,stand,target,index=job
    sim=CurriculumSim(scene,stand)
    initial=sim.prepare_joint_start(target)
    metrics=[initial]
    valid=physical_safe(sim,initial)
    for _ in range(100):
        sim.step_target(target)
        m=sim.measure(); metrics.append(m)
        valid=valid and physical_safe(sim,m)
    tail=metrics[-25:]
    qualified=valid and all(m['up_z']>.97 and m['foot_load_fraction']>.9
                            and min(m['foot_normal_forces_n'])>.1 for m in tail)
    return dict(index=index,target_rad=np.asarray(target).tolist(),qualified=bool(qualified),
                initial=initial,final=metrics[-1],safe=bool(valid))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim=CurriculumSim(scene,stand)
    targets=[]
    for hip in (-1.1,-.9,-.7,-.5):
        for knee in (1.45,1.55):
            for neck in (-.15,0.,.4):
                q=sim.home.copy(); q[[2,11]]=[hip,-hip]; q[[3,12]]=knee
                q[[4,13]]=np.clip(-hip-knee,sim.lower[[4,13]],sim.upper[[4,13]])
                q[5]=neck; targets.append(q)
    jobs=[(str(scene),str(stand),q,i) for i,q in enumerate(targets)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(qualify_job,jobs,chunksize=1))
    qualified=[r for r in rows if r['qualified'] and .12<r['final']['height_m']<.16]
    qualified.sort(key=lambda r:r['final']['height_m'])
    summary=dict(rows=rows,qualified_low_starts=qualified,scene_path=str(scene),
                 simulation_only=True,hardware_readiness=False,
                 hashes={str(f):digest(f) for f in (Path(__file__),scene,stand)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('QUALIFIED LOW STARTS:',len(qualified),flush=True)
    for r in qualified[:10]:
        print(json.dumps(dict(index=r['index'],height=r['final']['height_m'],up=r['final']['up_z'],
                              foot_load=r['final']['foot_load_fraction'])),flush=True)


if __name__=='__main__':
    main()
