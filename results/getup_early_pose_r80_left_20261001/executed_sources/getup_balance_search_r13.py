"""Search a small IMU feedback teacher before training a recovery network.

Original native collision, motors and success gates are retained. This is a
near-standing recovery task, not fallen get-up. Global up components are used
only for this no-yaw-command stage; no general heading invariance is claimed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_teacher_probe_r12 import normalized_target
from diagnostics.getup_independent_native import digest


def home_axis_map(sim):
    scratch=mujoco.MjData(sim.model)
    mujoco.mj_resetDataKeyframe(sim.model,scratch,sim.model.keyframe('home').id)
    mujoco.mj_forward(sim.model,scratch)
    axes=scratch.xaxis[sim.joints]
    return dict(roll=axes[[1,10],0].copy(),hip_pitch=axes[[2,11],1].copy(),
                ankle_pitch=axes[[4,13],1].copy())


def feedback_target(sim,gains,axes):
    gains=np.asarray(gains,dtype=float)
    if gains.shape!=(6,) or not np.isfinite(gains).all(): raise ValueError('invalid feedback gains')
    up=sim.sensor('upvector'); gyro=sim.sensor('gyro')
    roll=-up[1]; pitch=up[0]
    delta=np.zeros(14)
    delta[[1,10]]=axes['roll']*np.clip(gains[0]*roll+gains[1]*gyro[0],-.35,.35)
    delta[[2,11]]=axes['hip_pitch']*np.clip(gains[2]*pitch+gains[3]*gyro[1],-.5,.5)
    delta[[4,13]]=axes['ankle_pitch']*np.clip(gains[4]*pitch+gains[5]*gyro[1],-.5,.5)
    return np.clip(sim.home+delta,sim.lower,sim.upper)


_context=None


def init_worker(contract_path,tilt):
    global _context
    contract=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(contract['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    contract['starts_rad'],1,tilt)
    _context=(e,home_axis_map(e.sim))


def run_one(e,axes,gains,seed,record=False):
    e.rng=np.random.default_rng(seed); e.reset()
    obs=[]; actions=[]; times=[]; qpos=[]; qvel=[]; control=[]; trace=[]
    initial=e.sim.measure()
    for step in range(200):
        observation=e.sim.observation()
        target=feedback_target(e.sim,gains,axes)
        action=normalized_target(e.sim,target)
        if record:
            obs.append(observation); actions.append(action)
            times.append(float(e.sim.data.time)); qpos.append(e.sim.data.qpos.copy())
            qvel.append(e.sim.data.qvel.copy());control.append(target.copy())
            m=e.sim.measure()
            trace.append(dict(up_z=m['up_z'],height_m=m['height_m'],gyro=e.sim.sensor('gyro').tolist(),
                foot_load_fraction=m['foot_load_fraction'],com_support_margin_m=m['com_support_margin_m']))
        _,_,done,_,_,info=e.step(action)
        if done:
            return dict(seed=seed,initial_up_z=initial['up_z'],initial_angular_speed=initial['angular_speed_rad_s'],
                        initial_foot_load_fraction=initial['foot_load_fraction'],**info),dict(obs=obs,actions=actions,
                        time=times,qpos=qpos,qvel=qvel,target=control,trace=trace)
    raise RuntimeError('episode contract failed')


def evaluate(job):
    gains,seeds=job
    e,axes=_context
    rows=[run_one(e,axes,gains,seed)[0] for seed in seeds]
    successes=sum(r['success'] for r in rows)
    score=float(np.mean([r['return_sum']+1200*r['success']+r['steps'] for r in rows]))
    return dict(score=score,successes=successes,runs=len(rows),results=rows,
                environment_steps=sum(r['steps'] for r in rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generations',type=int,default=12)
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--tilt-max',type=float,default=.55)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    contract_path=args.experiment/'controller_contract.json'
    contract=json.loads(contract_path.read_text())
    lower=np.array([-2.,-.5,-2.,-.5,-2.,-.5]); upper=-lower
    mean=np.zeros(6); std=np.array([.8,.2,.8,.2,.8,.2])
    rng=np.random.default_rng(413)
    best=mean.copy(); best_score=-np.inf; history=[]; environment_steps=0; started=time.time()
    train_seeds=list(range(100000,100000+args.seeds))
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,
                             initargs=(str(contract_path),args.tilt_max)) as pool:
        baseline=list(pool.map(evaluate,[(np.zeros(6),train_seeds)]))[0]
        for generation in range(args.generations):
            candidates=np.clip(rng.normal(mean,std,(args.population,6)),lower,upper)
            candidates[0]=best; candidates[1]=0.
            results=list(pool.map(evaluate,[(g,train_seeds) for g in candidates],chunksize=1))
            scores=np.array([r['score'] for r in results])
            elites=candidates[np.argsort(scores)[-max(4,args.population//6):]]
            mean=.3*mean+.7*elites.mean(axis=0)
            std=np.maximum(.3*std+.7*elites.std(axis=0),np.array([.05,.012,.05,.012,.05,.012]))
            index=int(scores.argmax())
            if scores[index]>best_score:
                best=candidates[index].copy(); best_score=float(scores[index])
            environment_steps+=sum(r['environment_steps'] for r in results)
            row=dict(generation=generation+1,best_score=best_score,best_gains=best.tolist(),
                     successful_runs_this_generation=sum(r['successes'] for r in results),
                     best_this_generation=results[index],environment_steps=environment_steps,wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'search_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            (args.output/'gains.json').write_text(json.dumps(dict(gains=best.tolist(),tilt_max_rad=args.tilt_max,
                 contract_path=str(contract_path),training_seed_base=100000,simulation_only=True,hardware_readiness=False)),encoding='utf-8')
            print('FEEDBACK SEARCH:',json.dumps({k:v for k,v in row.items() if k!='best_this_generation'}),flush=True)
        validation_seeds=list(range(120000,120020))
        validation=list(pool.map(evaluate,[(np.zeros(6),validation_seeds),(best,validation_seeds)]))
    init_worker(str(contract_path),args.tilt_max)
    e,axes=_context
    demonstrations=[]; demo_rows=[]
    for seed in range(110000,110064):
        row,arrays=run_one(e,axes,best,seed,record=True); demo_rows.append(row)
        if row['success']: demonstrations.append((row,arrays))
    if demonstrations:
        np.savez_compressed(args.output/'demonstrations.npz',
            obs=np.concatenate([a['obs'] for _,a in demonstrations]).astype(np.float32),
            actions=np.concatenate([a['actions'] for _,a in demonstrations]).astype(np.float32),
            episode_seed=np.concatenate([np.full(len(a['obs']),r['seed']) for r,a in demonstrations]),
            step=np.concatenate([np.arange(len(a['obs'])) for _,a in demonstrations]))
    _,example=run_one(e,axes,best,120000,record=True)
    np.savez_compressed(args.output/'validation_example.npz',time=example['time'],qpos=example['qpos'],
                        qvel=example['qvel'],target=example['target'])
    report=dict(baseline_training=baseline,validation=dict(home=validation[0],feedback=validation[1]),
                gains=best.tolist(),axis_map={k:v.tolist() for k,v in axes.items()},environment_steps=environment_steps,
                demonstration_results=demo_rows,successful_demonstrations=len(demonstrations),
                stage='near-standing tilt recovery, no yaw command; NOT fallen get-up',simulation_only=True,hardware_readiness=False,
                hashes={str(f):digest(f) for f in (Path(__file__),contract_path,Path(contract['scene_path']))})
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FEEDBACK VALIDATION:',json.dumps({k:dict(successes=r['successes'],runs=r['runs']) for k,r in report['validation'].items()}),flush=True)
    print('SUCCESSFUL DEMONSTRATIONS:',len(demonstrations),flush=True)


if __name__=='__main__': main()
