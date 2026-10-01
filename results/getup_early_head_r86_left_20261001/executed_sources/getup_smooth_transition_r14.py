"""Test coordinated joint-target transitions, retaining physical motor limits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_independent_native import DT, digest
from diagnostics.getup_teacher_probe_r12 import normalized_target


def smooth_target(sim,tau):
    if not np.isfinite(tau) or tau<0: raise ValueError('invalid transition time constant')
    alpha=1. if tau==0 else -np.expm1(-DT/tau)
    return sim.prev+alpha*(sim.home-sim.prev)


_context=None


def init_worker(contract_path,tilt):
    global _context
    c=json.loads(Path(contract_path).read_text())
    _context=NativeEpisode(c['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                          c['starts_rad'],1,tilt)


def evaluate(job):
    tau,seeds,record=job
    e=_context; rows=[]; demos=[]
    for seed in seeds:
        e.rng=np.random.default_rng(seed);e.reset()
        initial=e.sim.measure(); observations=[]; labels=[]
        for step in range(200):
            obs=e.sim.observation(); action=normalized_target(e.sim,smooth_target(e.sim,tau))
            if record: observations.append(obs); labels.append(action)
            _,_,done,_,_,info=e.step(action)
            if done:
                rows.append(dict(tau_s=tau,seed=seed,initial_up_z=initial['up_z'],**info))
                if record and info['success']:
                    demos.append((seed,np.asarray(observations),np.asarray(labels)))
                break
    return dict(tau_s=tau,successes=sum(r['success'] for r in rows),runs=len(rows),results=rows),demos


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    contract=args.experiment/'controller_contract.json'
    taus=(0.,.06,.15,.3,.5,.8)
    seeds=list(range(130000,130020))
    # These seeds select the transition, not its final reported generalization.
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(contract),.55)) as pool:
        evaluated=list(pool.map(evaluate,[(tau,seeds,False) for tau in taus],chunksize=1))
        grid=[r for r,_ in evaluated]
        winner=max(grid,key=lambda r:(r['successes'],sum(x['steps'] for x in r['results']),
                                     sum(x['return_sum'] for x in r['results'])))
        validation=list(pool.map(evaluate,[(tau,list(range(140000,140020)),False)
                                          for tau in (0.,winner['tau_s'])]))
        collected=list(pool.map(evaluate,[(winner['tau_s'],[seed],True) for seed in range(150000,150064)],chunksize=1))
    demos=[d for _,group in collected for d in group]
    if demos:
        np.savez_compressed(args.output/'demonstrations.npz',obs=np.concatenate([d[1] for d in demos]).astype(np.float32),
                            actions=np.concatenate([d[2] for d in demos]).astype(np.float32),
                            episode_seed=np.concatenate([np.full(len(d[1]),d[0]) for d in demos]),
                            step=np.concatenate([np.arange(len(d[1])) for d in demos]))
    report=dict(grid=grid,selected_tau_s=winner['tau_s'],validation=[r for r,_ in validation],
                demonstration_results=[r for r,_ in collected],successful_demonstrations=len(demos),
                selection_seed_base=130000,validation_seed_base=140000,demonstration_seed_base=150000,
                stage='near-standing low-pose/tilt recovery; NOT full fallen get-up',
                physical_motor_parameters_unchanged=True,simulation_only=True,hardware_readiness=False,
                hashes={str(f):digest(f) for f in (Path(__file__),contract)})
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SMOOTH TRANSITION GRID:',json.dumps([{k:v for k,v in r.items() if k!='results'} for r in grid]),flush=True)
    print('SMOOTH TRANSITION VALIDATION:',json.dumps([{k:v for k,v in r.items() if k!='results'} for r,_ in validation]),flush=True)
    print('DEMONSTRATION EPISODES:',len(demos),flush=True)


if __name__=='__main__':main()
