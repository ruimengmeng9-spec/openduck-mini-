"""Compare the existing stand feedback with stage-1 recovery checkpoints.

No changes to physics, initialization or the success gate. Dataset generation
uses separate training seeds, and does not label a fallen state as recovered.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode, physical_safe, at_goal
from diagnostics.getup_independent_native import digest


def normalized_target(sim,target):
    return np.clip((target-(sim.lower+sim.upper)/2)/((sim.upper-sim.lower)/2),-1.,1.).astype(np.float32)


def teacher_target(sim):
    return np.clip(sim.home+.25*sim.stand_action(),sim.lower,sim.upper)


def run_case(job):
    directory,kind,seed,tilt,collect=job
    directory=Path(directory)
    contract=json.loads((directory/'controller_contract.json').read_text())
    episode=NativeEpisode(contract['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                          contract['starts_rad'],seed,tilt)
    sim=episode.sim
    if kind=='student':
        import onnxruntime as ort
        options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
        actor=ort.InferenceSession(str(directory/'final.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    initial=sim.measure(); trace=[]; observations=[]; labels=[]; targets=[]
    for step in range(200):
        obs=sim.observation()
        if kind=='official_stand':
            target=teacher_target(sim)
            action=normalized_target(sim,target)
        elif kind=='home':
            target=sim.home
            action=normalized_target(sim,target)
        elif kind=='student':
            action=actor.run(None,{'obs':obs[None]})[0][0]
            target=(sim.lower+sim.upper)/2+action*(sim.upper-sim.lower)/2
        else: raise ValueError(kind)
        if collect:
            observations.append(obs); labels.append(action); targets.append(target.copy())
        # Measure before step because episode.step resets after termination.
        before=sim.measure()
        if step<40 or step%10==0:
            trace.append(dict(step=step,up_z=before['up_z'],height_m=before['height_m'],
                gyro=sim.sensor('gyro').tolist(),foot_load_fraction=before['foot_load_fraction'],
                foot_forces=before['foot_normal_forces_n'],com_support_margin_m=before['com_support_margin_m'],
                joint_home_error_mean_rad=before['joint_home_error_mean_rad']))
        _,_,done,_,_,info=episode.step(action)
        if done:
            row=dict(kind=kind,seed=seed,tilt_max_rad=tilt,initial=initial,trace=trace,**info)
            arrays=(np.asarray(observations,np.float32),np.asarray(labels,np.float32),np.asarray(targets,np.float32))
            return row,arrays
    raise RuntimeError('episode length contract failed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-base',type=int,default=98000)
    p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--tilt-max',type=float,default=.55)
    p.add_argument('--collect',action='store_true')
    args=p.parse_args(); args.output.mkdir(exist_ok=False,parents=True)
    kinds=('official_stand',) if args.collect else ('home','official_stand','student')
    jobs=[(str(args.experiment),kind,args.seed_base+s,args.tilt_max,args.collect) for kind in kinds for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(run_case,jobs,chunksize=1))
    rows=[r for r,_ in results]
    summary={kind:dict(successes=sum(r['success'] for r in rows if r['kind']==kind),runs=args.seeds,
                        mean_steps=float(np.mean([r['steps'] for r in rows if r['kind']==kind]))) for kind in kinds}
    report=dict(results=rows,summary=summary,seed_base=args.seed_base,tilt_max_rad=args.tilt_max,
                simulation_only=True,hardware_readiness=False,stage='near-standing recovery only',
                hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'final.onnx',args.experiment/'controller_contract.json')})
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.collect:
        # Failed demonstrations are retained in the report but not distilled as
        # successful recovery. Keep early states as well as the stable tail.
        selected=[(r,a) for r,a in results if r['success']]
        if not selected: raise RuntimeError('no successful teacher demonstrations')
        np.savez_compressed(args.output/'demonstrations.npz',
            obs=np.concatenate([a[0] for _,a in selected]),
            actions=np.concatenate([a[1] for _,a in selected]),
            targets=np.concatenate([a[2] for _,a in selected]),
            episode_seed=np.concatenate([np.full(len(a[0]),r['seed']) for r,a in selected]),
            step=np.concatenate([np.arange(len(a[0])) for _,a in selected]))
        print('DEMONSTRATION EPISODES:',len(selected),flush=True)
    print('TEACHER PROBE:',json.dumps(summary),flush=True)


if __name__=='__main__': main()
