"""Diagnostic only: quantify tiny head-command sensitivity of the frozen path.

R89 and R90 lose recoveries despite small imitation loss. Probe prescribed
subdegree biases, not larger action budget or relaxed success thresholds.
No candidate promotion and no independent test seeds are used in this audit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_head_distillation_r89 as student
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.train_getup_head_feedback_r87 import path
from diagnostics.getup_independent_native import digest
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_prefix_pose_r73 import success


def grid():
    rows=[{'name':'identity','bias':[0.,0.]}]
    for size in (.0005,.001,.002,.005):
        for joint in (0,1):
            for sign in (-1.,1.):
                value=[0.,0.];value[joint]=sign*size
                rows.append({'name':f'head{joint}_bias{sign*size}', 'bias':value})
    return rows


def constant_network(bias):
    bias=np.asarray(bias,dtype=float)
    if bias.shape!=(2,) or not np.isfinite(bias).all() or abs(bias).max()>.005+1e-12:
        raise ValueError('Need two bounded diagnostic head biases')
    return {'center':np.zeros(35),'scale':np.ones(35),'w1':np.zeros((35,1)),
            'b1':np.zeros(1),'w2':np.zeros((1,2)),'b2':bias}


def evaluate(job):
    seed,index,bias,output=job;sim,cases,ck,ids=base._CTX
    targets,phases,ref,gains=path(2.);source,peaks,initial,initial_hash=cases[seed]
    result,trace,_,_=student.run(sim,source,peaks,targets,phases,ids,ref,gains,
        network=constant_network(bias),strength=0. if not np.any(bias) else 1.,record=True)
    result.update(seed=seed,initial=initial,initial_state_sha256=initial_hash,
                  success=success(result,len(targets),1.),bias=bias)
    dest=Path(output)/f'grid_{index:02d}'/f'case_{seed}';dest.mkdir(exist_ok=False)
    save_trace(dest/'trajectory.npz',trace)
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Fresh output and bounded workers required')
    root=Path('/data/shijinsheng/open_duck');priorpath=root/'outputs/getup_centered_head_r90_left_20261001/results.json'
    prior=json.loads(priorpath.read_text())
    scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if prior[key]!=digest(file):raise RuntimeError('Frozen baseline input changed')
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,sources/file.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'method':'prescribed tiny head-feedback bias sensitivity audit','diagnostic_not_training':True,
        'hypothesis':'small supervised feedback errors may leave a narrow physical recovery basin',
        'no_midpath_state_reset':True,'workers':args.workers,'development_seeds':list(TRAIN),
        'no_independent_seeds':True,'promotion_allowed':False,'grid':grid(),
        'training_gate_seconds_not_qualification':1.,'activation_window_seconds':[.5,1.5,2.5,3.5],
        'combined_residual_cap_rad':student.CAP,
        'frozen':['leg_and_head_reference_targets','leg_feedback','timing','settling','collision','torque',
                  'joint_range','target_slew','strict_standing'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
        'scene_sha256':digest(scene),'stand_sha256':digest(stand),'r90_result_sha256':digest(priorpath),
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2));rows=[];base_success=None
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
            initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        for index,arm in enumerate(grid()):
            (args.output/f'grid_{index:02d}').mkdir()
            values=list(pool.map(evaluate,[(s,index,arm['bias'],str(args.output)) for s in (None,*TRAIN)],chunksize=1))
            current={v['seed'] for v in values[1:] if v['success']}
            if index==0:
                if not values[0]['success'] or len(current)!=13:raise RuntimeError('Zero-bias baseline failed reproduction')
                base_success=current
            row={**arm,'nominal':values[0],'cases':values[1:],'successes':len(current),
                 'valid_cases':sum(v['valid'] for v in values[1:]),
                 'rescues':sorted(current-base_success),'regressions':sorted(base_success-current)}
            rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2))
            print('SENSITIVITY',index,arm['bias'],'NOMINAL',int(values[0]['success']),
                  'SUCCESS',row['successes'],'/24','RESCUES',len(row['rescues']),
                  'REGRESSIONS',len(row['regressions']),flush=True)
    (args.output/'results.json').write_text(json.dumps({**contract,'rows':rows},indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
