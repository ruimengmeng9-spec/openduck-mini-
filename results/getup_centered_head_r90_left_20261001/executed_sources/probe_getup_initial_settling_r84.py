"""R84: physical pre-motion settling, not changing initial state by assignment.

Repeat the original home command before recovery and allow actual simulation
to dissipate residual motion. Never restore at the end of the dwell. Original
reference geometry, get-up timing, gains, physics and acceptance stay fixed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np

from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT,digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.search_getup_reference_feedback_r64 import features,rollout
from diagnostics.train_getup_prefix_pose_r73 import success

HELDOUT=tuple(range(784000,784040))
GRID=(0,10,25,50,100,200,400)


def prepend(targets,phases,home,controls):
    if not isinstance(controls,(int,np.integer)) or controls not in GRID:
        raise ValueError('Unsupported physical settling duration')
    if controls==0: return targets.copy(),phases.copy()
    return (np.concatenate([np.tile(home,(controls,1)),targets]),
            np.concatenate([np.full(controls,8,dtype=phases.dtype),phases]))


def audited_reference(sim,case,targets,dest):
    sim.restore(case[0]);sim.peaks,sim.finite=case[1].copy(),True
    reference=[];trace=[]
    for target in targets:
        reference.append(features(sim))
        sim.step_target(target)
        trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy()))
        if not sim.physical_valid():
            np.savez_compressed(dest/'rejected_nominal_reference.npz',
                                time=[x[0] for x in trace],qpos=[x[1] for x in trace],
                                qvel=[x[2] for x in trace])
            return None,sim.peaks.copy()
    return np.asarray(reference),sim.peaks.copy()


def evaluate(job):
    controls,directory,hold,required=job
    sim,cases,ck,ids=base._CTX
    dest=Path(directory);dest.mkdir(exist_ok=False)
    original,phases=timed_targets(sim,ck,(.4,.6,.8),hold)
    targets,phases=prepend(original,phases,sim.home,controls)
    gains=np.concatenate([ck['prefix_gains'],ck['terminal_gains'],np.zeros((1,8))])
    ref,peaks=audited_reference(sim,cases[None],targets,dest)
    result={'settling_controls':controls,'settling_seconds':controls*DT,'nominal_success':False,
            'successes':0,'cases':[],'target_steps':len(targets),'nominal_reference_peaks':peaks}
    if ref is None:
        result['rejected']='Nominal reference failed unchanged physical audit'
    else:
        for seed,case in cases.items():
            value,trace=rollout(sim,*case[:2],True,targets,phases,ids,ref,gains,True)
            row={'seed':seed,'initial_fallen':True,'initial':case[2],
                 'initial_state_sha256':case[3],
                 'success':success(value,len(targets),required),**value}
            if controls and len(trace)>=controls:
                # Saved data are descriptive only, never restored or used as new starts.
                row['after_settling_root_linear_speed_mps']=float(np.linalg.norm(trace[controls-1][2][:3]))
                row['after_settling_root_angular_speed_rad_s']=float(np.linalg.norm(trace[controls-1][2][3:6]))
            save_trace(dest/f'case_{seed}.npz',trace)
            if seed is None:
                result['nominal']=row;result['nominal_success']=row['success']
                if not row['success']:break
            else:
                result['cases'].append(row);result['successes']+=int(row['success'])
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Need new output and bounded workers')
    root=Path('/data/shijinsheng/open_duck')
    scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    prior=root/'outputs/getup_early_feedback_r81_left_20261001/results.json'
    r81=json.loads(prior.read_text())
    for key,path in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if r81[key]!=digest(path):raise RuntimeError('Frozen inputs changed')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();executed=args.output/'executed_sources';executed.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,executed/path.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
              'method':'physical home-command settling before complete recovery',
              'hypothesis':'residual fallen-state motion or contact relaxation sensitizes the early path',
              'grid_controls':list(GRID),'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
              'no_midpath_state_reset':True,'initial_settling_has_zero_feedback':True,
              'recovery_reference_regenerated_by_physical_nominal_rollout':True,
              'training_gate_seconds':1.,'required_strict_seconds':30.,'qualification_hold_seconds':35.,
              'qualification_only_after_training_improvement':True,
              'frozen':['original_recovery_commands','recovery_phase_durations','existing_feedback_gains',
                        'collision','torque','joint_limits','target_slew','strict_gate'],
              'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
              'scene_sha256':digest(scene),'stand_sha256':digest(stand),'r81_result_sha256':digest(prior),
              'executed_source_hashes':{f.name:digest(f) for f in executed.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        for row in pool.map(evaluate,[(n,str(args.output/f'dwell_{n:03d}'),2.,1.) for n in GRID],chunksize=1):
            rows.append(row);(args.output/'grid_progress.json').write_text(json.dumps(rows,indent=2))
            print('SETTLING',row['settling_seconds'],'NOMINAL',row['nominal_success'],
                  'SUCCESS',row['successes'],'/24',flush=True)
    baseline=rows[0]
    if not baseline['nominal_success'] or baseline['successes']!=r81['training_best']['successes']:
        raise RuntimeError('Zero added settling must reproduce baseline')
    best=max([r for r in rows if r['nominal_success']],key=lambda r:(r['successes'],-r['settling_controls']))
    report={**contract,'training_grid':rows,'selected':best,'qualification_run':False}
    np.savez_compressed(args.output/'selected.npz',**ck,r84_settling_controls=best['settling_controls'])
    if best['successes']>baseline['successes']:
        report['qualification_run']=True
        with ProcessPoolExecutor(max_workers=2,initializer=base.init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT)) as pool:
            q=list(pool.map(evaluate,[(0,str(args.output/'qualification_baseline'),35.,30.),
                                    (best['settling_controls'],str(args.output/'qualification_candidate'),35.,30.)]))
        report['qualification']=q;a,b=q
        report['heldout_trials_executed']=len(b['cases'])
        report['baseline_successes']=a['successes'];report['heldout_successes']=b['successes']
        report['paired_rescues']=[y['seed'] for x,y in zip(a['cases'],b['cases']) if y['success'] and not x['success']]
        report['paired_regressions']=[y['seed'] for x,y in zip(a['cases'],b['cases']) if x['success'] and not y['success']]
        print('QUALIFIED',b['successes'],'/',len(b['cases']),'BASELINE',a['successes'],flush=True)
    else:
        report['qualification_skipped_reason']='No training improvement; fresh seeds not simulated'
        print('NO IMPROVEMENT; heldout seeds unused',flush=True)
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
