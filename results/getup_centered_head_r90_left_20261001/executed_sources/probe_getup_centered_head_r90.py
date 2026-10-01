"""Frozen R89 networks with nominal-reference output centering only.

No new fit, no state injection. Sensor reference is recorded in a separate
unmodified physical nominal rollout. Subtract its network output at the
current control time so nominal imitation bias cannot drive head motion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_head_distillation_r89 as student
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.train_getup_head_feedback_r87 import path
from diagnostics.getup_independent_native import DT,digest
from diagnostics.search_getup_reference_feedback_r64 import features,residual,rollout
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_prefix_pose_r73 import success

HELDOUT=tuple(range(790000,790040))
_CACHE={}
_OUT=None
_PREDICT=student.predict


def centered_prediction(net,x,nominal_output):
    step=int(round(float(x[-1])*3.5/DT))
    if not 0<=step<len(nominal_output):raise ValueError('Current control time outside reference')
    return np.clip(_PREDICT(net,x)-nominal_output[step],-student.RAW_BOUND,student.RAW_BOUND)


def run_centered(sim,source,peaks,targets,phases,ids,ref,gains,net,nominal_output,strength):
    original=student.predict
    student.predict=lambda network,x:centered_prediction(network,x,nominal_output)
    try:
        return student.run(sim,source,peaks,targets,phases,ids,ref,gains,
                           network=net,strength=strength,record=True)
    finally:student.predict=original  # No permanent module or original-source change.


def init_worker(scene,stand,ck,seeds,out):
    global _OUT,_CACHE
    base.init_worker(scene,stand,ck,seeds);_OUT=Path(out);_CACHE={}


def reference(hold):
    if hold in _CACHE:return _CACHE[hold]
    sim,cases,ck,ids=base._CTX
    targets,phases,ref,gains=path(hold)
    source,peaks=cases[None][:2];sim.restore(source);sim.peaks,sim.finite=peaks.copy(),True
    sensor=[]
    for i,(command,phase) in enumerate(zip(targets,phases)):
        error=features(sim)-ref[i];sensor.append(student.inputs(sim,error,i*DT,ids))
        target=command.copy()
        target[ids]=np.clip(target[ids]+residual(error,gains[phase]),sim.lower[ids],sim.upper[ids])
        sim.step_target(target)
        if not sim.physical_valid():raise RuntimeError('Nominal centering reference physical audit failed')
    sensor=np.stack(sensor)
    file=_OUT/'nominal_references'/f'worker_{os.getpid()}_hold{hold}.npz'
    np.savez_compressed(file,precontrol_observations=sensor)
    _CACHE[hold]=targets,phases,ref,gains,sensor
    return _CACHE[hold]


def evaluate(job):
    seed,file,strength,hold,required,directory=job;sim,cases,ck,ids=base._CTX
    with np.load(file,allow_pickle=False) as z:net={k:z[k] for k in z.files}
    targets,phases,ref,gains,sensor=reference(hold)
    nominal_output=np.stack([_PREDICT(net,x) for x in sensor])
    source,peaks,initial,initial_hash=cases[seed]
    old,oldtrace=rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,True)
    value,trace,_,_=run_centered(sim,source,peaks,targets,phases,ids,ref,gains,net,nominal_output,strength)
    old['success']=success(old,len(targets),required);value['success']=success(value,len(targets),required)
    same_nominal=None
    if seed is None:
        same_nominal=bool(len(trace)==len(oldtrace) and
            all(np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2]) and np.array_equal(a[7],b[7])
                for a,b in zip(trace,oldtrace)))
        if not same_nominal:raise RuntimeError('Centering does not exactly preserve nominal trajectory')
    row={'seed':seed,'initial':initial,'initial_state_sha256':initial_hash,
         'baseline':old,'candidate':value,'nominal_exact_identity':same_nominal,
         'max_nominal_network_bias_rad':float(abs(nominal_output).max())}
    dest=Path(directory)/f'case_{seed}';dest.mkdir(exist_ok=False)
    save_trace(dest/'baseline.npz',oldtrace);save_trace(dest/'candidate.npz',trace)
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Fresh output and bounded workers required')
    root=Path('/data/shijinsheng/open_duck');r89=root/'outputs/getup_head_distillation_r89_left_20261001'
    prior=json.loads((r89/'results.json').read_text())
    scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if prior[key]!=digest(file):raise RuntimeError('Frozen physical inputs changed')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();(args.output/'nominal_references').mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'method':'frozen R89 network nominal-output centering control',
        'hypothesis':'nonzero nominal imitation bias may destabilize the unchanged successful path',
        'no_new_fit':True,'no_midpath_state_reset':True,'training_seeds':list(TRAIN),
        'heldout_seeds':list(HELDOUT),'nominal_exact_identity_required':True,
        'selection_uses_development_results_only':True,'workers':args.workers,
        'required_strict_seconds':30.,'hold_seconds':35.,'training_gate_seconds':1.,
        'frozen':['R89_weights','R89_normalization','strength_grid','activation_envelope','residual_cap',
                  'reference_commands','leg_feedback','timing','collision','torque','joint_range','slew','strict_gate'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
        'scene_sha256':digest(scene),'stand_sha256':digest(stand),'r89_result_sha256':digest(r89/'results.json'),
        'network_hashes':{f.name:digest(f) for f in r89.glob('network_epoch*.npz')},
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    screens=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
            initargs=(str(scene),str(stand),ck,TRAIN,str(args.output))) as pool:
        for file in sorted(r89.glob('network_epoch*.npz')):
            for strength in (.25,.5,1.):
                dest=args.output/f'{file.stem}_strength{strength}';dest.mkdir()
                nominal=list(pool.map(evaluate,[(None,str(file),strength,2.,1.,str(dest))]))[0]
                rows=list(pool.map(evaluate,[(s,str(file),strength,2.,1.,str(dest)) for s in TRAIN],chunksize=1))
                baseline_count=sum(r['baseline']['success'] for r in rows)
                if baseline_count!=13 or not nominal['baseline']['success']:
                    raise RuntimeError('Frozen baseline reproduction failed')
                row={'network':file.name,'strength':strength,'nominal':nominal,'cases':rows,
                     'successes':sum(r['candidate']['success'] for r in rows),'baseline_successes':baseline_count}
                screens.append(row);(args.output/'training_screen.json').write_text(json.dumps(screens,indent=2))
                print('CENTERED',file.name,strength,'SUCCESS',row['successes'],'/24 NOMINAL_EXACT',
                      int(nominal['nominal_exact_identity']),flush=True)
    best=max(screens,key=lambda r:(r['successes'],-r['strength']))
    report={**contract,'training_best':best,'training_screen':screens,'qualification_run':False}
    if best['successes']>13:
        report['qualification_run']=True;qa=args.output/'qualification';qa.mkdir()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                initargs=(str(scene),str(stand),ck,HELDOUT,str(args.output))) as pool:
            rows=[]
            for row in pool.map(evaluate,[(s,str(r89/best['network']),best['strength'],35.,30.,str(qa))
                                         for s in (None,*HELDOUT)],chunksize=1):
                rows.append(row);(args.output/'qualification_progress.json').write_text(json.dumps(rows,indent=2))
                print('HELDOUT',row['seed'],'CANDIDATE',int(row['candidate']['success']),
                      'BASE',int(row['baseline']['success']),flush=True)
        rs=rows[1:];report.update(canonical=rows[0],heldout=rs,heldout_trials=len(rs),
            heldout_successes=sum(r['candidate']['success'] for r in rs),
            heldout_baseline_successes=sum(r['baseline']['success'] for r in rs),
            paired_rescues=[r['seed'] for r in rs if r['candidate']['success'] and not r['baseline']['success']],
            paired_regressions=[r['seed'] for r in rs if r['baseline']['success'] and not r['candidate']['success']])
    else:report['qualification_skipped_reason']='No development count improvement; fresh seeds unused'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
