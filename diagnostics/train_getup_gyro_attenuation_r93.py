"""R93 controlled early gyro-feedback attenuation and command-noise robustness.

R92 observes head-ground contact changes BEFORE feedback differences. This
tests subsequent amplification, not a claim that gyro feedback caused the
first contact change. No collision, reference target or motor-budget edits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT,digest
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_head_feedback_r87 import path,CAP
from diagnostics.train_getup_early_feedback_r81 import early_envelope
from diagnostics.search_getup_reference_feedback_r64 import features,residual,rollout
from diagnostics.search_getup_support_margin_r49 import foot_support,support_quality,supported_entry
from diagnostics.train_getup_prefix_pose_r73 import success

HELDOUT=tuple(range(793000,793040))
BIAS=((0.,0.),(-.0005,0.),(.0005,0.),(0.,-.0005),(0.,.0005))


def modified_gains(gains,pitch_scale,roll_scale,elapsed):
    if not (0<=pitch_scale<=1 and 0<=roll_scale<=1):raise ValueError('Attenuation factors must be in [0,1]')
    value=np.array(gains,copy=True);weight=early_envelope(elapsed)
    value[3:6]*=1-weight*(1-pitch_scale)
    value[7]*=1-weight*(1-roll_scale)
    return value


def run(sim,source,peaks,targets,phases,ids,ref,gains,scales,bias,record):
    if tuple(scales)==(1.,1.) and not np.any(bias):
        return rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,record)
    sim.restore(source);sim.peaks,sim.finite=peaks.copy(),True
    best=-np.inf;tail=longest=run_length=0;trace=[]
    for i,(command,phase) in enumerate(zip(targets,phases)):
        error=features(sim)-ref[i];weight=early_envelope(i*DT)
        offset=residual(error,modified_gains(gains[phase],*scales,i*DT))
        offset[8:10]+=weight*np.asarray(bias);offset=np.clip(offset,-CAP,CAP)
        target=command.copy();target[ids]=np.clip(target[ids]+offset,sim.lower[ids],sim.upper[ids])
        sim.step_target(target);valid=sim.physical_valid();m=sim.measure();s=foot_support(sim)
        best=max(best,support_quality(m,s));run_length=run_length+1 if supported_entry(m,s) else 0
        longest=max(longest,run_length);tail=tail+1 if m['stable'] else 0
        if record:trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),int(phase),
                               int(m['stable']),s['margin_m'],error.copy(),offset.copy()))
        if not valid:break
    score=-100. if not valid else best+10*min(longest*DT,1.)+20*min(tail*DT,1.)
    return {'score':float(score),'valid':bool(valid),'strict_tail_s':tail*DT,
            'supported_longest_s':longest*DT,'completed_steps':i+1,'final':m,
            'final_support':s,'peaks':sim.peaks.copy()},trace


def evaluate(job):
    seed,scales,bias,hold,required,directory=job;sim,cases,ck,ids=base._CTX
    targets,phases,ref,gains=path(hold);source,peaks,initial,initial_hash=cases[seed]
    value,trace=run(sim,source,peaks,targets,phases,ids,ref,gains,scales,bias,True)
    value.update(seed=seed,initial=initial,initial_state_sha256=initial_hash,
                 success=success(value,len(targets),required),scales=list(scales),bias=list(bias))
    dest=Path(directory)/f'case_{seed}';dest.mkdir(exist_ok=False)
    save_trace(dest/'trajectory.npz',trace);(dest/'summary.json').write_text(json.dumps(value,indent=2))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Fresh output and bounded workers required')
    root=Path('/data/shijinsheng/open_duck');prior=root/'outputs/getup_micro_contact_r92_20261001/results.json'
    r92=json.loads(prior.read_text())
    scene=root/'training/getup_decomposed_r4/model/scene.xml';stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if r92[key]!=digest(file):raise RuntimeError('Frozen R92 inputs changed')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    grid=list(itertools.product((1.,.75,.5,.25,0.),repeat=2))
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'method':'bounded early pitch and roll gyro-feedback attenuation grid fitting',
        'hypothesis':'legacy gyro response may amplify an already altered head-ground contact transition',
        'not_a_proven_root_cause':True,'no_midpath_state_reset':True,
        'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),'workers':args.workers,
        'grid':grid,'attenuated_gain_columns':{'pitch':[3,4,5],'roll':[7]},
        'window_seconds':[.5,1.5,2.5,3.5],'robustness_head_bias_grid_rad':BIAS,
        'robustness_uses_development_seeds_not_fresh_validation':True,
        'required_strict_seconds':30.,'hold_seconds':35.,'training_gate_seconds':1.,
        'promotion_requires_unbiased_count_improvement_and_no_worse_noise_mean_and_worst_count':True,
        'frozen':['all_reference_targets','tilt_feedback','gains_outside_window','phase_timing','initial_settling',
                  'collision','torque','joint_range','target_slew','residual_cap','strict_gate'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
        'scene_sha256':digest(scene),'stand_sha256':digest(stand),'r92_result_sha256':digest(prior),
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    grids=[];robust=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
            initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        for index,scales in enumerate(grid):
            dest=args.output/f'grid_{index:02d}';dest.mkdir()
            nominal=list(pool.map(evaluate,[(None,scales,BIAS[0],2.,1.,str(dest))]))[0]
            rows=[]
            if nominal['success']:
                rows=list(pool.map(evaluate,[(s,scales,BIAS[0],2.,1.,str(dest)) for s in TRAIN],chunksize=1))
            row={'grid_index':index,'scales':scales,'nominal':nominal,'nominal_success':nominal['success'],
                 'successes':sum(r['success'] for r in rows),'cases':rows}
            if index==0 and (not nominal['success'] or row['successes']!=13):
                raise RuntimeError('Identity must reproduce 13/24')
            grids.append(row);(args.output/'grid_progress.json').write_text(json.dumps(grids,indent=2))
            print('GRID',index,scales,'NOMINAL',int(nominal['success']),'SUCCESS',row['successes'],'/24',flush=True)
        improved=sorted([g for g in grids[1:] if g['nominal_success'] and g['successes']>13],
                        key=lambda g:g['successes'],reverse=True)[:3]
        # Noise screen is only needed for actual unbiased improvements.
        if improved:
            for setting in [grids[0],*improved]:
                dest=args.output/f'robust_grid_{setting["grid_index"]:02d}';dest.mkdir();noise=[]
                for arm,bias in enumerate(BIAS):
                    sub=dest/f'bias_{arm}';sub.mkdir()
                    rows=list(pool.map(evaluate,[(s,setting['scales'],bias,2.,1.,str(sub))
                                                for s in (None,*TRAIN)],chunksize=1))
                    noise.append({'bias':bias,'nominal':rows[0],'cases':rows[1:],
                                  'successes':sum(r['success'] for r in rows[1:])})
                    print('ROBUST',setting['grid_index'],arm,noise[-1]['successes'],'/24',flush=True)
                row={'setting':setting,'noise':noise,'mean_count':float(np.mean([x['successes'] for x in noise])),
                     'worst_count':min(x['successes'] for x in noise)}
                robust.append(row);(args.output/'robust_progress.json').write_text(json.dumps(robust,indent=2))
    eligible=[]
    if robust:
        eligible=[r for r in robust[1:] if r['mean_count']>=robust[0]['mean_count'] and
                  r['worst_count']>=robust[0]['worst_count']]
    chosen=max(eligible,key=lambda r:(r['worst_count'],r['mean_count'],r['setting']['successes'])) if eligible else None
    selected=chosen['setting'] if chosen else grids[0]
    np.savez_compressed(args.output/'checkpoint.npz',**ck,r93_scales=selected['scales'],
                        r93_grid_completed=len(grids),r93_promoted=chosen is not None)
    report={**contract,'grid_results':grids,'robustness_results':robust,'training_best':selected,
            'qualification_run':False}
    if chosen is not None:
        report['qualification_run']=True;qa=args.output/'qualification';qa.mkdir();paired=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                initargs=(str(scene),str(stand),ck,HELDOUT)) as pool:
            for name,scales in [('baseline',(1.,1.)),('candidate',selected['scales'])]:
                d=qa/name;d.mkdir()
                rows=list(pool.map(evaluate,[(s,scales,BIAS[0],35.,30.,str(d))
                                            for s in (None,*HELDOUT)],chunksize=1))
                paired.append(rows);(args.output/f'qualification_{name}.json').write_text(json.dumps(rows,indent=2))
                print('QUALIFICATION',name,sum(r['success'] for r in rows[1:]),'/40',flush=True)
        rows=[]
        for a,b in zip(*paired):
            if a['initial_state_sha256']!=b['initial_state_sha256']:raise RuntimeError('Unpaired full-fall starts')
            rows.append({'seed':a['seed'],'baseline':a,'candidate':b})
        rs=rows[1:];report.update(canonical=rows[0],heldout=rs,heldout_trials=len(rs),
            heldout_successes=sum(r['candidate']['success'] for r in rs),
            heldout_baseline_successes=sum(r['baseline']['success'] for r in rs),
            paired_rescues=[r['seed'] for r in rs if r['candidate']['success'] and not r['baseline']['success']],
            paired_regressions=[r['seed'] for r in rs if r['baseline']['success'] and not r['candidate']['success']])
    else:report['qualification_skipped_reason']='No unbiased improvement passing command-noise development comparison'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
