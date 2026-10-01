"""R89: distill successful R87 teachers into causal nonlinear head feedback.

Teacher choice may use development outcomes OFFLINE. Runtime never sees case
seed, teacher ID, outcome, root state or future observation. Full fallen-start
closed-loop simulation, not teacher-forced loss, is the promotion criterion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np

from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT, digest
from diagnostics.probe_getup_wait_library_r77 import TRAIN, save_trace
from diagnostics.search_getup_reference_feedback_r64 import features, residual, rollout
from diagnostics.search_getup_support_margin_r49 import foot_support, support_quality, supported_entry
from diagnostics.train_getup_prefix_pose_r73 import success
from diagnostics.train_getup_head_feedback_r87 import path, matrix, correction, CAP
from diagnostics.train_getup_early_feedback_r81 import early_envelope

HELDOUT = tuple(range(789000,789040))
RAW_BOUND = 2.
OUT = None


def inputs(sim, error, elapsed, ids):
    # 14 measured encoders, 14 joint velocities, two OWN last command targets.
    # No floating-base qpos/qvel and no contact/privileged state.
    return np.r_[error, sim.data.qpos[sim.qadr]-sim.home,
                 .15*sim.data.qvel[sim.vadr], sim.prev[ids[8:10]]-sim.home[ids[8:10]],
                 elapsed/3.5]


def predict(network, x):
    normalized = np.clip((x-network['center'])/network['scale'], -10., 10.)
    hidden = np.tanh(normalized@network['w1']+network['b1'])
    return np.clip(hidden@network['w2']+network['b2'], -RAW_BOUND, RAW_BOUND)


def head_offset(error, gains, elapsed, raw):
    value=residual(error,gains)
    value[8:10] += early_envelope(elapsed)*np.asarray(raw)
    return np.clip(value,-CAP,CAP)


def init_worker(scene,stand,ck,seeds,out):
    global OUT
    base.init_worker(scene,stand,ck,seeds); OUT=Path(out)


def run(sim,source,peaks,targets,phases,ids,ref,gains,
        teacher=None,network=None,strength=1.,record=False,collect=False):
    if teacher is None and network is None:
        return (*rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,record),[],[])
    if network is not None and strength==0.:
        return (*rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,record),[],[])
    sim.restore(source); sim.peaks,sim.finite=peaks.copy(),True
    best=-np.inf;tail=longest=run_length=0;trace=[];xs=[];ys=[]
    head=matrix(teacher) if teacher is not None else None
    for i,(target_base,phase) in enumerate(zip(targets,phases)):
        elapsed=i*DT;error=features(sim)-ref[i];x=inputs(sim,error,elapsed,ids)
        raw=head@error if head is not None else strength*predict(network,x)
        if head is not None:
            offset=correction(error,gains[phase],head,elapsed)  # Exact R87 teacher.
        else:offset=head_offset(error,gains[phase],elapsed,raw)
        if collect and early_envelope(elapsed)>.05:
            xs.append(x);ys.append(raw.copy())
        target=target_base.copy()
        target[ids]=np.clip(target[ids]+offset,sim.lower[ids],sim.upper[ids])
        sim.step_target(target);valid=sim.physical_valid();m=sim.measure();s=foot_support(sim)
        best=max(best,support_quality(m,s));run_length=run_length+1 if supported_entry(m,s) else 0
        longest=max(longest,run_length);tail=tail+1 if m['stable'] else 0
        if record:trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),int(phase),
                               int(m['stable']),s['margin_m'],error.copy(),offset.copy()))
        if not valid:break
    score=-100. if not valid else best+10*min(longest*DT,1.)+20*min(tail*DT,1.)
    result={'score':float(score),'valid':bool(valid),'strict_tail_s':tail*DT,
            'supported_longest_s':longest*DT,'completed_steps':i+1,'final':m,
            'final_support':s,'peaks':sim.peaks.copy()}
    return result,trace,xs,ys


def collect_teacher(job):
    seed,flat,name=job;sim,cases,ck,ids=base._CTX
    targets,phases,ref,gains=path(2.)
    source,peaks,initial,initial_hash=cases[seed]
    result,trace,x,y=run(sim,source,peaks,targets,phases,ids,ref,gains,teacher=flat,
                       record=True,collect=True)
    result['success']=success(result,len(targets),1.)
    dest=OUT/'teachers'/f'case_{seed}';dest.mkdir(exist_ok=False)
    save_trace(dest/'teacher.npz',trace)
    np.savez_compressed(dest/'dataset.npz',features=x,labels=y)
    result.update(seed=seed,teacher_name=name,teacher_gains=flat,
                  initial_state_sha256=initial_hash,initial=initial)
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    if not result['success']:raise RuntimeError(f'Teacher failed exact reproduction: {seed}')
    return result


def fit(x,y,seed,output):
    if x.shape[1]!=35 or y.shape!=(len(x),2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid causal teacher dataset')
    rng=np.random.default_rng(seed)
    net={'center':x.mean(axis=0),'scale':np.maximum(x.std(axis=0),.01),
         'w1':rng.normal(0,.08,(35,64)), 'b1':np.zeros(64),
         'w2':np.zeros((64,2)), 'b2':np.zeros(2)}
    u=np.clip((x-net['center'])/net['scale'],-10.,10.)
    # Bound only the internal representation; physical combined cap remains .18.
    target=np.clip(y,-RAW_BOUND,RAW_BOUND)
    moments={k:[np.zeros_like(net[k]),np.zeros_like(net[k])] for k in ('w1','b1','w2','b2')}
    history=[];saved=[]
    for epoch in range(1,301):
        hidden=np.tanh(u@net['w1']+net['b1']);prediction=hidden@net['w2']+net['b2']
        delta=2*(prediction-target)/(len(x)*2)
        dh=(delta@net['w2'].T)*(1-hidden**2)
        gradients={'w2':hidden.T@delta+1e-4*net['w2'],'b2':delta.sum(axis=0),
                   'w1':u.T@dh+1e-4*net['w1'],'b1':dh.sum(axis=0)}
        for key,g in gradients.items():
            m,v=moments[key];m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            net[key]-=.003*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
        if epoch%25==0:
            history.append({'epoch':epoch,'training_mse':float(np.mean((prediction-target)**2))})
            (output/'fit_history.json').write_text(json.dumps(history,indent=2))
            print('FIT',epoch,'MSE',history[-1]['training_mse'],flush=True)
        if epoch in (50,150,300):
            file=output/f'network_epoch{epoch:03d}.npz';np.savez_compressed(file,**net);saved.append(file)
    return saved


def evaluate(job):
    seed,file,strength,hold,required,directory=job;sim,cases,ck,ids=base._CTX
    with np.load(file,allow_pickle=False) as z:net={k:z[k] for k in z.files}
    targets,phases,ref,gains=path(hold);source,peaks,initial,initial_hash=cases[seed]
    result,trace,_,_=run(sim,source,peaks,targets,phases,ids,ref,gains,
                        network=net,strength=strength,record=True)
    result.update(seed=seed,success=success(result,len(targets),required),
                  initial_state_sha256=initial_hash,initial=initial)
    dest=Path(directory)/f'case_{seed}';dest.mkdir(exist_ok=False)
    save_trace(dest/'candidate.npz',trace)
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    return result


def baseline(job):
    seed,hold,required,directory=job;sim,cases,ck,ids=base._CTX
    targets,phases,ref,gains=path(hold);source,peaks,initial,initial_hash=cases[seed]
    result,trace=rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,True)
    result.update(seed=seed,success=success(result,len(targets),required),
                  initial_state_sha256=initial_hash)
    dest=Path(directory)/f'case_{seed}';dest.mkdir(exist_ok=False)
    save_trace(dest/'baseline.npz',trace);(dest/'summary.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--seed',type=int,default=189)
    args=parser.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:parser.error('Fresh output and bounded workers required')
    root=Path('/data/shijinsheng/open_duck');r87dir=root/'outputs/getup_head_feedback_r87_left_20261001'
    previous=json.loads((r87dir/'results.json').read_text())
    scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if digest(file)!=previous[key]:raise RuntimeError('Frozen R87 inputs changed')
    assert not set(TRAIN)&set(HELDOUT)
    library=[(p,json.loads(p.read_text())) for p in sorted((r87dir/'candidates').glob('*/summary.json'))]
    teacher_jobs=[];teacher_hashes={}
    for seed in (None,*TRAIN):
        if seed is None or next(c for c in previous['training_best']['cases'] if c['seed']==seed)['success']:
            teacher_jobs.append((seed,[0.]*8,'identity'))
        else:
            eligible=[(p,c) for p,c in library if c['nominal_success'] and
                      any(v['seed']==seed and v['success'] for v in c['cases'])]
            if not eligible:raise RuntimeError(f'No audited teacher for {seed}')
            p,c=min(eligible,key=lambda pc:(np.linalg.norm(pc[1]['params']),-pc[1]['quality']))
            teacher_jobs.append((seed,c['params'],str(p.relative_to(r87dir))))
            teacher_hashes[str(p.relative_to(r87dir))]=digest(p)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();(args.output/'teachers').mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,sources/file.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
              'method':'successful teacher nonlinear head-feedback behavioral distillation',
              'no_midpath_state_reset':True,'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
              'seed':args.seed,'workers':args.workers,'teacher_selection_offline_only':True,
              'inputs':['four_IMU_reference_errors','14_joint_encoder_offsets','14_scaled_joint_velocities',
                        'two_own_previous_head_targets','elapsed_time'],
              'forbidden_runtime_inputs':['trial_seed','root_state','contact_force','future_observation','outcome'],
              'network_shape':[35,64,2],'epochs':300,'learning_rate':.003,
              'strength_grid':[.25,.5,1.],'same_window_seconds':[.5,1.5,2.5,3.5],
              'combined_residual_cap_rad':CAP,'internal_prediction_bound_not_actuation':RAW_BOUND,
              'required_strict_seconds':30.,'hold_seconds':35.,'training_gate_seconds':1.,
              'qualification_only_after_training_count_improvement':True,
              'frozen':['all_reference_targets','leg_feedback','phase_timing','initial_settling',
                        'collision','torque','joint_range','target_slew','strict_gate'],
              'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
              'scene_sha256':digest(scene),'stand_sha256':digest(stand),
              'r87_result_sha256':digest(r87dir/'results.json'),'selected_teacher_hashes':teacher_hashes,
              'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
            initargs=(str(scene),str(stand),ck,TRAIN,str(args.output))) as pool:
        teachers=list(pool.map(collect_teacher,teacher_jobs,chunksize=1))
        (args.output/'teacher_review.json').write_text(json.dumps(teachers,indent=2))
        print('TEACHERS_REPRODUCED',sum(r['success'] for r in teachers),'/25',flush=True)
        x=[];y=[]
        for seed in (None,*TRAIN):
            with np.load(args.output/f'teachers/case_{seed}/dataset.npz',allow_pickle=False) as z:
                x.append(z['features']);y.append(z['labels'])
        x,y=np.concatenate(x),np.concatenate(y)
        np.savez_compressed(args.output/'teacher_dataset.npz',features=x,labels=y)
        (args.output/'dataset_summary.json').write_text(json.dumps({'samples':len(x),
            'feature_count':35,'max_teacher_raw':float(abs(y).max()),
            'internal_bound_clipped_samples':int((abs(y)>RAW_BOUND).any(axis=1).sum()),
            'not_a_policy_success_rate':True},indent=2))
        networks=fit(x,y,args.seed,args.output)
        base_dir=args.output/'baseline_training';base_dir.mkdir()
        baseline_rows=list(pool.map(baseline,[(s,2.,1.,str(base_dir)) for s in (None,*TRAIN)],chunksize=1))
        baseline_count=sum(r['success'] for r in baseline_rows[1:])
        if not baseline_rows[0]['success'] or baseline_count!=13:raise RuntimeError('Baseline reproduction failed')
        screen=[]
        for file in networks:
            for strength in (.25,.5,1.):
                directory=args.output/f'{file.stem}_strength{strength}';directory.mkdir()
                nominal=list(pool.map(evaluate,[(None,str(file),strength,2.,1.,str(directory))]))[0]
                rows=[]
                if nominal['success']:
                    rows=list(pool.map(evaluate,[(s,str(file),strength,2.,1.,str(directory)) for s in TRAIN],chunksize=1))
                row={'network':file.name,'network_sha256':digest(file),'strength':strength,
                     'nominal':nominal,'nominal_success':nominal['success'],
                     'successes':sum(r['success'] for r in rows),'cases':rows}
                screen.append(row);(args.output/'training_screen.json').write_text(json.dumps(screen,indent=2))
                print('CLOSED_LOOP',file.name,strength,'NOMINAL',int(nominal['success']),
                      'SUCCESS',row['successes'],'/24',flush=True)
    best=max(screen,key=lambda r:(r['nominal_success'],r['successes'],-r['strength']))
    report={**contract,'teacher_successes':sum(r['success'] for r in teachers),'baseline_successes':baseline_count,
            'training_best':best,'qualification_run':False,'training_screen':screen}
    if best['nominal_success'] and best['successes']>baseline_count:
        report['qualification_run']=True;qa=args.output/'qualification';qa.mkdir()
        cd,bd=qa/'candidate',qa/'baseline';cd.mkdir();bd.mkdir()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                initargs=(str(scene),str(stand),ck,HELDOUT,str(args.output))) as pool:
            pairs=[]
            candidate=pool.map(evaluate,[(s,str(args.output/best['network']),best['strength'],35.,30.,str(cd))
                                         for s in (None,*HELDOUT)],chunksize=1)
            old=pool.map(baseline,[(s,35.,30.,str(bd)) for s in (None,*HELDOUT)],chunksize=1)
            for a,b in zip(candidate,old):
                if a['initial_state_sha256']!=b['initial_state_sha256']:raise RuntimeError('Unpaired initial state')
                pairs.append({'seed':a['seed'],'candidate':a,'baseline':b})
                (args.output/'qualification_progress.json').write_text(json.dumps(pairs,indent=2))
                print('HELDOUT',a['seed'],'CANDIDATE',int(a['success']),'BASE',int(b['success']),flush=True)
        rs=pairs[1:];report.update(canonical=pairs[0],heldout=rs,heldout_trials=len(rs),
            heldout_successes=sum(r['candidate']['success'] for r in rs),
            heldout_baseline_successes=sum(r['baseline']['success'] for r in rs),
            paired_rescues=[r['seed'] for r in rs if r['candidate']['success'] and not r['baseline']['success']],
            paired_regressions=[r['seed'] for r in rs if r['baseline']['success'] and not r['candidate']['success']])
    else:report['qualification_skipped_reason']='No closed-loop training-count improvement; fresh seeds unused'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
