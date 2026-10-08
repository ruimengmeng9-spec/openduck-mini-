"""Offline causal sensor-transition identification, not a recovery controller.

Future observations are supervised targets only. No environment, root state,
case lookup, policy fitting, physical integration or success relabeling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path('/data/shijinsheng/open_duck')
SOURCE=ROOT/'outputs/getup_complementary_feedback_r133_left_20261006'
OUTPUT=ROOT/'outputs/getup_sensor_dynamics_r172b_20261008'
SMOKE=ROOT/'outputs/getup_sensor_dynamics_r172b_smoke_20261008'
HELD_CASES=(769000,769001,773000,773001)
CASES=tuple(range(769000,769008))+tuple(range(773000,773016))
RIDGE=1.
SCALE=np.array([1.]*3+[.05]*3+[.05]*28+[.05]*14+[1.]*2)
GROUPS={'gyro_rad_s':(0,3,1.),'upvector':(3,6,1.),'joint_position_rad':(6,20,1.),'joint_velocity_rad_s':(20,34,20.)}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def features(current,previous,planned_target_delta_rad,phase):
    """Only current and previous sensor deviations and current chosen action.

    Deviations use fixed same-phase nominal sensors. Planned target is AFTER
    the original joint/slew planner, BEFORE physical integration. In R172b it
    is logged training data; no online planning interface is claimed verified.
    """
    x=np.asarray(current,dtype=float);p=np.asarray(previous,dtype=float)
    a=np.asarray(planned_target_delta_rad,dtype=float);t=np.asarray(phase,dtype=float)
    if x.shape[-1:]!=(50,) or p.shape!=x.shape or a.shape!=x.shape[:-1]+(14,) or t.shape!=x.shape[:-1]:raise ValueError('Sensor50/current/previous, target14 and scalar phase required')
    if not all(np.isfinite(v).all() for v in (x,p,a,t)) or np.any(t<0) or np.any(t>1):raise ValueError('Finite causal inputs and phase in [0,1] required')
    b=np.tanh(np.concatenate([x/SCALE,p/SCALE,a/.05],axis=-1))
    return np.concatenate([b,b*np.sin(np.pi*t)[...,None],b*np.cos(np.pi*t)[...,None]],axis=-1)

def predict_increment(weights,current,previous,planned_target_delta_rad,phase):
    w=np.asarray(weights,dtype=float)
    if w.shape!=(342,34) or not np.isfinite(w).all():raise ValueError('Finite342x34 forward-dynamics weights required')
    return features(current,previous,planned_target_delta_rad,phase)@w*SCALE[:34]

def regressions():
    rng=np.random.default_rng(272)
    w=rng.normal(0,.01,(342,34));x=rng.normal(0,.01,(8,50));p=x*.5;a=rng.normal(0,.001,(8,14));t=np.linspace(0,1,8)
    np.testing.assert_array_equal(predict_increment(w,np.zeros_like(x),np.zeros_like(x),np.zeros_like(a),t),np.zeros((8,34)))
    np.testing.assert_array_equal(predict_increment(np.zeros_like(w),x,p,a,t),np.zeros((8,34)))
    out=predict_increment(w,x,p,a,t)
    for i in range(8):np.testing.assert_allclose(out[i],predict_increment(w,x[i],p[i],a[i],t[i]),rtol=1e-13,atol=1e-15)
    np.testing.assert_array_equal(features(x,p,a,t)[:,:50],np.tanh(x/SCALE))
    np.testing.assert_array_equal(features(x,p,a,t)[:,50:100],np.tanh(p/SCALE))
    np.testing.assert_array_equal(features(x,p,a,t)[:,100:114],np.tanh(a/.05))
    assert not np.array_equal(features(x,p,a,t),features(x,p,a+1e-4,t))
    assert np.abs(features(x*1e10,p,a,t)).max()<=1
    for bad in (np.nan,np.inf):
        b=x.copy();b[0,0]=bad
        try:features(b,p,a,t)
        except ValueError:pass
        else:raise AssertionError('Nonfinite input accepted')
    try:features(x,p,a,np.full(8,2.))
    except ValueError:pass
    else:raise AssertionError('Invalid phase accepted')
    # Immutable inputs and deterministic batch evaluation; no random runtime.
    before=[v.copy() for v in (x,p,a,t,w)];predict_increment(w,x,p,a,t)
    for old,new in zip(before,(x,p,a,t,w)):np.testing.assert_array_equal(old,new)
    return dict(passed=True,checks=10,seed=272,no_dynamic_replay=True)

def load_trajectory(program,case):
    path=SOURCE/f'program_{program:02d}'/f'case_{case}'
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:
        # Explicit white list: no qpos, qvel, root audit, rewards or labels.
        obs=z['observations'][:,:50].copy();applied=z['applied'].copy()
    row=json.loads((path/'result.json').read_text())
    assert obs.shape==(2279,50) and applied.shape==(2279,14) and row['valid'] and row['controls']==2279
    return obs,applied,dict(directory=str(path),initial_hash=row['initial_hash'],trajectory_sha256=sha(path/'trajectory.npz'),result_sha256=sha(path/'result.json'))

def samples(obs,applied,nominal,nominal_applied):
    d=obs.astype(float)-nominal.astype(float)
    prev=np.concatenate([d[:1],d[:-1]],axis=0)
    action=applied[:-1]-nominal_applied[:-1]
    # Next observed previous target corresponds to this saved applied command.
    # This contract checks logged action alignment, not an online planner.
    native=obs[1:,34:48];standard=nominal[1:,34:48]
    reconstructed=native.astype(float)-standard.astype(float)
    rounding=(np.abs(np.spacing(native)).astype(float)+np.abs(np.spacing(standard)).astype(float))/2+1e-14
    assert np.all(np.abs(reconstructed-action)<=rounding),float(np.max(np.abs(reconstructed-action)-rounding))
    phase=np.minimum(np.arange(2278)/529.,1.)
    f=features(d[:-1],prev[:-1],action,phase)
    y=(d[1:,:34]-d[:-1,:34])/SCALE[:34]
    return f,y,d[:-1,:34],d[1:,:34]

def stage_metrics(predicted,blind,previous,truth):
    stages={'early_control_0_49':slice(0,50),'recovery_control_50_528':slice(50,529),'home_control_529_2277':slice(529,2278)}
    out={}
    for name,s in stages.items():
        out[name]={}
        for group,(lo,hi,mult) in GROUPS.items():
            errors=[(z[s,lo:hi]-truth[s,lo:hi])*mult for z in (predicted,blind,previous)]
            out[name][group]={key:float(np.sqrt(np.mean(e*e))) for key,e in zip(('action_model_rmse','action_blind_model_rmse','nominal_deviation_persistence_rmse'),errors)}
            out[name][group]['action_model_max_abs_error']=float(np.abs(errors[0]).max())
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and not args.output.exists()
    args.output.mkdir();src=args.output/'executed_sources';src.mkdir()
    for name in (Path(__file__).name,'launch_getup_sensor_dynamics_r172b.py'):
        shutil.copy2(Path(__file__).with_name(name),src/name)
    regression=regressions();write(args.output/'regressions.json',regression)
    nominal,nominal_applied,nominal_meta=load_trajectory(0,None)
    manifest=[];dataset={};hashes={}
    cases=[None,769000,773004] if args.smoke else [None,*CASES]
    programs=[0,1] if args.smoke else list(range(4))
    for program in programs:
        for case in cases:
            obs,applied,meta=load_trajectory(program,case)
            if case is None:np.testing.assert_array_equal(obs,nominal);np.testing.assert_array_equal(applied,nominal_applied)
            if case in hashes:assert hashes[case]==meta['initial_hash']
            hashes[case]=meta['initial_hash'];manifest.append(dict(program=program,case_seed=case,**meta))
            dataset[program,case]=samples(obs,applied,nominal,nominal_applied)
    write(args.output/'input_manifest.json',manifest)
    trainkeys=[k for k in dataset if k[0]<(1 if args.smoke else 3) and k[1] not in ((769000,) if args.smoke else HELD_CASES)]
    f=np.concatenate([dataset[k][0] for k in trainkeys]);y=np.concatenate([dataset[k][1] for k in trainkeys])
    # One fixed ridge solve, no optimizer/grid/model selection. Nonlinear sensor
    # transition features, NOT sensor-to-action or teacher behavior cloning.
    gram=f.T@f;rhs=f.T@y
    w=np.linalg.solve(gram+RIDGE*np.eye(342),rhs)
    assert np.isfinite(w).all()
    fb=f.copy();fb[:,100:114]=0;fb[:,214:228]=0;fb[:,328:342]=0
    blind=np.linalg.solve(fb.T@fb+RIDGE*np.eye(342),fb.T@y)
    np.savez_compressed(args.output/'model.npz',weights=w,action_blind_weights=blind,sensor_scales=SCALE,ridge=RIDGE,seed=272)
    checkpoint=args.output/'checkpoint_closed_01';checkpoint.mkdir()
    np.savez_compressed(checkpoint/'normal_equations.npz',gram=gram,rhs=rhs,weights=w)
    write(checkpoint/'learner.json',dict(method='deterministic float64 ridge normal equations',ridge=RIDGE,seed=272,random_sampling=False,rng=np.random.default_rng(272).bit_generator.state,training_keys=[list(k) for k in trainkeys],training_transitions=len(f)))
    rows=[];signals=args.output/'predictions';signals.mkdir()
    for (program,case),(x,target,past,truth) in dataset.items():
        pred=past+x@w*SCALE[:34];abl=past+x@blind*SCALE[:34]
        if case is None:np.testing.assert_array_equal(pred,np.zeros_like(pred))
        np.savez_compressed(signals/f'program_{program:02d}_case_{case}.npz',predicted_native34_deviation=pred,action_blind_deviation=abl,persistence_deviation=past,actual_next_native34_deviation=truth)
        split='training' if (program,case) in trainkeys else 'both_held' if program>=3 and case in HELD_CASES else 'case_held' if case in HELD_CASES else 'program_held'
        if args.smoke:split='training' if (program,case) in trainkeys else 'held_smoke'
        rows.append(dict(program=program,case_seed=case,split=split,metrics=stage_metrics(pred,abl,past,truth)))
    summary={}
    for split in sorted({r['split'] for r in rows}):
        selected=[r for r in rows if r['split']==split and r['case_seed'] is not None]
        summary[split]=dict(trajectories=len(selected),metrics={})
        if not selected:continue
        for stage in selected[0]['metrics']:
            summary[split]['metrics'][stage]={}
            for group in GROUPS:
                summary[split]['metrics'][stage][group]={key:float(np.mean([r['metrics'][stage][group][key] for r in selected])) for key in ('action_model_rmse','action_blind_model_rmse','nominal_deviation_persistence_rmse')}
    after=[]
    for m in manifest:
        p=Path(m['directory']);after.append(sha(p/'trajectory.npz')==m['trajectory_sha256'] and sha(p/'result.json')==m['result_sha256'])
    assert all(after)
    if not args.smoke:
        sm=json.loads((SMOKE/'results.json').read_text());assert sm['terminal_result_saved'] and sm['regressions']==regression
        old=json.loads((SMOKE/'input_manifest.json').read_text())
        for m in old:assert m in manifest
    write(args.output/'contract.json',dict(seed=272,smoke=args.smoke,method='sensor-space nonlinear-feature one-step transition regression, not a policy',ridge=RIDGE,feature_dimension=342,output_dimension=34,held_cases=list(HELD_CASES),held_program=3,case_split_chosen_before_training=True,hyperparameter_search=False,no_case_or_label_or_root_truth_in_predictor=True,inputs='current and previous causal native50 sensor deviations; current already planned joint/slew-limited14 target deviation; fixed phase',targets='next observed native34 deviation increment, offline supervision only',planned_action_interface_not_yet_runtime_verified=True,source_directory=str(SOURCE),nominal=nominal_meta,original_source_contract_hash=sha(SOURCE/'contract.json'),executed_source_hash=sha(src/Path(__file__).name),physics_and_original_acceptance_unchanged=True,no_dynamic_replay=True))
    write(args.output/'results.json',dict(smoke=args.smoke,regressions=regression,rows=rows,summary=summary,source_hashes_unchanged=True,input_trajectories=len(dataset),training_trajectories=len(trainkeys),training_transitions=len(f),model_hash=sha(args.output/'model.npz'),terminal_result_saved=True,offline_forward_model_training=True,new_dynamic_trajectories=0,new_recovery_controller=False,online_planner_verified=False,action_causal_identification_claim=False,full_task_completed=False,hardware_readiness=False,qualification_run=False))
    print('R172b_CLOSED_SENSOR_DYNAMICS',len(dataset),len(trainkeys),len(f),json.dumps(summary),flush=True)

if __name__=='__main__':main()
