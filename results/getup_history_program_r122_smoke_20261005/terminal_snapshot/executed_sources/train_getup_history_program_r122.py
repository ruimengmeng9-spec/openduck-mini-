"""Matched regularized snapshot/history program prediction, simulation only.

No lookup table or case metadata in exported models. The comparison changes
only preparation sensor features. Teacher recipes are offline targets.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import probe_getup_sensor_history_r121 as audit
from diagnostics import train_getup_program_r113 as program
from diagnostics import search_getup_case_teachers_r109 as search
from diagnostics.train_getup_joint_anchor_r102 import write_json, aggregate
from diagnostics.getup_independent_native import digest

ROOT=audit.ROOT
OUTPUT=ROOT/'outputs/getup_history_program_r122_left_20261005'
RIDGES=(1.,10.,100.)


def sensor_features(history,mode):
    h=np.asarray(history,dtype=np.float64)
    if h.shape!=(40,50) or not np.isfinite(h).all():
        raise ValueError('Only original 40 by 50 finite causal sensors allowed')
    if mode=='snapshot':return h[-1].copy()
    if mode=='history':return np.concatenate((h[-1],h[:10].mean(0),h[10:30].mean(0),h[30:].mean(0)))
    raise ValueError('Unknown feature mode')


def fit_ridge(x,flags,nodes,nominal,ridge):
    # All statistics are calculated from the supplied training partition only.
    mean=x.mean(0);std=np.maximum(x.std(0),1e-4)
    z=(x-mean)/std;nom=(nominal-mean)/std
    flagx=np.column_stack((z,np.ones(len(z))))
    difference=z-nom
    fw=flagx.T@np.linalg.solve(flagx@flagx.T+ridge*np.eye(len(x)),flags*2.-1.)
    nw=difference.T@np.linalg.solve(difference@difference.T+ridge*np.eye(len(x)),nodes.reshape(len(x),60))
    return dict(mean=mean,std=std,nominal_feature=nominal.copy(),flags=fw,nodes=nw)


def scalar_program(model,history):
    mode=str(model['feature_mode'].item())
    feature=sensor_features(history,mode)
    z=(feature-model['mean'])/model['std']
    logits=np.append(z,1.)@model['flags']
    # Subtract features before matrix multiplication: exact nominal zero, not
    # an exact-match branch or a sensor-context lookup.
    delta=(feature-model['nominal_feature'])/model['std']
    knots=np.clip(delta@model['nodes'],-1.,1.).reshape(6,10)
    has_knots=bool(logits[1]>=0.)
    if not has_knots:knots=np.zeros((6,10))
    if not np.isfinite(logits).all() or not np.isfinite(knots).all():raise ValueError('Nonfinite program')
    return int(logits[0]>=0.),knots,dict(logits=logits.tolist(),has_knots=has_knots)


def fit(output):
    source=audit.OUTPUT
    assert json.loads((source/'results.json').read_text())['full_original_teacher_bitwise_equal']
    manifest=json.loads((source/'manifest.json').read_text())
    with np.load(source/'causal_sensor_dataset.npz',allow_pickle=False) as data:histories=data['histories'][:65].copy()
    cases=[None,*program.TRAIN,*range(3160000,3160040)]
    assert [r['case_seed'] for r in manifest[:65]]==cases
    flags=[];nodes=[]
    for s in cases:
        with np.load(audit.TEACHERS/f'teachers/case_{s}/teacher_parameters.npz',allow_pickle=False) as data:
            knots=data['knots'].copy();flags.append([int(data['profile']),bool(np.any(knots!=0.))]);nodes.append(knots)
    flags=np.asarray(flags,dtype=float);nodes=np.stack(nodes)
    output.mkdir(exist_ok=False)
    with np.load(program.MODEL,allow_pickle=False) as data:frozen={'feedback_'+k:data[k].copy() for k in data.files}
    with np.load(audit.TEACHERS/'frozen_profiles.npz',allow_pickle=False) as data:
        frozen.update(feedback_anchors=data['anchors'].copy(),feedback_gains=data['gains'].copy())
    reports={};models={}
    for mode in ('snapshot','history'):
        x=np.stack([sensor_features(h,mode) for h in histories]);cv=[]
        for ridge in RIDGES:
            errors=[];flagerrors=[]
            for held in range(1,65):
                keep=np.arange(65)!=held
                w=fit_ridge(x[keep],flags[keep],nodes[keep],x[0],ridge)
                m={**w,'feature_mode':np.array(mode)}
                p,k,info=scalar_program(m,histories[held])
                errors.append(float(np.mean((k-nodes[held])**2)))
                flagerrors.append((int(p!=int(flags[held,0]))+int(info['has_knots']!=bool(flags[held,1])))/2.)
            # Predeclared selection on teacher target fit only; this score is
            # never called recovery performance and uses no reserved states.
            cv.append(dict(ridge=ridge,node_mse=float(np.mean(errors)),flag_error=float(np.mean(flagerrors)),
                selection_score=float(np.mean(errors)+.01*np.mean(flagerrors))))
        chosen=min(cv,key=lambda r:(r['selection_score'],-r['ridge']))['ridge']
        w=fit_ridge(x,flags,nodes,x[0],chosen)
        w.update(frozen,feature_mode=np.array(mode),ridge=np.array(chosen))
        np.testing.assert_array_equal(scalar_program(w,histories[0])[1],np.zeros((6,10)))
        path=output/(mode+'.npz');np.savez_compressed(path,**w);models[mode]=str(path)
        predictions=[scalar_program(w,h) for h in histories]
        reports[mode]=dict(cross_validation=cv,selected_ridge=chosen,input_dim=len(x[0]),
            teacher_node_mse=float(np.mean((np.stack([p[1] for p in predictions])-nodes)**2)),
            profile_correct=sum(p[0]==int(y[0]) for p,y in zip(predictions,flags)),
            knot_flag_correct=sum(p[2]['has_knots']==bool(y[1]) for p,y in zip(predictions,flags)),
            model_sha256=digest(path),teacher_fit_not_closed_loop_success=True)
        np.savez_compressed(output/(mode+'_closed_fit_state.npz'),**w,
            ridge_grid=np.array(RIDGES),training_feature_matrix=x,training_flags=flags,training_nodes=nodes)
    # Closed form deterministic fit: there is no iterative optimizer or random
    # sampling. Store the complete solve state separately from inference models.
    write_json(output/'fit_results.json',reports)
    write_json(output/'rng.json',dict(seed=222,deterministic_closed_form=True,no_optimizer_or_sampling=True))
    return models


def full_trial(job):
    model_file,case,directory,baseline,zero=job
    if case not in audit.KNOWN:raise ValueError('Reserved independent states are not part of this development experiment')
    with np.load(model_file,allow_pickle=False) as data:model={k:data[k].copy() for k in data.files}
    env=audit.HistoryReferenceEpisode(str(program.SCENE),str(program.STAND),program.REFERENCE,222,True)
    env.reset(case);context=env.preparation_sensors.copy()
    profile,knots,info=scalar_program(model,context)
    if zero:profile=0;knots=np.zeros((6,10));info=dict(fixed_zero_interface_test=True)
    records=[]
    while True:
        obs=env.observe();k=env.controls
        action=program.execute_program(model,obs,k,profile,knots)
        step=env.step(action,auto_reset=False)
        records.append((obs.copy(),action.copy(),env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if step[2]:break
    row=step[5];row.update(predicted_profile=profile,predicted_knots=knots.tolist(),program_info=info,
        full_path_trial=True,independent_qualification_trial=False,initial_sensor_history_only=True,
        case_identity_not_policy_input=True,model_sha256=digest(model_file),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None
            and row['entry_time_s']<=12. and row['strict_tail_s']>=30.-1e-8))
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],
        preparation_sensors=context,time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],
        applied=[r[5] for r in records],strict=[r[6] for r in records])
    np.savez_compressed(dest/'trajectory.npz',**arrays)
    assert row['initial_hash']==baseline['initial_hash']
    row['paired_original_initial_hash']=True
    if case is None or zero:
        reference=audit.TEACHERS/'teachers/case_None' if case is None else Path(baseline['_directory'])
        with np.load(reference/'trajectory.npz') as old:
            equality={key:bool(np.array_equal(arrays[key],old[key])) for key in ('qpos','qvel','applied','strict')}
        row['zero_interface_or_nominal_parity']=equality
        write_json(dest/'parity.json',row)
        assert all(equality.values())
    write_json(dest/'result.json',row)
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False)
    shutil.copytree(audit.OUTPUT/'executed_sources',args.output/'executed_sources')
    for name in (Path(__file__).name,'test_getup_history_program_r122.py','launch_getup_history_program_r122.py'):
        shutil.copy2(Path(__file__).with_name(name),args.output/'executed_sources'/name)
    write_json(args.output/'contract.json',dict(seed=222,simulation_only=True,
        hypothesis='Matched regularized predictors test whether causal preparation history improves complete closed-loop recovery over a single sensor snapshot',
        modes=['snapshot','history'],ridge_grid=RIDGES,teacher_examples=65,
        no_exact_decoder_calibration=True,no_lookup_or_case_input=True,leave_one_out_training_only_normalization=True,
        fixed_cv_selection='node_mse + 0.01 * binary_flag_error; tie favors larger ridge',
        closed_form_fit_complete_state_saved=True,preparation_controls_unchanged=40,
        original_acceptance_unchanged=True,control_hz=50,physics_hz=500,full_controls=2279,
        entry_deadline_s=12,strict_tail_s=30,total_feedback_cap_rad=.18,
        unseen_3200000_to_3200039_never_loaded=True,no_mid_episode_root_reset=True,hardware_readiness=False,
        executed_hashes={str(p):digest(p) for p in [program.SCENE,program.STAND,program.REFERENCE,program.MODEL,
            audit.OUTPUT/'causal_sensor_dataset.npz',*(args.output/'executed_sources').iterdir()]}))
    models=fit(args.output/'training')
    r116=json.loads((audit.FORMER/'results.json').read_text())
    cases=[None,*program.TRAIN,*range(3160000,3160040),*range(3180000,3180040)]
    old_zero=json.loads((ROOT/'outputs/getup_program_r113_left_20261005/results.json').read_text())['baseline_reused_from_R110']['rows']
    expanded_zero=json.loads((ROOT/'outputs/getup_program_calibration_r114_left_20261005/results.json').read_text())['independent']['baseline']['rows']
    baselines={r['case_seed']:r for r in [*old_zero,*expanded_zero,*r116['independent']['baseline']['rows']]}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        # Standalone full-path smoke includes zero-interface parity and both
        # learned sensor models. Short-time fit results cannot bypass parity.
        zeros=[]
        for s in (None,769000):
            b=baselines[s].copy()
            b['_directory']=str(audit.FORMER/('baseline_original'/f'case_{s}'))
            zeros.append((models['snapshot'],s,str(args.output/'interface'/f'case_{s}'),b,True))
        # Original zero feedback was preserved by R110, not R116's candidate.
        for i,j in enumerate(zeros):
            b=j[3]
            if j[1] is not None:
                matches=list((ROOT/'outputs/getup_distill_r110_left_20261005').rglob(f'case_{j[1]}/result.json'))
                matches=[p for p in matches if 'baseline' in str(p)]
                assert len(matches)==1
                b['_directory']=str(matches[0].parent)
        interface=list(pool.map(full_trial,zeros));summaries={}
        validation_cases=[None,769000] if args.smoke else cases
        for mode,file in models.items():
            jobs=[(file,s,str(args.output/mode/f'case_{s}'),baselines[s],False) for s in validation_cases]
            rows=list(pool.map(full_trial,jobs))
            summaries[mode]=dict(all=aggregate(rows),original=aggregate([r for r in rows if r['case_seed'] in [None,*program.TRAIN]]),
                expanded=aggregate([r for r in rows if r['case_seed'] in range(3160000,3160040)]),
                former_qualification_now_development=aggregate([r for r in rows if r['case_seed'] in range(3180000,3180040)]))
            write_json(args.output/'progress.json',dict(completed_mode=mode,summary=summaries[mode]))
            print('R122_FULL_DEVELOPMENT',mode,json.dumps({g:{k:v for k,v in a.items() if k!='rows'} for g,a in summaries[mode].items()}),flush=True)
    write_json(args.output/'results.json',dict(smoke=args.smoke,summaries=summaries,interface_trials=interface,
        independent_qualification_run=False,full_task_completed=False,hardware_readiness=False,
        qualification_requires_separate_frozen_gate_audit=True))
    print('R122_TERMINAL',flush=True)


if __name__=='__main__':main()
