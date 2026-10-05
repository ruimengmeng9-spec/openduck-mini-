"""Frozen R122 ablation: remove only the predicted binary node gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_history_program_r122 as previous
from diagnostics.train_getup_joint_anchor_r102 import aggregate,write_json
from diagnostics.getup_independent_native import digest

ROOT=previous.ROOT
OUTPUT=ROOT/'outputs/getup_continuous_nodes_r124_left_20261005'


def predict(model,history,variant):
    profile,knots,info=previous.scalar_program(model,history)
    if variant not in ('original','continuous'):raise ValueError('Unsupported ablation')
    if variant=='continuous':
        feature=previous.sensor_features(history,str(model['feature_mode'].item()))
        knots=np.clip(((feature-model['nominal_feature'])/model['std'])@model['nodes'],-1.,1.).reshape(6,10)
    return profile,knots,info


def full_trial(job):
    file,case,variant,directory,old,parity=job
    if case not in previous.audit.KNOWN:raise ValueError('Unseen states excluded')
    with np.load(file,allow_pickle=False) as data:model={k:data[k].copy() for k in data.files}
    env=previous.audit.HistoryReferenceEpisode(str(previous.program.SCENE),str(previous.program.STAND),previous.program.REFERENCE,224,True)
    env.reset(case);context=env.preparation_sensors.copy();profile,knots,info=predict(model,context,variant)
    records=[]
    while True:
        obs=env.observe();k=env.controls
        action=previous.program.execute_program(model,obs,k,profile,knots)
        step=env.step(action,auto_reset=False)
        records.append((obs.copy(),action.copy(),env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if step[2]:break
    row=step[5];row.update(variant=variant,profile=profile,knots=knots.tolist(),original_node_gate_info=info,
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12. and row['strict_tail_s']>=30.-1e-8),
        full_path_trial=True,independent_qualification_trial=False,model_sha256=digest(file),
        original_initial_hash_equal=row['initial_hash']==old['initial_hash'],no_case_metadata_in_control=True)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],preparation_sensors=context,
        time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records])
    np.savez_compressed(dest/'trajectory.npz',**arrays)
    assert row['original_initial_hash_equal']
    if variant=='continuous' and not info['has_knots'] and np.any(knots!=0.):
        mode=str(model['feature_mode'].item())
        with np.load(previous.OUTPUT/mode/f'case_{case}/trajectory.npz') as stored:
            count=min(len(records),len(stored['normalized_residual']))
            changed=bool(not np.array_equal(np.array(arrays['normalized_residual'])[:count],stored['normalized_residual'][:count]))
        row['changed_actions_observed_after_gate_removal']=changed
        assert changed
    if parity or case is None:
        mode=str(model['feature_mode'].item())
        with np.load(previous.OUTPUT/mode/f'case_{case}/trajectory.npz') as stored:
            equal={name:bool(np.array_equal(values,stored[name])) for name,values in arrays.items()}
        row['full_bitwise_parity']=equal
        write_json(dest/'parity.json',row)
        assert all(equal.values())
    write_json(dest/'result.json',row);return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    old=json.loads((previous.OUTPUT/'results.json').read_text())
    args.output.mkdir(exist_ok=False)
    shutil.copytree(previous.OUTPUT/'executed_sources',args.output/'executed_sources')
    for name in (Path(__file__).name,'test_getup_continuous_nodes_r124.py','launch_getup_continuous_nodes_r124.py','audit_getup_program_gate_r123.py'):
        shutil.copy2(Path(__file__).with_name(name),args.output/'executed_sources'/name)
    models={};(args.output/'frozen_models').mkdir()
    for mode in ('snapshot','history'):
        src=previous.OUTPUT/'training'/(mode+'.npz');dst=args.output/'frozen_models'/src.name
        shutil.copy2(src,dst);assert digest(src)==digest(dst);models[mode]=str(dst)
    write_json(args.output/'contract.json',dict(simulation_only=True,seed=224,smoke=args.smoke,workers=args.workers,
        hypothesis='A learned hard node switch may discard continuous residual predictions; remove only this switch and test paired full recovery',
        frozen_model_hashes={m:digest(p) for m,p in models.items()},no_new_weight_training=True,
        unchanged_profile_classifier=True,unchanged_node_prediction=True,unchanged_node_bounds=True,
        no_context_lookup_or_case_policy_input=True,no_amplitude_scaling=True,
        preparation_history_unchanged=True,control_hz=50,physics_hz=500,full_controls=2279,
        entry_deadline_s=12,strict_tail_s=30,total_correction_cap_rad=.18,
        physics_reward_acceptance_unchanged=True,no_mid_episode_root_reset=True,
        reserved_qualification_never_loaded=True,hardware_readiness=False,full_task_completed=False,
        source_hashes={str(p):digest(p) for p in [previous.program.SCENE,previous.program.STAND,previous.program.REFERENCE,
            previous.OUTPUT/'results.json',*(args.output/'executed_sources').iterdir()]}))
    with np.load(previous.audit.OUTPUT/'causal_sensor_dataset.npz',allow_pickle=False) as data:histories=data['histories'].copy()
    cases=[r['case_seed'] for r in json.loads((previous.audit.OUTPUT/'manifest.json').read_text())]
    regression={}
    for mode,file in models.items():
        with np.load(file,allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
        before=old['summaries'][mode]['all']['rows'];assert [r['case_seed'] for r in before]==cases
        for case,h,row in zip(cases,histories,before):
            p,k,info=predict(w,h,'original');np.testing.assert_array_equal(k,row['predicted_knots']);assert p==row['predicted_profile']
            q,continuous,_=predict(w,h,'continuous');assert p==q and np.abs(continuous).max()<=1.
            if info['has_knots']:np.testing.assert_array_equal(k,continuous)
            if case is None:np.testing.assert_array_equal(continuous,np.zeros((6,10)))
        regression[mode]=dict(all_105_original_saved_predictions_equal=True,profile_unchanged=True,
            originally_enabled_nodes_equal=True,nominal_continuous_nodes_exact_zero=True)
    write_json(args.output/'prediction_parity.json',regression)
    summaries={};parities={}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for mode,file in models.items():
            previous_rows={r['case_seed']:r for r in old['summaries'][mode]['all']['rows']}
            parity_jobs=[(file,s,'original',str(args.output/'parity'/mode/f'case_{s}'),previous_rows[s],True) for s in (None,769000)]
            parities[mode]=list(pool.map(full_trial,parity_jobs))
            selected=[None,769000,769001] if args.smoke else cases
            jobs=[(file,s,'continuous',str(args.output/'continuous'/mode/f'case_{s}'),previous_rows[s],False) for s in selected]
            rows=list(pool.map(full_trial,jobs))
            groups=dict(all=aggregate(rows),original=aggregate([r for r in rows if r['case_seed'] in [None,*previous.program.TRAIN]]),
                expanded=aggregate([r for r in rows if r['case_seed'] in range(3160000,3160040)]),
                former_development=aggregate([r for r in rows if r['case_seed'] in range(3180000,3180040)]))
            gate=bool(not args.smoke and groups['original']['nominal_success'] and groups['original']['successes']>=22
                and groups['expanded']['successes']>=36 and groups['all']['physical_failures']==0)
            summaries[mode]=dict(groups=groups,development_gate=gate,
                rescued_vs_original=[r['case_seed'] for r in rows if r['success'] and not previous_rows[r['case_seed']]['success']],
                regressed_vs_original=[r['case_seed'] for r in rows if not r['success'] and previous_rows[r['case_seed']]['success']])
            write_json(args.output/'progress.json',dict(completed_mode=mode,summary=summaries[mode]))
            print('R124_FULL',mode,json.dumps({g:{k:v for k,v in a.items() if k!='rows'} for g,a in groups.items()}),flush=True)
    write_json(args.output/'results.json',dict(smoke=args.smoke,summaries=summaries,parities=parities,
        independent_qualification_run=False,qualification_requires_separate_frozen_audit=True,
        hardware_readiness=False,full_task_completed=False))
    print('R124_TERMINAL',flush=True)


if __name__=='__main__':main()
