"""R117 causal program-component ablations after failed unseen R116 tests."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_program_r113 as program
from diagnostics.getup_reference_env_r100 import ReferenceEpisode
from diagnostics.train_getup_joint_anchor_r102 import write_json,aggregate
from diagnostics.getup_independent_native import digest

ROOT=program.ROOT
SOURCE=ROOT/'outputs/getup_expanded_program_r116_left_20261005'
MODEL=SOURCE/'frozen_candidate.npz'
OUTPUT=ROOT/'outputs/getup_program_extrapolation_r117_20261005'
CASES=list(range(3180000,3180040))


def transform(profile,knots,mode):
    k=np.asarray(knots,dtype=np.float64)
    assert profile in (0,1) and k.shape==(6,10) and np.isfinite(k).all() and np.abs(k).max()<=1.
    if mode=='original':return profile,k.copy()
    if mode=='no_nodes':return profile,np.zeros((6,10))
    if mode=='no_feedback':return 0,k.copy()
    raise ValueError('Unknown fixed diagnostic mode')


def rollout(job):
    case,mode,parity=job
    with np.load(MODEL,allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
    env=ReferenceEpisode(str(program.SCENE),str(program.STAND),program.REFERENCE,217,qualification=True);env.reset(case)
    context=env.observe().copy();original_profile,original_knots,info=program.predict_program(w,context)
    profile,knots=transform(original_profile,original_knots,mode);records=[]
    while True:
        obs=env.observe();action=program.execute_program(w,obs,env.controls,profile,knots)
        step=env.step(action,auto_reset=False)
        records.append((obs.copy(),action.copy(),env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if step[2]:break
    result=step[5];result.update(mode=mode,profile=profile,knots=knots.tolist(),original_profile=original_profile,
        actual_causal_context_only=True,diagnostic_development_trial=True,independent_qualification_trial=False,
        former_qualification_now_development=case in CASES,success=bool(step[5]['valid'] and step[5]['controls']==2279
            and step[5]['entry_time_s'] is not None and step[5]['entry_time_s']<=12. and step[5]['strict_tail_s']>=30.-1e-8))
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],initial_context=context,
        time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records])
    if parity:
        path=Path(parity);assert result['initial_hash']==json.loads((path/'result.json').read_text())['initial_hash']
        with np.load(path/'trajectory.npz') as old:
            for name in ('qpos','qvel','applied','strict'):np.testing.assert_array_equal(np.array(arrays[name]),old[name])
        result['original_full_path_bitwise_parity']=True
    folder=OUTPUT/mode/f'case_{case}';folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'trajectory.npz',**arrays);write_json(folder/'result.json',result)
    return result


def audit_parameters():
    from diagnostics.train_getup_expanded_program_r116 import load_data
    with np.load(MODEL,allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
    x,y,nodes,_,records=load_data();training=program.normalize(x,w['context_mean'],w['context_std'])
    previous=json.loads((SOURCE/'results.json').read_text())['independent']['candidate']['rows'];rows=[]
    for result in previous:
        case=result['case_seed'];path=SOURCE/'qualification_candidate'/f'case_{case}'
        with np.load(path/'trajectory.npz') as data:context=data['initial_context'].copy()
        profile,knots,info=program.predict_program(w,context)
        assert profile==result['profile'];np.testing.assert_array_equal(knots,np.array(result['knots']))
        norm=program.normalize(context,w['context_mean'],w['context_std'])
        distances=np.linalg.norm(training-norm,axis=1)
        rows.append(dict(case_seed=case,success=result['success'],valid=result['valid'],profile=profile,
            has_nodes=info['has_knots'],max_node_magnitude=float(np.abs(knots).max()),node_rms=float(np.sqrt(np.mean(knots**2))),
            node_near_bound_fraction=float(np.mean(np.abs(knots)>=.99)),normalized_context_nearest_distance=float(distances.min())))
    return dict(read_only=True,not_acceptance_relabeling=True,diagnostic_threshold_not_physics_limit=.99,
        teacher_node_max_magnitude=float(np.abs(nodes).max()),teacher_node_rms=float(np.sqrt(np.mean(nodes**2))),rows=rows)


def main():
    OUTPUT.mkdir(exist_ok=False);sources=OUTPUT/'executed_sources';sources.mkdir()
    previous=json.loads((SOURCE/'results.json').read_text());assert not previous['left_stage_passed']
    assert previous['independent']['candidate']['successes']==8 and previous['independent']['candidate']['physical_failures']==10
    for name in (Path(__file__).name,'test_getup_program_extrapolation_r117.py','launch_getup_program_extrapolation_r117.py',
        'train_getup_expanded_program_r116.py','train_getup_program_r113.py','train_getup_program_calibration_r114.py',
        'search_getup_expanded_teachers_r115.py','getup_reference_env_r100.py','probe_getup_anchor_r101.py',
        'train_getup_joint_anchor_r102.py','getup_independent_native.py','validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py','getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py',
        'search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    write_json(OUTPUT/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='Frozen R116 node extrapolation or fixed feedback-profile selection may explain its unseen physical failures; remove only one component per matched rollout',
        diagnostic_development_cases=CASES,former_qualification_reclassified_as_development=True,
        new_unseen_reserved=list(range(3200000,3200040)),new_unseen_never_loaded=True,
        no_training_in_this_probe=True,original_physics_rewards_acceptance_unchanged=True,
        model_sha256=digest(MODEL),source_result_sha256=digest(SOURCE/'results.json'),
        source_hashes={str(f):digest(f) for f in [program.SCENE,program.STAND,program.REFERENCE,*sources.iterdir()]}))
    write_json(OUTPUT/'parameter_audit.json',audit_parameters())
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        parity=list(pool.map(rollout,[(None,'no_nodes',str(program.TEACHERS/'case_None')),
            (None,'no_feedback',str(program.TEACHERS/'case_None')),
            (CASES[0],'original',str(SOURCE/'qualification_candidate'/f'case_{CASES[0]}'))]))
        write_json(OUTPUT/'parity.json',parity)
        reports={}
        for mode in ('no_nodes','no_feedback'):
            rows=list(pool.map(rollout,[(s,mode,None) for s in CASES]));report=aggregate(rows)
            assert all(r['initial_hash']==old['initial_hash'] for r,old in zip(rows,previous['independent']['candidate']['rows']))
            reports[mode]=report;write_json(OUTPUT/mode/'results.json',report)
            write_json(OUTPUT/'progress.json',reports);print('R117_ABLATION',mode,report['successes'],report['physical_failures'],flush=True)
        write_json(OUTPUT/'results.json',dict(ablations=reports,original=previous['independent'],
            diagnostic_development_only=True,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
    print('R117_TERMINAL',flush=True)


if __name__=='__main__':main()
