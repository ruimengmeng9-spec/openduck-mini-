"""R118 offline local robustness of verified programs, not a lookup policy."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import search_getup_case_teachers_r109 as search
from diagnostics import train_getup_program_r113 as program
from diagnostics.train_getup_expanded_program_r116 import load_data
from diagnostics.search_getup_expanded_teachers_r115 import OUTPUT as TEACHER_RUN
from diagnostics.train_getup_joint_anchor_r102 import write_json
from diagnostics.getup_independent_native import digest

ROOT=program.ROOT
OUTPUT=ROOT/'outputs/getup_teacher_neighborhood_r118_20261005'


def nearest_pairs(contexts):
    x=np.asarray(contexts,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==55 and len(x)>=3 and np.isfinite(x).all()
    z=(x-x.mean(0))/np.maximum(x.std(0),1e-4)
    distances=np.sqrt(np.sum((z[:,None]-z[None,:])**2,axis=-1));np.fill_diagonal(distances,np.inf)
    return [(i,int(j),float(distances[i,j])) for i in range(len(x)) for j in np.argsort(distances[i])[:2]]


def init_worker(anchors,gains,rows):
    search.init_worker(anchors,gains);search.BASE_ROWS={r['case_seed']:r for r in rows}


def rollout(job):
    source,target,profile,knots,distance,own=job
    directory=OUTPUT/('own_replay' if own else 'neighbors')/f'source_{source}'/f'case_{target}'
    row=search.rollout((target,profile,knots,True,str(directory),None))
    row.update(source_teacher_case=source,target_development_case=target,sensor_context_distance=distance,
        offline_counterfactual_program_transfer=True,deployed_lookup_policy=False,independent_qualification_trial=False)
    if own:
        original=TEACHER_RUN/'teachers'/f'case_{target}'
        assert row['initial_hash']==json.loads((original/'result.json').read_text())['initial_hash']
        with np.load(directory/'trajectory.npz') as current,np.load(original/'trajectory.npz') as previous:
            for name in ('qpos','qvel','applied','strict','normalized_residual'):np.testing.assert_array_equal(current[name],previous[name])
        assert row['qualified_teacher'];row['own_complete_path_bitwise_parity']=True
    write_json(directory/'result.json',row)
    return row


def main():
    OUTPUT.mkdir(exist_ok=False);x,y,nodes,frozen,manifest=load_data()
    cases=[r['case_seed'] for r in manifest]
    rows=[json.loads((TEACHER_RUN/'teachers'/f'case_{s}'/'result.json').read_text()) for s in cases]
    programs=[]
    for s in cases:
        with np.load(TEACHER_RUN/'teachers'/f'case_{s}'/'teacher_parameters.npz') as data:
            programs.append((int(data['profile']),data['knots'].copy()))
    with np.load(TEACHER_RUN/'frozen_profiles.npz') as data:anchors=data['anchors'].copy();gains=data['gains'].copy()
    sources=OUTPUT/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_teacher_neighborhood_r118.py','launch_getup_teacher_neighborhood_r118.py',
        'train_getup_expanded_program_r116.py','train_getup_program_r113.py','train_getup_program_calibration_r114.py',
        'search_getup_expanded_teachers_r115.py','search_getup_case_teachers_r109.py','getup_reference_env_r100.py',
        'train_getup_joint_anchor_r102.py','probe_getup_anchor_r101.py','getup_independent_native.py',
        'validate_getup_fullpath_r27.py','train_getup_fullpath_r27.py','getup_fullfallen_env_r32.py',
        'getup_fullfallen_contract_r32.py','search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    pairs=nearest_pairs(x)
    write_json(OUTPUT/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='Individually verified teacher recipes may have narrow or incompatible local stability basins; matched neighbor transfers test this before another uniform parameter fit',
        teacher_count=65,own_replays=65,nearest_neighbors_per_teacher=2,neighbor_trials=130,workers=6,
        distance_uses_actual_initial_sensors_only=True,case_indices_offline_metadata_only=True,no_deployed_nearest_neighbor_policy=True,
        no_new_policy_training=True,known_development_only=True,unseen_3200000_to_3200039_never_loaded=True,
        physics_rewards_acceptance_unchanged=True,full_path_controls=2279,entry_deadline_s=12,strict_tail_s=30,
        control_hz=50,physics_hz=500,no_mid_episode_root_reset=True,
        hashes={str(f):digest(f) for f in [program.SCENE,program.STAND,program.REFERENCE,program.MODEL,TEACHER_RUN/'results.json',*sources.iterdir()]}))
    write_json(OUTPUT/'pairs.json',[dict(source_case=cases[i],target_case=cases[j],distance=d) for i,j,d in pairs])
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(anchors,gains,rows)) as pool:
        own=list(pool.map(rollout,[(s,s,*programs[i],0.,True) for i,s in enumerate(cases)]))
        write_json(OUTPUT/'own_replay.json',own);print('R118_OWN_PARITY',len(own),flush=True)
        trials=list(pool.map(rollout,[(cases[i],cases[j],*programs[i],d,False) for i,j,d in pairs]))
        groups={}
        for label,has_nodes in (('zero_nodes',False),('nonzero_nodes',True)):
            selected=[r for r in trials if bool(np.any(programs[cases.index(r['source_teacher_case'])][1]!=0))==has_nodes]
            groups[label]=dict(trials=len(selected),successes=sum(r['success'] for r in selected),physical_failures=sum(not r['valid'] for r in selected))
        result=dict(own_replays_bitwise_verified=len(own),neighbor_trials=trials,groups=groups,
            successes=sum(r['success'] for r in trials),physical_failures=sum(not r['valid'] for r in trials),
            diagnostic_development_only=True,unified_policy_success=False,independent_qualification_run=False,
            full_task_completed=False,hardware_readiness=False)
        write_json(OUTPUT/'results.json',result);print('R118_NEIGHBOR_TRANSFER',result['successes'],result['physical_failures'],flush=True)
    print('R118_TERMINAL',flush=True)


if __name__=='__main__':main()
