"""R115 expanded complete-fall offline teachers after R114 generalization fails.

Former independent seeds are explicitly development now. Case-specific recipes
are offline supervision, never a deployed case-indexed control policy.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import search_getup_case_teachers_r109 as search
from diagnostics import train_getup_program_r113 as program
from diagnostics.train_getup_joint_anchor_r102 import write_json,aggregate
from diagnostics.getup_independent_native import digest

ROOT=program.ROOT
R114=ROOT/'outputs/getup_program_calibration_r114_left_20261005'
OUTPUT=ROOT/'outputs/getup_expanded_teachers_r115_left_20261005'
EXPANDED=list(range(3160000,3160040))


def init_worker(anchors,gains,base_rows):
    search.init_worker(anchors,gains)
    search.BASE_ROWS={r['case_seed']:r for r in base_rows}


def compare_saved(row,directory,previous):
    assert row['initial_hash']==json.loads((previous/'result.json').read_text())['initial_hash']
    with np.load(directory/'trajectory.npz') as now,np.load(previous/'trajectory.npz') as old:
        for name in ('qpos','qvel','applied','strict'):np.testing.assert_array_equal(now[name],old[name])


def make_proposals(state,count):
    zero=np.zeros((6,10));proposals=[(0,zero.copy()),(1,zero.copy()),(state['profile'],state['knots'].copy())]
    while len(proposals)<count:
        profile=int(state['rng'].integers(2));center=state['knots'] if len(proposals)%3==0 else state['mean']
        knots=np.clip(center+state['rng'].normal(0.,state['std']),-1.,1.)
        proposals.append((profile,knots))
    return proposals


def main():
    OUTPUT.mkdir(exist_ok=False)
    prior=json.loads((R114/'results.json').read_text())
    assert not prior['left_stage_passed'] and prior['independent']['candidate']['successes']==15
    assert prior['independent']['baseline']['successes']==16
    baseline=prior['independent']['baseline']['rows'];predicted=prior['independent']['candidate']['rows']
    older=json.loads((program.ROOT/'outputs/getup_program_r113_left_20261005/results.json').read_text())['baseline_reused_from_R110']['rows']
    base_rows=[*older,*baseline]
    with np.load(program.TEACHERS/'frozen_profiles.npz') as data:anchors=data['anchors'].copy();gains=data['gains'].copy()
    np.savez_compressed(OUTPUT/'frozen_profiles.npz',anchors=anchors,gains=gains)
    sources=OUTPUT/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_expanded_teachers_r115.py','launch_getup_expanded_teachers_r115.py',
        'search_getup_case_teachers_r109.py','train_getup_program_r113.py','train_getup_joint_anchor_r102.py',
        'getup_reference_env_r100.py','getup_independent_native.py','validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py','probe_getup_anchor_r101.py','getup_fullfallen_env_r32.py',
        'getup_fullfallen_contract_r32.py','search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    write_json(OUTPUT/'contract.json',dict(simulation_only=True,teacher_data_only=True,unified_policy_success=False,
        hardware_readiness=False,full_task_completed=False,seed=215,workers=6,generations=8,population=14,
        hypothesis='R114 fits known programs but fails unseen starts; expanded fully verified program teachers test insufficient context coverage before another unified fit',
        development_cases=[*program.TRAIN,*EXPANDED],former_qualification_reclassified_as_development=EXPANDED,
        future_unseen_qualification_reserved=list(range(3180000,3180040)),future_unseen_qualification_never_loaded=True,
        case_identifiers_only_offline_data_keys=True,no_case_indexed_policy_deployment=True,
        teacher_full_path_controls=2279,entry_deadline_s=12,teacher_strict_tail_s=30,short_training_tail_s=1,
        control_hz=50,physics_hz=500,combined_correction_limit_rad=.18,normalized_nodes_bounds=[-1,1],
        rewards_physics_acceptance_unchanged=True,no_mid_episode_root_reset=True,
        hashes={str(f):digest(f) for f in [program.SCENE,program.STAND,program.REFERENCE,program.MODEL,R114/'results.json',*sources.iterdir()]}))
    zero=np.zeros((6,10));teachers=[];history=[]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(anchors,gains,base_rows)) as pool:
        # Independent executor regression: both fixed zero and learned R114
        # program replay the same actual fallen start and original complete path.
        first=predicted[0]
        jobs=[(EXPANDED[0],0,zero,True,str(OUTPUT/'parity_zero'),None),
            (EXPANDED[0],first['profile'],np.array(first['knots']),True,str(OUTPUT/'parity_predicted'),None)]
        parity=list(pool.map(search.rollout,jobs))
        compare_saved(parity[0],OUTPUT/'parity_zero',R114/'qualification_baseline'/f'case_{EXPANDED[0]}')
        compare_saved(parity[1],OUTPUT/'parity_predicted',R114/'qualification_candidate'/f'case_{EXPANDED[0]}')
        write_json(OUTPUT/'parity.json',dict(rows=parity,original_complete_path_bitwise_equal=True))
        print('R115_EXECUTOR_PARITY_PASS',flush=True)
        # New fixed-controller baseline is development, not an independent test.
        jobs=[(s,1,zero,True,str(OUTPUT/'expanded_r102'/f'case_{s}'),None) for s in EXPANDED]
        r102=list(pool.map(search.rollout,jobs));write_json(OUTPUT/'expanded_r102/results.json',aggregate(r102))
        programs={}
        for case in (None,*program.TRAIN):
            with np.load(program.TEACHERS/f'case_{case}/teacher_parameters.npz') as data:
                programs[case]=(int(data['profile']),data['knots'].copy())
        for s,b,p,r in zip(EXPANDED,baseline,predicted,r102):
            assert s==b['case_seed']==p['case_seed']==r['case_seed']
            if b['success']:programs[s]=(0,zero.copy())
            elif r['success']:programs[s]=(1,zero.copy())
            elif p['success']:programs[s]=(int(p['profile']),np.array(p['knots']))
        jobs=[(s,p,k,True,str(OUTPUT/'teachers'/f'case_{s}'),None) for s,(p,k) in programs.items()]
        teachers=list(pool.map(search.rollout,jobs));assert all(t['qualified_teacher'] for t in teachers)
        missing=[s for s in EXPANDED if s not in programs]
        predicted_by={r['case_seed']:r for r in predicted}
        state={}
        for s in missing:
            old=predicted_by[s];knots=np.array(old['knots']);state[s]=dict(rng=np.random.default_rng(215+s),
                mean=knots.copy(),std=np.full((6,10),.10),best=None,profile=int(old['profile']),knots=knots.copy(),full_checks={},qualified=None)
        write_json(OUTPUT/'coverage_before_search.json',dict(verified_teachers=len(teachers),missing_cases=missing,
            zero_successes=16,r114_successes=15,r102_successes=sum(r['success'] for r in r102),teacher_data_only=True))
        for gen in range(1,9):
            active=[s for s in missing if state[s]['qualified'] is None]
            if not active:break
            proposals={s:make_proposals(state[s],14) for s in active}
            jobs=[(s,p,k,False,None,None) for s in active for p,k in proposals[s]]
            rows=list(pool.map(search.rollout,jobs));cursor=0;generation=[]
            for s in active:
                st=state[s];items=proposals[s];reports=rows[cursor:cursor+14];cursor+=14
                order=sorted(range(14),key=lambda i:search.rank(reports[i]),reverse=True);winner=order[0]
                if st['best'] is None or search.rank(reports[winner])>search.rank(st['best']):
                    st.update(best=reports[winner],profile=items[winner][0],knots=items[winner][1].copy())
                elite=np.stack([items[i][1] for i in order[:3]])
                st['mean']=.6*st['mean']+.4*elite.mean(0);st['std']=np.clip(.6*st['std']+.4*elite.std(0),.025,.35)
                checked=0
                for i in order:
                    if not reports[i]['training_success']:break
                    p,k=items[i];key=str(p)+k.tobytes().hex()
                    if key in st['full_checks']:continue
                    dest=OUTPUT/'full_checks'/f'case_{s}'/f'generation_{gen:04d}_candidate_{i:03d}'
                    verified=list(pool.map(search.rollout,[(s,p,k,True,str(dest),None)]))[0]
                    st['full_checks'][key]=verified;checked+=1
                    if verified['qualified_teacher']:
                        st.update(qualified=verified,profile=p,knots=k.copy());teachers.append(verified)
                        shutil.copytree(dest,OUTPUT/'teachers'/f'case_{s}')
                    print('R115_FULL_CHECK',s,gen,i,verified['success'],verified['valid'],flush=True)
                    if st['qualified'] is not None or checked>=3:break
                checkpoint=OUTPUT/'checkpoints'/f'case_{s}';checkpoint.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(checkpoint/f'generation_{gen:04d}.npz',knots=st['knots'],profile=st['profile'],mean=st['mean'],std=st['std'])
                generation.append(dict(case_seed=s,candidates=[dict(profile=p,knots=k.tolist()) for p,k in items],reports=reports,
                    best=st['best'],qualified=st['qualified'],full_checks=list(st['full_checks'].values()),rng_state=st['rng'].bit_generator.state))
            history.append(dict(generation=gen,cases=generation));write_json(OUTPUT/'history.json',history)
            write_json(OUTPUT/'progress.json',dict(completed_generations=gen,verified_teacher_count=len(teachers),
                total_expected_teachers=65,missing_cases=[s for s in missing if state[s]['qualified'] is None],teacher_data_only=True,unified_policy_success=False))
            print('R115_GENERATION',gen,'TEACHERS',len(teachers),flush=True)
        write_json(OUTPUT/'results.json',dict(teachers=teachers,verified_teacher_count=len(teachers),total_expected_teachers=65,
            uncovered=[s for s in missing if state[s]['qualified'] is None],completed_generations=len(history),
            teacher_data_only=True,unified_policy_success=False,independent_qualification_run=False,
            simulation_only=True,hardware_readiness=False,full_task_completed=False))
    print('R115_TERMINAL',flush=True)


if __name__=='__main__':main()
