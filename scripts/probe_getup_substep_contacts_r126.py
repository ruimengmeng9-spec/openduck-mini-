"""Unchanged frozen-controller replay with passive mj_step contact recording.

No change to integration, contact model, control, root state or acceptance.
Saved original full traces must match every executed control bitwise.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import probe_getup_continuous_nodes_r124 as ablation
from diagnostics import audit_getup_self_geometry_r125 as geometry
from diagnostics.getup_independent_native import digest

previous = ablation.previous
ROOT = previous.ROOT
OUTPUT = ROOT/'outputs/getup_substep_contacts_r126_left_20261005'


def replay(job):
    label, case, tracefile, old, directory = job
    assert case in previous.audit.KNOWN
    model_file = previous.OUTPUT/'training'/('history.npz' if label.endswith('history') else 'snapshot.npz')
    with np.load(model_file, allow_pickle=False) as z: model={k:z[k].copy() for k in z.files}
    env=previous.audit.HistoryReferenceEpisode(str(previous.program.SCENE),str(previous.program.STAND),previous.program.REFERENCE,226,True)
    env.reset(case)
    assert env.initial_hash == old['saved_initial_hash']
    if label == 'teacher':
        file=previous.audit.TEACHERS/f'teachers/case_{case}/teacher_parameters.npz'
        with np.load(file,allow_pickle=False) as z: profile=int(z['profile']);knots=z['knots'].copy()
    elif label == 'zero': profile=0;knots=np.zeros((6,10))
    else:
        profile,knots,_=ablation.predict(model,env.preparation_sensors,'continuous' if label.startswith('r124') else 'original')
    with np.load(tracefile,allow_pickle=False) as z: stored={k:z[k].copy() for k in z.files}
    sim=env.sim; model_physics=sim.model; records=[]; crossings=[]
    original_step=mujoco.mj_step
    def recorded(m,d,*args,**kwargs):
        # Original single integration call. No FK, new contact calculation,
        # extra mj_step or simulator state write is introduced by this wrapper.
        pre_q=d.qpos.copy() if d is sim.data else None
        value=original_step(m,d,*args,**kwargs)
        if d is sim.data:
            pairs=[c for c in d.contact if sim.floor not in c.geom]
            if pairs:
                c=min(pairs,key=lambda c:c.dist);ids=tuple(sorted(map(int,c.geom)));pen=max(0.,-float(c.dist))
            else: ids=(-1,-1);pen=0.
            index=len(records)
            records.append((index,env.controls,index%10,float(d.time),pen,*ids))
            if pen >= .004:
                crossings.append(dict(substep_index=index,control_index=env.controls,substep_in_control=index%10,
                    post_step_time=float(d.time),self_penetration_m=pen,**geometry.describe(m,ids),
                    pre_step_qpos=pre_q.tolist(),post_step_qpos=d.qpos.tolist(),applied_target=sim.prev.tolist()))
        return value
    comparisons={k:True for k in ('observations','normalized_residual','time','qpos','qvel','applied','strict')}
    mujoco.mj_step=recorded
    try:
        while True:
            k=env.controls;obs=env.observe()
            action=previous.program.execute_program(model,obs,k,profile,knots)
            step=env.step(action,auto_reset=False)
            current=dict(observations=obs,normalized_residual=action,time=sim.data.time,
                qpos=sim.data.qpos,qvel=sim.data.qvel,applied=sim.prev,strict=env.tail>0)
            for name,val in current.items():
                comparisons[name] &= bool(np.array_equal(val,stored[name][k]))
            assert all(comparisons.values()), (label,case,k,comparisons)
            if step[2]: break
    finally: mujoco.mj_step=original_step
    row=step[5]
    assert row['controls']==len(stored['qpos'])
    if old['saved_substep_peaks'] is not None: assert row['peaks']==old['saved_substep_peaks']
    assert row['valid']==old['saved_valid']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'substep_contacts.npz',records=np.array(records),
        columns=np.array(['index','control','substep','post_step_time','self_penetration_m','geom0','geom1']))
    (dest/'threshold_crossings.json').write_text(json.dumps(crossings,indent=2))
    worst=max(records,key=lambda r:r[4])
    result=dict(label=label,case_seed=case,initial_hash=env.initial_hash,full_control_trace_bitwise_equal=comparisons,
        controls=row['controls'],physics_steps=len(records),saved_valid=row['valid'],saved_success=old['saved_success'],
        original_peaks_identical=True,peaks=row['peaks'],trace_sha256=digest(tracefile),model_sha256=digest(model_file),
        deepest_substep=dict(substep_index=int(worst[0]),control_index=int(worst[1]),substep_in_control=int(worst[2]),
            post_step_time=worst[3],self_penetration_m=worst[4],
            **(geometry.describe(model_physics,tuple(worst[5:])) if worst[5]>=0 else {})),
        threshold_crossings=len(crossings),first_crossing=None if not crossings else {k:v for k,v in crossings[0].items()
            if k not in ('pre_step_qpos','post_step_qpos','applied_target')},
        no_new_strategy=True,no_mid_episode_root_edits=True,physics_control_acceptance_unchanged=True)
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False)
    (args.output/'executed_sources').mkdir()
    for module in (Path(__file__),Path(geometry.__file__)):
        shutil.copy2(module,args.output/'executed_sources'/module.name)
    report=json.loads((geometry.OUTPUT/'results.json').read_text())
    selected=list(report['rows'])
    if args.smoke:
        selected=[r for r in selected if (r['label'],r['case_seed']) in [('r122_history',773004),('teacher',773004),('zero',773004)]]
        assert len(selected)==3
    standard=next(r for r in json.loads((previous.OUTPUT/'results.json').read_text())['summaries']['snapshot']['all']['rows'] if r['case_seed'] is None)
    selected.append(dict(label='r122_snapshot',case_seed=None,trace_path=str(previous.OUTPUT/'snapshot/case_None/trajectory.npz'),
        saved_initial_hash=standard['initial_hash'],saved_substep_peaks=standard['peaks'],
        saved_valid=standard['valid'],saved_success=standard['success']))
    jobs=[(r['label'],r['case_seed'],r['trace_path'],r,str(args.output/(r['label']+'_'+str(r['case_seed'])))) for r in selected]
    contract=dict(seed=226,workers=args.workers,smoke=args.smoke,read_only_contact_wrapper=True,
        unchanged_original_mj_step_calls=True,no_new_strategy=True,not_training=True,
        reserved_qualification_never_loaded=True,no_state_reset_during_recovery=True,
        contact_records_after_original_step_not_forward_reconstruction=True,
        hashes={str(f):digest(f) for f in [Path(__file__),previous.program.SCENE,previous.program.REFERENCE,geometry.OUTPUT/'results.json']})
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for r in pool.map(replay,jobs):
            rows.append(r)
            (args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),last=r),indent=2))
            print('R126_REPLAY',r['label'],r['case_seed'],r['deepest_substep'],flush=True)
    result=dict(rows=rows,smoke=args.smoke,all_full_trace_bitwise_equal=all(all(r['full_control_trace_bitwise_equal'].values()) for r in rows),
        no_new_model_training=True,not_causal_proof=True,independent_qualification_run=False,
        simulation_only=True,hardware_readiness=False,full_task_completed=False)
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    print('R126_TERMINAL',len(rows),flush=True)


if __name__=='__main__':main()
