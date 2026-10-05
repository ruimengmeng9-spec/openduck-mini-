"""Passive unchanged replay: relate causal sensor FK to original substep peaks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import probe_getup_common_prefix_r136 as prefix
from diagnostics import audit_getup_sensor_geometry_r137 as geometry
from diagnostics import probe_getup_causal_acceleration_r140 as acceleration
from diagnostics.getup_reference_env_r100 import native_observation
from diagnostics.validate_getup_fullpath_r27 import StrictSim
from diagnostics.getup_independent_native import digest

ROOT=geometry.ROOT
OUTPUT=ROOT/'outputs/getup_passive_risk_r141_left_20261006'


def replay(job):
    label,program,case,source,directory=job;source=Path(source)
    prefix.init_worker(10 if label in ('delayed','standard') else 0)
    with np.load(geometry.R133/'frozen_programs.npz',allow_pickle=False) as z:gains=z['gains'][program].copy()
    if label in ('snapshot','standard'):gains=np.zeros(6)
    old=json.loads((source/'result.json').read_text())
    with np.load(source/'trajectory.npz',allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
    original_clear=StrictSim.clear_audit;original_step=mujoco.mj_step
    state={};records=[];crossings=[];decisions=[]
    def cleared(sim):
        value=original_clear(sim);state.clear();state.update(sim=sim,fk=geometry.SensorGeometry(sim.model),past=None)
        records.clear();crossings.clear();decisions.clear();return value
    def recorded(m,d,*args,**kwargs):
        active=state.get('sim') is not None and d is state['sim'].data
        if active:
            sim=state['sim'];index=len(records);k=index//10;sub=index%10
            pre_q=d.qpos.copy();pre_v=d.qvel.copy()
            if sub==0 and k<400:
                sensor=np.concatenate([native_observation(sim),np.zeros(5)]).astype(np.float32)
                fk=state['fk'];risk=fk.risk(sensor,.02)
                accel=None if state['past'] is None else fk.from_pose(acceleration.predicted_pose(fk,sensor,state['past'])).tolist()
                decisions.append(dict(control=k,actual_native50=sensor[:50].tolist(),velocity_geometry=risk.tolist(),acceleration_geometry=accel))
                state['past']=sensor.copy()
        value=original_step(m,d,*args,**kwargs)
        if active:
            pairs=[c for c in d.contact if sim.floor not in c.geom]
            c=min(pairs,key=lambda c:c.dist) if pairs else None
            ids=(-1,-1) if c is None else tuple(sorted(map(int,c.geom)))
            pen=0. if c is None else max(0.,-float(c.dist))
            records.append((index,k,sub,float(d.time),pen,*ids))
            if pen>=.004:
                crossings.append(dict(control_index=k,substep=sub,post_step_time=float(d.time),self_penetration_m=pen,
                    geom_ids=list(ids),pre_step_qpos=pre_q.tolist(),pre_step_qvel=pre_v.tolist(),post_step_qpos=d.qpos.tolist()))
        return value
    StrictSim.clear_audit=cleared;mujoco.mj_step=recorded
    try:row=prefix.local.evaluate((gains.tolist(),case,True,directory,not np.any(gains)))
    finally:StrictSim.clear_audit=original_clear;mujoco.mj_step=original_step
    dest=Path(directory)
    with np.load(dest/'trajectory.npz',allow_pickle=False) as z:equality={k:bool(np.array_equal(z[k],v)) for k,v in stored.items()}
    assert all(equality.values()),(label,program,case,equality)
    assert row['initial_hash']==old['initial_hash'] and row['peaks']==old['peaks'] and row['valid']==old['valid'] and row['success']==old['success']
    assert len(records)==10*row['controls']
    # Native observations actually read on the untouched simulator must match
    # saved precontrol sensors exactly; FK calls cannot alter dynamics.
    for decision in decisions:np.testing.assert_array_equal(decision['actual_native50'],stored['observations'][decision['control'],:50])
    np.savez_compressed(dest/'substep_contacts.npz',records=np.array(records),columns=np.array(['index','control','substep','post_step_time','self_penetration_m','geom0','geom1']))
    (dest/'causal_decisions.json').write_text(json.dumps(decisions))
    (dest/'threshold_crossings.json').write_text(json.dumps(crossings,indent=2))
    row.update(label=label,program=program,full_trace_bitwise_equal=equality,original_peaks_identical=True,
        source_sha256=digest(source/'trajectory.npz'),first_crossing=None if not crossings else {k:v for k,v in crossings[0].items() if k not in ('pre_step_qpos','pre_step_qvel','post_step_qpos')},
        passive_original_step_count=len(records),no_root_edits_during_recovery=True,no_policy_changes=True)
    (dest/'result.json').write_text(json.dumps(row,indent=2));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for module in (Path(__file__),Path(geometry.__file__),Path(acceleration.__file__),Path(prefix.__file__),Path(prefix.local.__file__)):
        shutil.copy2(module,sources/module.name)
    selected=geometry.selected_jobs(False)
    if args.smoke:selected=[next(j for j in selected if j[0]=='delayed'),next(j for j in selected if j[0]=='standard')]
    jobs=[(*j[:-1],str(j[-1]),str(args.output/(j[0]+'_p'+str(j[1])+'_'+str(j[2])))) for j in selected]
    rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(replay,jobs):
            rows.append(row);(args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),last=row),indent=2))
            print('R141_PASSIVE_REPLAY',row['label'],row['case_seed'],row['first_crossing'],flush=True)
    (args.output/'results.json').write_text(json.dumps(dict(smoke=args.smoke,rows=rows,all_full_trace_bitwise_equal=True,
        no_new_model_training=True,no_success_relabeling=True,unchanged_control_physics_and_acceptance=True,
        independent_qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(f):digest(f) for f in [geometry.SCENE,*sources.iterdir()]}),indent=2))
    print('R141_TERMINAL',len(rows),flush=True)


if __name__=='__main__':main()
