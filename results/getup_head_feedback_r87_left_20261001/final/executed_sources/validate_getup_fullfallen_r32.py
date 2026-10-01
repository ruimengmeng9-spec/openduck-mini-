"""Independent actual-fallen recovery plus continuous 30s loaded standing.

No reset, snapshot restore, root edit or external force within any trial.
Discovery tests and final held-out acceptance use distinct seed namespaces.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

from diagnostics.getup_fullfallen_contract_r32 import POSES, DECISION_CONTROLS, decode_action, accepted, completion_summary
from diagnostics.getup_fullfallen_env_r32 import observation
from diagnostics.getup_independent_native import DT, digest
from diagnostics.train_getup_fullpath_r27 import state_hash, save_trace
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def trial(job):
    root,actor_path,pose,seed,recovery_s,hold_s,trace_path=job
    root=Path(root)
    sim=StrictSim(root/'training/getup_decomposed_r4/model/scene.xml',
                  root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    actor=ort.InferenceSession(str(actor_path),sess_options=options,providers=['CPUExecutionProvider'])
    if actor.get_inputs()[0].shape[-1]!=50 or actor.get_outputs()[0].shape[-1]!=14:
        raise ValueError('Independent getup requires 50 observation inputs and 14 normalized targets')
    sim.prepare(pose,seed,True)
    initial=sim.measure();initial_hash=state_hash(sim)
    initial_fallen=bool(initial['up_z']<.5 and initial['torso_contact'])
    if not initial_fallen:raise RuntimeError('Trial did not start on the ground')
    sim.clear_audit()
    controls=0;tail=0;entry=False;trace=[];entry_time=None
    def advance(target,phase):
        nonlocal controls,tail
        sim.step_target(target);controls+=1
        measure=sim.measure()
        tail=tail+1 if measure['stable'] and sim.physical_valid() else 0
        if trace_path:
            trace.append((float(sim.data.time),sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),phase))
        return measure
    max_controls=round(recovery_s/DT)
    while controls<max_controls and sim.physical_valid():
        action=actor.run(None,{actor.get_inputs()[0].name:observation(sim)[None]})[0].reshape(-1)
        target=decode_action(action,sim.lower,sim.upper)
        for _ in range(min(DECISION_CONTROLS,max_controls-controls)):
            advance(target,'getup_actor')
            # 0.2s entry dwell prevents a one-frame pose from entering stand.
            if tail>=10:
                entry=True;entry_time=controls*DT;break
            if not sim.physical_valid():break
        if entry:break
    recovery_end=sim.measure()
    if entry and sim.physical_valid():
        start=sim.prev.copy()
        # 1s blended handoff, then the requested uninterrupted hold interval.
        for i in range(round((1.+hold_s)/DT)):
            mix=min((i+1)*DT,1.);mix=mix*mix*(3.-2.*mix)
            target=(1.-mix)*start+mix*(sim.home+.25*sim.stand_action())
            advance(target,'stand_handoff' if i<50 else 'stand_hold')
            if not sim.physical_valid():break
    valid=sim.physical_valid()
    # Short discovery runs NEVER satisfy the published full task gate.
    success=accepted(initial_fallen,entry,valid,tail,DT) and hold_s>=30.
    row=dict(pose=pose,seed=seed,initial_fallen=initial_fallen,initial=initial,
             initial_hash=initial_hash,entry_reached=entry,entry_time_s=entry_time,
             recovery_end=recovery_end,final=sim.measure(),valid=valid,success=bool(success),
             short_discovery_pass=bool(initial_fallen and entry and valid and tail*DT>=hold_s),
             sustained_tail_s=tail*DT,hold_s=hold_s,control_steps=controls,peaks=sim.peaks.copy())
    if trace_path:save_trace(Path(trace_path),trace)
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--discovery',action='store_true',help='Four short development trials; NEVER final completion')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    # Freeze the selected actor before trials. Final seeds cannot select a model.
    frozen=args.output/'tested_actor.onnx';shutil.copy2(args.actor,frozen)
    (args.output/'executed_validator.py').write_bytes(Path(__file__).read_bytes())
    base=1100000 if args.discovery else 1300000
    count=1 if args.discovery else 20
    hold=4. if args.discovery else 30.
    jobs=[(str(args.root),str(frozen),pose,base+1000*k+i,12.,hold,
           str(args.output/f'{pose}_seed_{base+1000*k+i}.npz') if i==0 else None)
          for k,pose in enumerate(POSES) for i in range(count)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(trial,jobs,chunksize=1):
            rows.append(row);print(json.dumps(row),flush=True)
            (args.output/'partial_results.json').write_text(json.dumps(rows,indent=2))
    summary,completed=completion_summary(rows)
    result=dict(summary=summary,results=rows,full_task_completed=bool(completed and not args.discovery),
        discovery_only=args.discovery,simulation_only=True,hardware_readiness=False,
        default_controller_replaced=False,actor_sha256=digest(frozen),validator_sha256=digest(__file__),
        seed_base=base,final_gate='>=18/20 each actual fallen orientation, strict loaded home stance continuously >=30s',
        policy_decision_controls=DECISION_CONTROLS,motor_dt_s=DT,root_edits_after_initialization=0)
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    print('FULLFALLEN_ACCEPTANCE',json.dumps(summary),flush=True)
    print('STRICT_FULL_TASK_COMPLETED',result['full_task_completed'],flush=True)


if __name__=='__main__':main()
