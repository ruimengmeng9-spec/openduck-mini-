"""Replay and explain goal gates for failed R15 actors, without state resets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diagnostics.getup_native_curriculum import NativeEpisode,at_goal,physical_safe
from diagnostics.getup_independent_native import digest


def evaluate(job):
    folder,kind,seed=job;folder=Path(folder)
    c=json.loads((folder/'controller_contract.json').read_text())
    e=NativeEpisode(c['scene_path'],
                    '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    c['starts_rad'],seed,.55)
    sim=e.sim
    if kind!='home':
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        actor=ort.InferenceSession(str(folder/'final.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    trace=[];stable_tail=0;safe=True
    for step in range(200):
        if step%5==0:
            if kind=='home':target=sim.home.copy()
            else:
                action=actor.run(None,{'obs':sim.observation()[None]})[0][0]
                target=(sim.lower+sim.upper)/2+np.clip(action,-1.,1.)*(sim.upper-sim.lower)/2
        sim.step_target(target);m=sim.measure()
        safe=safe and physical_safe(sim,m);stable_tail=stable_tail+1 if at_goal(m) else 0
        trace.append(dict(step=step+1,goal=at_goal(m),stable=m['stable'],
                          height=m['height_m'],up=m['up_z'],joint_mean_error=m['joint_home_error_mean_rad'],
                          joint_max_error=m['joint_home_error_max_rad'],angular_speed=m['angular_speed_rad_s'],
                          linear_speed=m['linear_speed_mps'],foot_load=m['foot_load_fraction'],
                          foot_alignment=min(m['foot_up_alignment_to_home']),
                          minimum_foot_force=min(m['foot_normal_forces_n']),
                          requested_mean_home_delta=float(np.mean(np.abs(target-sim.home)))))
        if not physical_safe(sim,m) or m['up_z']<.45:break
    tail=trace[-100:]
    gates=dict(height=lambda x:.155<x['height']<.185,
               joint_mean=lambda x:x['joint_mean_error']<.2,
               joint_max=lambda x:x['joint_max_error']<.45,
               stable=lambda x:x['stable'],loaded_feet=lambda x:x['foot_load']>.9 and x['minimum_foot_force']>.1,
               foot_alignment=lambda x:x['foot_alignment']>.85)
    return dict(kind=kind,seed=seed,steps=len(trace),success=bool(safe and stable_tail>=100),safe=safe,
                final=trace[-1],trace=trace,
                tail_gate_fractions={name:float(np.mean([check(x) for x in tail])) for name,check in gates.items()},
                tail_mean={name:float(np.mean([x[name] for x in tail])) for name in
                           ('joint_mean_error','joint_max_error','angular_speed','linear_speed','requested_mean_home_delta')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    jobs=[(str(args.experiment),kind,180000+s) for kind in ('home','r15_final') for s in range(20)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate,jobs,chunksize=1))
    summary={}
    for kind in ('home','r15_final'):
        group=[r for r in rows if r['kind']==kind];full=[r for r in group if r['steps']==200]
        summary[kind]=dict(successes=sum(r['success'] for r in group),runs=len(group),full_duration_runs=len(full),
                           full_duration_tail_means={key:float(np.mean([r['tail_mean'][key] for r in full]))
                                                     for key in group[0]['tail_mean']})
    report=dict(summary=summary,results=rows,simulation_only=True,hardware_readiness=False,
                script_sha256=digest(__file__),warning='Gate diagnostics, not a proof of a single causal root.')
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GOAL GATES:',json.dumps(summary),flush=True)


if __name__=='__main__':main()
