"""Independent 30-second recovery/hold evaluation; no episode auto-reset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diagnostics.getup_native_curriculum import NativeEpisode,at_goal,physical_safe
from diagnostics.getup_independent_native import DT,digest


def evaluate(job):
    contract_path,kind,model_path,seed=job
    c=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(c['scene_path'],
                    '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    c['starts_rad'],seed,.55)
    sim=e.sim
    initial_hash=hashlib.sha256(sim.data.qpos.tobytes()+sim.data.qvel.tobytes()).hexdigest()
    if model_path:
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        actor=ort.InferenceSession(model_path,sess_options=options,providers=['CPUExecutionProvider'])
    safe=True;stable_tail=0;success_4s=False;all_goal_after4=True;trace=[]
    maximum_force=0.;maximum_slew=0.;before=sim.prev.copy()
    for step in range(1500):
        if step%5==0:
            if model_path:
                action=actor.run(None,{'obs':sim.observation()[None]})[0][0]
                target=(sim.lower+sim.upper)/2+np.clip(action,-1.,1.)*(sim.upper-sim.lower)/2
            else:target=sim.home.copy()
        applied=sim.step_target(target)
        maximum_slew=max(maximum_slew,float(np.max(np.abs(applied-before))/DT));before=applied.copy()
        maximum_force=max(maximum_force,float(np.max(np.abs(sim.data.actuator_force))))
        m=sim.measure();safe=safe and physical_safe(sim,m)
        goal=at_goal(m);stable_tail=stable_tail+1 if goal else 0
        if step==199:success_4s=bool(safe and stable_tail>=100)
        if step>=200:all_goal_after4=all_goal_after4 and goal
        if step<200 and step%10==0 or step%50==0 or step==1499:
            trace.append(dict(time_s=(step+1)*DT,height=m['height_m'],up_z=m['up_z'],goal=goal,
                              linear_speed=m['linear_speed_mps'],angular_speed=m['angular_speed_rad_s'],
                              joint_error=m['joint_home_error_mean_rad'],foot_load=m['foot_load_fraction']))
        if not physical_safe(sim,m) or m['up_z']<.45:break
    steps=step+1
    return dict(kind=kind,seed=seed,initial_hash=initial_hash,steps=steps,seconds=steps*DT,
                success_4s=success_4s,success_30s=bool(steps==1500 and safe and stable_tail>=100),
                goal_continuous_after4s=bool(steps==1500 and success_4s and all_goal_after4),
                safe=safe,final=m,max_sampled_force_nm=maximum_force,max_target_slew_rad_s=maximum_slew,
                trace=trace)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--height',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    contract=args.base/'controller_contract.json'
    expected=json.loads(contract.read_text())
    for folder in (args.height,args.previous):
        c=json.loads((folder/'controller_contract.json').read_text())
        if c['scene_path']!=expected['scene_path'] or c['starts_rad']!=expected['starts_rad']:
            raise RuntimeError('model scene or start qualification mismatch')
    models=dict(home=None,r18=str(args.previous/'final.onnx'),
                r20_base=str(args.base/'final.onnx'),r20_height=str(args.height/'final.onnx'))
    jobs=[(str(contract),kind,model,270000+s) for kind,model in models.items() for s in range(20)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate,jobs,chunksize=1))
    for seed in range(270000,270020):
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('paired state mismatch')
    summary={kind:dict(runs=20,success_4s=sum(r['success_4s'] for r in rows if r['kind']==kind),
                      success_30s=sum(r['success_30s'] for r in rows if r['kind']==kind),
                      continuous_goal_after4s=sum(r['goal_continuous_after4s'] for r in rows if r['kind']==kind),
                      full_duration_runs=sum(r['steps']==1500 for r in rows if r['kind']==kind)) for kind in models}
    report=dict(summary=summary,results=rows,seed_base=270000,seconds=30,policy_action_repeat=5,
                motor_target_period_s=.02,paired_initial_states_verified=True,auto_resets=0,
                simulation_only=True,hardware_readiness=False,stage='near-standing recovery; not full fallen get-up',
                hashes={str(path):digest(path) for path in [Path(__file__),*[Path(x) for x in models.values() if x]]})
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('30 SECOND RECOVERY:',json.dumps(summary),flush=True)


if __name__=='__main__':main()
