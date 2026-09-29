"""Matched native closed-loop BC validation, actual 50 Hz policy, no autoreset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diagnostics.getup_native_curriculum import NativeEpisode, at_goal, physical_safe
from diagnostics.getup_independent_native import DT, SLEW, digest
from diagnostics.getup_teacher_probe_r12 import normalized_target


_context=None


def init_worker(experiment):
    global _context
    experiment=Path(experiment);c=json.loads((experiment/'controller_contract.json').read_text())
    e=NativeEpisode(c['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',c['starts_rad'],1,.55)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    actor=ort.InferenceSession(str(experiment/'final.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    _context=e,actor


def evaluate(job):
    kind,seed,steps=job;e,actor=_context
    e.rng=np.random.default_rng(seed);e.reset();sim=e.sim
    initial_hash=hashlib.sha256(sim.data.qpos.tobytes()+sim.data.qvel.tobytes()).hexdigest()
    safe=True;tail=0;success4=False;all_goal=True;trace=[];before=sim.prev.copy();max_slew=0.
    for step in range(steps):
        action=normalized_target(sim,sim.home) if kind=='home' else actor.run(None,{'obs':sim.observation()[None]})[0][0]
        if action.shape!=(14,) or not np.isfinite(action).all() or np.abs(action).max()>1.+1e-6:
            raise RuntimeError('invalid exported actor action')
        target=(sim.lower+sim.upper)/2+np.clip(action,-1.,1.)*(sim.upper-sim.lower)/2
        applied=sim.step_target(target)
        max_slew=max(max_slew,float(np.abs(applied-before).max()/DT));before=applied.copy()
        if max_slew>SLEW+1e-6:raise RuntimeError('slew violated')
        m=sim.measure();safe=safe and physical_safe(sim,m);goal=at_goal(m)
        tail=tail+1 if goal else 0
        if step==199:success4=bool(safe and tail>=100)
        if step>=200:all_goal=all_goal and goal
        if step%25==0:
            trace.append(dict(time_s=(step+1)*DT,height=m['height_m'],up_z=m['up_z'],goal=goal))
        if not physical_safe(sim,m) or m['up_z']<.45:break
    completed=step+1
    return dict(kind=kind,seed=seed,steps=completed,initial_hash=initial_hash,
        success=bool(completed==steps and safe and tail>=100),success_4s=success4,
        continuous_goal_after4s=bool(steps>200 and completed==steps and success4 and all_goal),
        safe=safe,max_slew_rad_s=max_slew,final=m,trace=trace)


def run_block(pool,seeds,steps):
    rows=list(pool.map(evaluate,[(kind,s,steps) for kind in ('home','student') for s in seeds]))
    for seed in seeds:
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:
            raise RuntimeError('paired state mismatch')
    summary={kind:dict(successes=sum(r['success'] for r in rows if r['kind']==kind),runs=len(seeds),
        full_duration_runs=sum(r['steps']==steps for r in rows if r['kind']==kind),
        safe_runs=sum(r['safe'] for r in rows if r['kind']==kind)) for kind in ('home','student')}
    h={r['seed']:r['success'] for r in rows if r['kind']=='home'}
    new=[r['seed'] for r in rows if r['kind']=='student' and r['success'] and not h[r['seed']]]
    lost=[r['seed'] for r in rows if r['kind']=='student' and not r['success'] and h[r['seed']]]
    return dict(summary=summary,results=rows,new_success_seeds=new,new_failure_seeds=lost,
        actual_control_steps=sum(r['steps'] for r in rows),seconds=steps*DT)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    path=args.experiment/'closed_loop_evaluation.json'
    if path.exists():raise FileExistsError(path)
    if not(args.experiment/'checkpoint_audit.json').exists():raise RuntimeError('export audit missing')
    c=json.loads((args.experiment/'controller_contract.json').read_text())
    demo_seeds=[r['seed'] for r in c['teacher_provenance']]
    short=list(range(330000,330100));long=list(range(340000,340020))
    if set(demo_seeds)&set(short+long):raise RuntimeError('evaluation seed contamination')
    report=dict(policy_period_s=DT,policy_action_repeat=1,motor_target_period_s=DT,
        paired_initial_states_verified=True,auto_resets=0,simulation_only=True,hardware_readiness=False,
        stage='near-standing recovery ONLY; NOT full fallen get-up',default_controller_replaced=False,
        hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'final.onnx')})
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(args.experiment),)) as pool:
        for name,seeds,steps in [('demonstration_starts',demo_seeds,200),('independent_100',short,200),('independent_30s',long,1500)]:
            block=run_block(pool,seeds,steps);report[name]=block
            path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('CLOSED LOOP:',name,json.dumps(block['summary']),'NEW',len(block['new_success_seeds']),'LOST',len(block['new_failure_seeds']),flush=True)
    report['complete']=True
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
