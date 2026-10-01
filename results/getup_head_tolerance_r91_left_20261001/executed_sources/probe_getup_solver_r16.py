"""Numerical convergence check on paired native initial physical states.

Only solver iterations differ; this is not a physical motor/ground adjustment.
No successful outcome here is counted as a trained neural recovery skill.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode,at_goal,physical_safe
from diagnostics.getup_independent_native import digest


def evaluate(job):
    contract_path,seed,iterations=job
    c=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(c['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    c['starts_rad'],seed,.55)
    sim=e.sim
    original_iterations=int(sim.model.opt.iterations)
    initial_hash=hashlib.sha256(sim.data.qpos.tobytes()+sim.data.qvel.tobytes()).hexdigest()
    sim.model.opt.iterations=iterations
    # Recompute constraints, but never edit the dynamic pose or velocity.
    mujoco.mj_forward(sim.model,sim.data)
    stable_tail=0; safe=True; trace=[]; max_force=0.; max_contact=0; actual_iterations=[]
    for step in range(200):
        sim.step_target(sim.home)
        m=sim.measure();safe=safe and physical_safe(sim,m)
        stable_tail=stable_tail+1 if at_goal(m) else 0
        max_force=max(max_force,float(np.max(np.abs(sim.data.actuator_force))))
        max_contact=max(max_contact,int(sim.data.ncon))
        actual_iterations.append(int(np.asarray(sim.data.solver_niter).max()))
        if step<40 or step%10==0:
            trace.append(dict(step=step+1,up=m['up_z'],height=m['height_m'],
                         foot_load=m['foot_load_fraction'],ncon=int(sim.data.ncon),
                         solver_niter=actual_iterations[-1],floor_penetration=m['floor_penetration_m'],
                         self_penetration=m['self_penetration_m']))
        if not physical_safe(sim,m) or m['up_z']<.45: break
    return dict(seed=seed,iterations=iterations,compiled_original_iterations=original_iterations,
                initial_hash=initial_hash,steps=step+1,success=bool(stable_tail>=100 and safe),
                safe=safe,final=m,trace=trace,max_sampled_force_nm=max_force,max_contacts=max_contact,
                max_actual_solver_iterations=max(actual_iterations),mean_actual_solver_iterations=float(np.mean(actual_iterations)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=20)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    contract=args.experiment/'controller_contract.json'
    jobs=[(str(contract),190000+s,n) for n in (1,20,50) for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=4) as pool: rows=list(pool.map(evaluate,jobs,chunksize=1))
    for seed in range(190000,190000+args.seeds):
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('paired state mismatch')
    summary={n:dict(successes=sum(r['success'] for r in rows if r['iterations']==n),runs=args.seeds,
                   safe_runs=sum(r['safe'] for r in rows if r['iterations']==n)) for n in (1,20,50)}
    report=dict(results=rows,summary=summary,paired_initial_states=True,
                physical_parameters_unchanged=True,simulation_only=True,hardware_readiness=False,
                warning='Numerical convergence test, not training or full fallen recovery.',script_sha256=digest(__file__))
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SOLVER COMPARISON:',json.dumps(summary),flush=True)
    print('COMPILED ITERATIONS:',rows[0]['compiled_original_iterations'],flush=True)


if __name__=='__main__':main()
