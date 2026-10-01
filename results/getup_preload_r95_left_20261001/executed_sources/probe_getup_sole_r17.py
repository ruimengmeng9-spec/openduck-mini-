"""Paired simulation-only sole topology probe, not a learned recovery skill.

Keep the same initial physical state and motor/ground settings. Swap only
the decomposed TPU sole contact shapes for their original convex mesh geoms.
All remaining decomposed body collision shapes stay enabled.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode, at_goal, physical_safe
from diagnostics.getup_independent_native import digest


def original_soles(model):
    record=[]
    for source_id,name in ((17,'left_foot_bottom_tpu'),(42,'right_foot_bottom_tpu')):
        parts=[g for g in range(model.ngeom)
               if model.geom(g).name.startswith(f'getup_collision_{source_id}_')]
        if not parts:
            raise RuntimeError('expected R4 TPU decomposition geoms are missing')
        original=model.geom(name).id
        # Retain the decomposed geometry's contact parameters, not vendor defaults.
        properties=('geom_friction','geom_solref','geom_solimp','geom_margin',
                    'geom_gap','geom_condim','geom_priority','geom_solmix')
        for key in properties:
            values=getattr(model,key)
            if not all(np.array_equal(values[g],values[parts[0]]) for g in parts):
                raise RuntimeError('sole decomposition contact parameters differ')
            values[original]=values[parts[0]]
        model.geom_contype[original]=model.geom_contype[parts[0]]
        model.geom_conaffinity[original]=model.geom_conaffinity[parts[0]]
        for g in parts:
            model.geom_contype[g]=model.geom_conaffinity[g]=0
        record.append(dict(original=name,disabled_parts=len(parts),
                           friction=model.geom_friction[original].tolist()))
    return record


def evaluate(job):
    contract_path,seed,variant=job
    c=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(c['scene_path'],
                    '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    c['starts_rad'],seed,.55)
    sim=e.sim
    initial_hash=hashlib.sha256(sim.data.qpos.tobytes()+sim.data.qvel.tobytes()).hexdigest()
    unchanged=('body_mass','body_inertia','body_ipos','body_iquat','jnt_range',
               'dof_damping','dof_armature','dof_frictionloss','actuator_forcerange',
               'actuator_ctrlrange','actuator_gainprm','actuator_biasprm')
    before={key:getattr(sim.model,key).copy() for key in unchanged}
    sole_audit=original_soles(sim.model) if variant=='original_sole' else []
    checks={key:bool(np.array_equal(value,getattr(sim.model,key))) for key,value in before.items()}
    if not all(checks.values()):raise RuntimeError('mechanical parameter change')
    mujoco.mj_forward(sim.model,sim.data)
    stable_tail=0; safe=True; trace=[]
    for step in range(200):
        sim.step_target(sim.home)
        metric=sim.measure()
        safe=safe and physical_safe(sim,metric)
        stable_tail=stable_tail+1 if at_goal(metric) else 0
        if step<40 or step%10==0:
            trace.append(dict(step=step+1,up_z=metric['up_z'],height=metric['height_m'],
                              feet=metric['foot_normal_forces_n'],
                              support_margin=metric['com_support_margin_m'],
                              angular_speed=metric['angular_speed_rad_s']))
        if not physical_safe(sim,metric) or metric['up_z']<.45:break
    return dict(seed=seed,variant=variant,initial_hash=initial_hash,steps=step+1,
                success=bool(safe and stable_tail>=100),safe=safe,final=metric,
                trace=trace,sole_audit=sole_audit,unchanged_motor_inertia_checks=checks)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=20)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    jobs=[(str(args.experiment/'controller_contract.json'),200000+s,variant)
          for variant in ('decomposed_sole','original_sole') for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate,jobs,chunksize=1))
    for seed in range(200000,200000+args.seeds):
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:
            raise RuntimeError('paired start mismatch')
    summary={v:dict(successes=sum(r['success'] for r in rows if r['variant']==v),
                    runs=args.seeds,safe_runs=sum(r['safe'] for r in rows if r['variant']==v))
             for v in ('decomposed_sole','original_sole')}
    report=dict(results=rows,summary=summary,paired_initial_states_verified=True,
                geometry_changed=True,training=False,simulation_only=True,hardware_readiness=False,
                limitation='Post-initialization contact-geometry diagnostic, not deployable training.',
                script_sha256=digest(__file__))
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SOLE COMPARISON:',json.dumps(summary),flush=True)


if __name__=='__main__':main()
