"""Matched deterministic ONNX replay at declared policy action-repeat cadence."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_macro_curriculum_r15 import advance_macro
from diagnostics.getup_teacher_probe_r12 import normalized_target
from diagnostics.getup_independent_native import digest


def evaluate(job):
    directory,kind,seed,tilt,repeat=job
    directory=Path(directory); contract=json.loads((directory/'controller_contract.json').read_text())
    e=NativeEpisode(contract['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    contract['starts_rad'],seed,tilt)
    initial_hash=hashlib.sha256(e.sim.data.qpos.tobytes()+e.sim.data.qvel.tobytes()).hexdigest()
    if kind!='home':
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        actor=ort.InferenceSession(str(directory/f'{kind}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    for decision in range(200):
        obs=e.sim.observation()
        action=normalized_target(e.sim,e.sim.home) if kind=='home' else actor.run(None,{'obs':obs[None]})[0][0]
        _,_,done,_,_,info,n=advance_macro(e,action,repeat,.99**(1/repeat))
        if done: return dict(kind=kind,seed=seed,repeat=repeat,policy_decisions=decision+1,initial_hash=initial_hash,**info)
    raise RuntimeError('native episode length exceeded')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=180000)
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--tilt-max',type=float,default=.55)
    p.add_argument('--output-name',default='evaluation_macro.json')
    args=p.parse_args()
    output=args.experiment/args.output_name
    if output.exists():raise FileExistsError(output)
    jobs=[(str(args.experiment),kind,args.seed_base+s,args.tilt_max,args.repeat)
          for kind in ('home','initial','final') for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=4) as pool: rows=list(pool.map(evaluate,jobs,chunksize=1))
    for seed in range(args.seed_base,args.seed_base+args.seeds):
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1: raise RuntimeError('paired start mismatch')
    summary={kind:dict(successes=sum(r['success'] for r in rows if r['kind']==kind),runs=args.seeds,
                      full_duration_runs=sum(r['steps']==200 for r in rows if r['kind']==kind),
                      safe_runs=sum(r['safe'] for r in rows if r['kind']==kind)) for kind in ('home','initial','final')}
    report=dict(results=rows,summary=summary,seed_base=args.seed_base,policy_action_repeat=args.repeat,
                tilt_max_rad=args.tilt_max,paired_initial_states_verified=True,simulation_only=True,hardware_readiness=False,
                stage='near-standing recovery only, no full fallen recovery',
                hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'initial.onnx',args.experiment/'final.onnx')})
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('MACRO POLICY EVALUATION:',json.dumps(summary),flush=True)


if __name__=='__main__': main()
