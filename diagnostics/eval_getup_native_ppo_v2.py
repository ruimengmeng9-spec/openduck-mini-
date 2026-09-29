"""Held-out native replay of initial/final actors and a hold-low control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_independent_native import digest


def evaluate_case(job):
    directory,kind,seed,tilt=job
    directory=Path(directory)
    contract=json.loads((directory/'controller_contract.json').read_text())
    root=Path('/data/shijinsheng/open_duck')
    episode=NativeEpisode(contract['scene_path'],root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                          contract['starts_rad'],seed,tilt)
    initial=episode.initial
    if kind=='hold_low':
        sim=episode.sim
        constant=(sim.prev-(sim.lower+sim.upper)/2)/((sim.upper-sim.lower)/2)
        session=None
    else:
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(directory/f'{kind}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    for _ in range(200):
        observation=episode.sim.observation()
        action=constant if session is None else session.run(None,{'obs':observation[None]})[0][0]
        _,_,done,_,_,info=episode.step(action)
        if done:
            return dict(case=kind,seed=seed,tilt_max_rad=tilt,initial_up_z=initial['up_z'],**info)
    raise RuntimeError('episode did not finish in declared 200-step window')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=74000)
    p.add_argument('--tilt-max',type=float,default=.25)
    p.add_argument('--output-name',default='evaluation.json')
    args=p.parse_args()
    jobs=[(str(args.experiment),kind,args.seed_base+seed,args.tilt_max) for kind in ('hold_low','initial','final') for seed in range(args.seeds)]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate_case,jobs,chunksize=1):
            rows.append(row)
    aggregates={kind:dict(successes=sum(r['success'] for r in rows if r['case']==kind),runs=args.seeds,
                         safe_runs=sum(r['safe'] for r in rows if r['case']==kind)) for kind in ('hold_low','initial','final')}
    report=dict(stage='mild tilt / loaded low posture -> goal stand; NOT full fallen recovery',
                results=rows,aggregates=aggregates,simulation_only=True,hardware_readiness=False,seed_base=args.seed_base,
                hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'initial.onnx',args.experiment/'final.onnx')})
    (args.experiment/args.output_name).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HELD-OUT EVALUATION:',json.dumps(aggregates),flush=True)


if __name__=='__main__':
    main()
