"""Explain stage-1 failures without changing physics or success criteria."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode, at_goal, physical_safe
from diagnostics.getup_independent_native import digest


def run_case(job):
    directory, kind, seed, tilt = job
    directory = Path(directory)
    contract = json.loads((directory/'controller_contract.json').read_text())
    root = Path('/data/shijinsheng/open_duck')
    e = NativeEpisode(contract['scene_path'], root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                      contract['starts_rad'], 66000+seed, tilt)
    sim = e.sim
    if kind == 'final':
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        actor = ort.InferenceSession(str(directory/'final.onnx'), sess_options=options,
                                    providers=['CPUExecutionProvider'])
    rows = [sim.measure()]
    for step in range(200):
        if kind == 'official_stand':
            target = sim.home + .25*sim.stand_action()
        elif kind == 'home':
            target = sim.home
        else:
            a = actor.run(None, {'obs':sim.observation()[None]})[0][0]
            target = (sim.lower+sim.upper)/2 + a*(sim.upper-sim.lower)/2
        sim.step_target(target)
        m = sim.measure()
        m.update(safe=physical_safe(sim,m), goal=at_goal(m), step=step+1)
        rows.append(m)
        if not m['safe'] or m['up_z'] < .45:
            break
    tail = 0
    for m in reversed(rows[1:]):
        if not m['goal']: break
        tail += 1
    final = rows[-1]
    reason = ('unsafe' if not final['safe'] else 'tilted_below_gate' if final['up_z'] < .45
              else 'goal_hold' if tail >= 100 else 'goal_not_sustained')
    return dict(kind=kind,seed=66000+seed,tilt_max_rad=tilt,steps=len(rows)-1,
                success=tail >= 100 and all(m['safe'] for m in rows[1:]),reason=reason,
                sustained_goal_seconds=tail*.02,initial=rows[0],final=final,
                trace=rows[::10]+([final] if (len(rows)-1)%10 else []))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    jobs=[(str(args.experiment),kind,seed,tilt) for kind in ('home','official_stand','final')
          for tilt in (0.,.1,.25) for seed in range(3)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run_case,jobs,chunksize=1))
    summary={f'{kind}@{tilt}':dict(successes=sum(r['success'] for r in rows if r['kind']==kind and r['tilt_max_rad']==tilt),
                                 runs=3,reasons=[r['reason'] for r in rows if r['kind']==kind and r['tilt_max_rad']==tilt])
             for kind in ('home','official_stand','final') for tilt in (0.,.1,.25)}
    report=dict(results=rows,summary=summary,simulation_only=True,hardware_readiness=False,
                hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'final.onnx',
                                                 args.experiment/'controller_contract.json')})
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BASELINE PROBE:',json.dumps(summary),flush=True)


if __name__=='__main__': main()
