"""R83 offline leave-one-out clock-selection audit; no new physical trial.

Only the shared pre-intervention IMU observation is a policy input. Oracle
union success is explicitly not a causal selector or qualification result.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from diagnostics.getup_independent_native import digest


def choose(query, x, y, k):
    if len(x) < k or x.ndim != 2 or x.shape[1] != 4 or y.shape[0] != len(x):
        raise ValueError('Invalid training-only selector data')
    scale = np.maximum(x.std(axis=0), [.01,.01,.02,.02])
    distance = np.linalg.norm((x - query) / scale, axis=1)
    neighbors = np.argsort(distance, kind='stable')[:k]
    weights = 1. / (.1 + distance[neighbors])
    score = np.sum(y[neighbors] * weights[:,None],axis=0) / weights.sum()
    return int(np.argmax(score))  # Stable tie preference includes baseline first.


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Need fresh output')
    report=json.loads((args.input/'results.json').read_text())
    grid=report['training_grid']; seeds=[r['seed'] for r in grid[0]['cases']]
    if len(grid)!=9 or len(seeds)!=24 or not all(len(g['cases'])==24 for g in grid):
        raise RuntimeError('Need fully evaluated nine-arm training grid')
    x=np.stack([np.load(args.input/f'grid_00/case_{s}.npz')['imu_error'][25] for s in seeds])
    y=np.array([[g['cases'][i]['success'] for g in grid] for i in range(len(seeds))])
    for arm in range(9):
        for i,s in enumerate(seeds):
            with np.load(args.input/f'grid_{arm:02d}/case_{s}.npz') as z:
                if not np.allclose(z['imu_error'][25],x[i],atol=1e-10,rtol=0):
                    raise RuntimeError('Selector observation is not shared before intervention')
    rows=[]
    for k in (1,3,5,7):
        predicted=[]
        for i,s in enumerate(seeds):
            mask=np.arange(len(seeds))!=i
            arm=choose(x[i],x[mask],y[mask],k)
            predicted.append({'seed':s,'selected_arm':arm,'success':bool(y[i,arm]),
                              'baseline_success':bool(y[i,0])})
        rows.append({'neighbors':k,'successes':sum(v['success'] for v in predicted),
                     'predictions':predicted})
    args.output.mkdir()
    np.savez_compressed(args.output/'training_only_dataset.npz',features=x,arm_success=y,seeds=seeds)
    result={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
            'new_rollouts':False,'heldout_seeds_used':False,'oracle_not_policy':True,
            'source_sha256':digest(__file__),'input_sha256':digest(args.input/'results.json'),
            'baseline_successes':int(y[:,0].sum()),'oracle_union':int(y.any(axis=1).sum()),
            'leave_one_out':rows,'promotion_allowed':False}
    # Threshold is a development screen, never a relaxation of physical acceptance.
    result['promotion_allowed']=max(r['successes'] for r in rows)>result['baseline_successes']
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    print('ORACLE_NOT_POLICY',result['oracle_union'],'BASE',result['baseline_successes'],
          'LOOCV',[(r['neighbors'],r['successes']) for r in rows],
          'PROMOTE',result['promotion_allowed'],flush=True)


if __name__=='__main__': main()
