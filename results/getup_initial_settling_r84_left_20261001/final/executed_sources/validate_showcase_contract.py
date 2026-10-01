"""Paired multi-seed regression of the recording execution contract."""
import contextlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
from diagnostics.record_motion_showcase import rollout


def run(task):
    seed,legacy,folder=task
    out=Path(folder)/f'seed_{seed}_{"legacy" if legacy else "v2"}'
    out.mkdir(exist_ok=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _,_,summary=rollout(Path('/data/shijinsheng/open_duck'),out,seed,legacy)
    return summary


if __name__=='__main__':
    out=Path('/data/shijinsheng/open_duck/outputs/showcase_contract_v2_regression')
    out.mkdir(exist_ok=False)
    tasks=[(seed,legacy,str(out)) for seed in range(1600,1610) for legacy in (True,False)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        rows=list(pool.map(run,tasks))
        summary=dict(simulation_only=True,paired=True,weights_changed=False,rows=rows)
        (out/'results.json').write_text(json.dumps(summary,indent=2))
        for mode in ['legacy_shared_filter','per_skill_v2']:
            selected=[r for r in rows if r['execution_contract']==mode]
            forward=[p for r in selected for p in r['phases'] if p['phase']=='Forward']
            backward=[p for r in selected for p in r['phases'] if p['phase']=='Backward']
            stops=[p for r in selected for p in r['phases'] if p['phase']=='Stop']
            print(json.dumps(dict(mode=mode,completed=sum(r['completed'] for r in selected),runs=len(selected),
                forward_avg_mps=sum(p['axial_speed_mps'] for p in forward)/len(forward),
                forward_tail_range=[min(p['mean_local_forward_last2s_mps'] for p in forward),max(p['mean_local_forward_last2s_mps'] for p in forward)],
                backward_avg_mps=sum(p['axial_speed_mps'] for p in backward)/len(backward),
                stop_last1s_max_drift_m=max(p['last_1s_drift_m'] for p in stops))),flush=True)
