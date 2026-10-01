"""Coupled-noise comparison of a home-centred policy before any PPO update."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_macro_curriculum_r15 import advance_macro
from diagnostics.train_getup_residual_r18 import residual_decoder
from diagnostics.getup_independent_native import digest


def evaluate(job):
    contract_path,seed,std=job;c=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(c['scene_path'],
                    '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    c['starts_rad'],seed,.55)
    sim=e.sim
    initial_hash=hashlib.sha256(sim.data.qpos.tobytes()+sim.data.qvel.tobytes()).hexdigest()
    rng=np.random.default_rng(seed+777000)
    for _ in range(200):
        latent=std*rng.normal(size=14)
        action=residual_decoder(latent,sim.home,sim.lower,sim.upper,.15)
        _,_,done,_,_,info,_=advance_macro(e,action,5,.99**.2)
        if done:return dict(seed=seed,std=std,initial_hash=initial_hash,**info)
    raise RuntimeError('episode exceeded length')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    jobs=[(str(args.experiment/'controller_contract.json'),245000+s,std) for std in (0.,.03,.3) for s in range(20)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate,jobs,chunksize=1))
    for seed in range(245000,245020):
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('paired start mismatch')
    summary={std:dict(successes=sum(r['success'] for r in rows if r['std']==std),runs=20,
                      full_duration_runs=sum(r['steps']==200 for r in rows if r['std']==std)) for std in (0.,.03,.3)}
    report=dict(summary=summary,results=rows,paired_initial_states_verified=True,
                coupled_standard_normal_draws=True,simulation_only=True,training=False,
                hardware_readiness=False,script_sha256=digest(__file__))
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HOME NOISE COMPARISON:',json.dumps(summary),flush=True)


if __name__=='__main__':main()
