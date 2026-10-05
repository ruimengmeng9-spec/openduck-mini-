"""Bounded R100 training, development selection then untouched left-side gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

from diagnostics.getup_reference_env_r100 import TRAIN,full_trial
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
SCENE=ROOT/'training/getup_decomposed_r4/model/scene.xml'
STAND=ROOT/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
REFERENCE=ROOT/'outputs/getup_path_audit_r99_20261005/frozen_path.npz'


def execute(args,log):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-u',*args],cwd=PROJECT,env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def trial(job):
    model,seed,file=job
    weights=None
    if model:
        with np.load(model,allow_pickle=False) as data:weights={k:data[k].copy() for k in data.files}
    return full_trial(str(SCENE),str(STAND),REFERENCE,weights,seed,file)


def evaluate(model,seeds,output,workers):
    output.mkdir(exist_ok=False)
    jobs=[(str(model) if model else None,seed,str(output/f'case_{seed}.npz')) for seed in seeds]
    import multiprocessing as mp
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        rows=[]
        for row in pool.map(trial,jobs):
            rows.append(row);print('R100_CASE',str(output.name),json.dumps(row),flush=True)
    report=dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
        nominal_success=next((r['success'] for r in rows if r['case_seed'] is None),None),
        trials=sum(r['case_seed'] is not None for r in rows),physical_failures=sum(not r['valid'] for r in rows),
        model_sha256=digest(model) if model else None,reference_sha256=digest(REFERENCE),
        simulation_only=True,hardware_readiness=False,full_task_completed=False)
    (output/'results.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.iterations!=64:p.error('Frozen experiment budget is 64 iterations')
    args.output.mkdir(exist_ok=False)
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'train_getup_reference_r100.py','getup_reference_env_r100.py','test_getup_reference_r100.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    contract=dict(experiment='R100',hypothesis='Full-path prior plus causal state residuals at 50Hz may improve early fallen deviations; 6s 10Hz standalone PPO lacked coverage',
        train_seeds=list(TRAIN),nominal_training_probability=.1,development_seeds=list(TRAIN),
        qualification_seeds=list(range(3160000,3160040)),
        development_gate='nominal retained, >=22/24 and improvement over exact zero-residual baseline',
        left_qualification_gate='>=18/20 in each of two disjoint 20-seed groups, strict continuous 30s, valid substep physics',
        other_poses_not_trained=True,full_task_completed=False,simulation_only=True,hardware_readiness=False,
        cpu_workers=args.workers,policy_device='cpu',iterations=args.iterations,envs=8,horizon=256,seed=200,
        hashes={p.name:digest(p) for p in sources.iterdir()},scene_sha256=digest(SCENE),standing_actor_sha256=digest(STAND),reference_sha256=digest(REFERENCE))
    (args.output/'experiment.json').write_text(json.dumps(contract,indent=2))
    execute(['-m','unittest','diagnostics.test_getup_reference_r100','diagnostics.test_getup_fullfallen_r32',
             'diagnostics.test_getup_temporal_noise_r33','-v'],args.output/'tests.log')
    execute(['-m','diagnostics.train_getup_reference_r100','--output',str(args.output/'smoke'),
             '--iterations','2','--horizon','16','--envs','4','--workers','2'],args.output/'smoke.log')
    baseline=evaluate(None,[None,*TRAIN],args.output/'baseline_development',args.workers)
    execute(['-m','diagnostics.train_getup_reference_r100','--output',str(args.output/'training'),
             '--iterations','64','--horizon','256','--envs','8','--workers',str(args.workers),
             '--seed','200','--learning-rate','1e-5'],args.output/'training.log')
    outcomes={}
    for step in (16,32,48,64):
        name=f'checkpoint_{step:04d}'
        outcomes[name]=evaluate(args.output/'training'/f'{name}.npz',[None,*TRAIN],
                                args.output/f'{name}_development',args.workers)
        (args.output/'comparison_partial.json').write_text(json.dumps(outcomes,indent=2))
    retained={k:v for k,v in outcomes.items() if v['nominal_success']}
    chosen=max(retained,key=lambda k:retained[k]['successes']) if retained else None
    report=dict(baseline=baseline,outcomes=outcomes,selected_on_development_only=chosen,
                full_task_completed=False,simulation_only=True,hardware_readiness=False)
    if chosen and outcomes[chosen]['successes']>=22 and outcomes[chosen]['successes']>baseline['successes']:
        frozen=args.output/'frozen_candidate.npz';shutil.copy2(args.output/'training'/f'{chosen}.npz',frozen)
        candidate=evaluate(frozen,list(range(3160000,3160040)),args.output/'independent_candidate',args.workers)
        independent_base=evaluate(None,list(range(3160000,3160040)),args.output/'independent_baseline',args.workers)
        groups=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
        report.update(independent_candidate=candidate,independent_baseline=independent_base,
                      left_stage_passed=all(x>=18 for x in groups),group_successes=groups,
                      remaining='left expanded noise/delay validation and right/prone/supine development')
        for a,b in zip(candidate['rows'],independent_base['rows']):
            if a['initial_hash']!=b['initial_hash']:raise RuntimeError('Paired initial state mismatch')
    else:
        report.update(left_stage_passed=False,qualification_not_run_reason='Development gate not met; preserve complete failed traces, no automatic blind rerun')
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('R100_TERMINAL',json.dumps({k:v for k,v in report.items() if k not in ('baseline','outcomes','independent_candidate','independent_baseline')}),flush=True)


if __name__=='__main__':main()
