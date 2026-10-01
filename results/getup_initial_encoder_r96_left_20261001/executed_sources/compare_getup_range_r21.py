"""Fresh paired tests across R18/R20/R21, including uninterrupted 30s holds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from diagnostics.eval_getup_macro_r15 import evaluate as evaluate_short
from diagnostics.validate_getup_long_r20 import evaluate as evaluate_long
from diagnostics.getup_independent_native import digest


def short_job(job):
    folder,kind,seed=job
    row=evaluate_short((folder,'home' if kind=='home' else 'final',seed,.55,5))
    row['kind']=kind
    return row


def aggregate(rows,kinds,long=False):
    for seed in {r['seed'] for r in rows}:
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('paired state mismatch')
    result={}
    for kind in kinds:
        group=[r for r in rows if r['kind']==kind]
        if long:
            result[kind]=dict(runs=len(group),success_4s=sum(r['success_4s'] for r in group),
                              success_30s=sum(r['success_30s'] for r in group),
                              continuous_goal_after4s=sum(r['goal_continuous_after4s'] for r in group),
                              full_duration_runs=sum(r['steps']==1500 for r in group))
        else:
            baseline={r['seed']:r['success'] for r in rows if r['kind']=='home'}
            result[kind]=dict(runs=len(group),successes=sum(r['success'] for r in group),
                              full_duration_runs=sum(r['steps']==200 for r in group),
                              gained_vs_home=sum(not baseline[r['seed']] and r['success'] for r in group),
                              lost_vs_home=sum(baseline[r['seed']] and not r['success'] for r in group))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    models=dict(home=args.root/'training/getup_range_r21',r18=args.root/'training/getup_residual_ppo_r18',
                r20_base=args.root/'training/getup_alignment_r20_base',
                r20_height=args.root/'training/getup_alignment_r20_height',r21=args.root/'training/getup_range_r21')
    expected=json.loads((models['r21']/'controller_contract.json').read_text())
    for folder in models.values():
        c=json.loads((folder/'controller_contract.json').read_text())
        if c['scene_path']!=expected['scene_path'] or c['starts_rad']!=expected['starts_rad']:
            raise RuntimeError('scene/start mismatch')
        if not (folder/'final.onnx').is_file():raise FileNotFoundError(folder/'final.onnx')
    jobs=[(str(folder),kind,280000+s) for kind,folder in models.items() for s in range(100)]
    with ProcessPoolExecutor(max_workers=4) as pool:short=list(pool.map(short_job,jobs,chunksize=1))
    short_summary=aggregate(short,models)
    common=dict(simulation_only=True,hardware_readiness=False,paired_initial_states_verified=True,
                policy_action_repeat=5,motor_period_s=.02,stage='near-standing; not full fallen recovery',
                hashes={str(f):digest(f) for f in [Path(__file__),Path(__file__).with_name('eval_getup_macro_r15.py'),
                        Path(__file__).with_name('validate_getup_long_r20.py'),
                        *[folder/'final.onnx' for kind,folder in models.items() if kind!='home']]})
    (args.output/'short_100.json').write_text(json.dumps(dict(summary=short_summary,results=short,seed_base=280000,**common),indent=2),encoding='utf-8')
    print('FRESH 100 STARTS:',json.dumps(short_summary),flush=True)
    jobs=[(str(models['r21']/'controller_contract.json'),kind,None if kind=='home' else str(folder/'final.onnx'),300000+s)
          for kind,folder in models.items() for s in range(20)]
    with ProcessPoolExecutor(max_workers=4) as pool:long=list(pool.map(evaluate_long,jobs,chunksize=1))
    long_summary=aggregate(long,models,long=True)
    (args.output/'long_30s.json').write_text(json.dumps(dict(summary=long_summary,results=long,seed_base=300000,
                                                          seconds=30,auto_resets=0,**common),indent=2),encoding='utf-8')
    print('FRESH 30 SECOND HOLD:',json.dumps(long_summary),flush=True)


if __name__=='__main__':main()
