"""Frozen global feedback programs, complete tests before set-valued teaching.

Offline greedy selection is diagnostic data selection, not a case controller.
Every selected program is run on every known original development start.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics.getup_independent_native import digest

ROOT=local.ROOT
OUTPUT=ROOT/'outputs/getup_complementary_feedback_r133_left_20261006'


def selected_programs(audit):
    eligible=[r for r in audit['groups'] if r['physical_failures']==0]
    sets={g['directory']:{r['case_seed'] for r in json.loads((Path(g['directory'])/'results.json').read_text())['rows'] if r['success'] and r['case_seed'] is not None} for g in eligible}
    zero=[g for g in eligible if not any(g['gains'])];assert len(zero)==1
    chosen=zero.copy();covered=sets[zero[0]['directory']].copy();remaining=[g for g in eligible if any(g['gains'])]
    for _ in range(4):
        winner=max(remaining,key=lambda g:(len(sets[g['directory']]-covered),g['successes'],g['return_sum']))
        if not (sets[winner['directory']]-covered):break
        chosen.append(winner);covered.update(sets[winner['directory']]);remaining.remove(winner)
    return chosen,sorted(covered)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    source=ROOT/'outputs/getup_local_hip_audit_r131_20261006/results.json'
    chosen,union=selected_programs(json.loads(source.read_text()));assert len(chosen)==4 and len(union)==24
    args.output.mkdir(exist_ok=False);executed=args.output/'executed_sources';executed.mkdir()
    for name in (Path(__file__).name,'test_getup_complementary_feedback_r133.py','launch_getup_complementary_feedback_r133.py',
        'train_getup_local_hip_r130.py','audit_getup_local_hip_r131.py'):
        shutil.copy2(Path(__file__).with_name(name),executed/name)
    gains=np.array([g['gains'] for g in chosen]);np.savez_compressed(args.output/'frozen_programs.npz',gains=gains)
    local.write_json(args.output/'contract.json',dict(smoke=args.smoke,workers=args.workers,initialization_seed=230,
        offline_selection='Greedy short-success coverage among all-physical-valid R130 global proposals; not runtime case selection',
        selected_source_groups=chosen,short_union=union,short_union_not_unified_success=True,
        hypothesis='Different legal local feedback programs rescue different starts; obtain complete success sets before learning a fixed sensor-only choice rule',
        full_controls=2279,control_hz=50,physics_hz=500,entry_deadline_s=12,strict_tail_s=30,
        no_controller_training=True,no_root_truth_or_case_in_control=True,physics_rewards_acceptance_unchanged=True,
        no_mid_episode_root_reset=True,no_unseen_qualification=True,
        hashes={str(p):digest(p) for p in [source,local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE,*executed.iterdir()]}))
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=local.init_worker) as pool:
        reports=[]
        for i,gain in enumerate(gains):
            cases=[None,769000] if args.smoke else [None,*local.prior.program.TRAIN]
            report=local.group(pool,gain,cases,True,args.output/f'program_{i:02d}',i==0)
            reports.append(report);local.write_json(args.output/'progress.json',dict(completed_programs=len(reports),total=len(gains),last=report))
            print('R133_COMPLETE_PROGRAM',i,report['successes'],report['physical_failures'],flush=True)
    cases=[r['case_seed'] for r in reports[0]['rows']]
    initial=[r['initial_hash'] for r in reports[0]['rows']]
    for report in reports:assert [r['initial_hash'] for r in report['rows']]==initial
    success=np.array([[r['success'] for r in report['rows']] for report in reports]).T
    valid=np.array([[r['valid'] for r in report['rows']] for report in reports]).T
    np.savez_compressed(args.output/'offline_success_sets.npz',success=success,valid=valid)
    local.write_json(args.output/'manifest.json',[dict(case_seed=s,initial_hash=h) for s,h in zip(cases,initial)])
    union_cases=[case for case,ok in zip(cases,np.any(success,axis=1)) if case is not None and ok]
    local.write_json(args.output/'results.json',dict(smoke=args.smoke,reports=reports,complete_union_cases=union_cases,
        complete_union_count=len(union_cases),not_unified_controller_success=True,not_independent_qualification=True,
        full_task_completed=False,hardware_readiness=False))
    print('R133_TERMINAL_UNION',len(union_cases),'NOT_UNIFIED_POLICY',flush=True)


if __name__=='__main__':main()
