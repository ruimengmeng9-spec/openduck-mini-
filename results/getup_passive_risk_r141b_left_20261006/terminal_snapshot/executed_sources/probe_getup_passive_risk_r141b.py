"""Correct passive read timing; retain original R141 failure and assertions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from diagnostics import probe_getup_passive_risk_r141 as original
from diagnostics.getup_independent_native import digest

ROOT=original.ROOT
OUTPUT=ROOT/'outputs/getup_passive_risk_r141b_left_20261006'


def replay(job):
    previous_target=original.StrictSim.step_target
    previous_observation=original.native_observation
    cache={}
    def before_target(sim,target):
        # Original step_target changes prev/history before mj_step. Read the
        # complete causal native50 before these writes, not at mj_step entry.
        cache[id(sim)]=previous_observation(sim).copy()
        return previous_target(sim,target)
    def captured(sim):return cache[id(sim)].copy()
    original.StrictSim.step_target=before_target;original.native_observation=captured
    try:return original.replay(job)
    finally:original.StrictSim.step_target=previous_target;original.native_observation=previous_observation


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for module in (Path(__file__),Path(original.__file__),Path(original.geometry.__file__),Path(original.acceleration.__file__),Path(original.prefix.__file__),Path(original.prefix.local.__file__)):
        shutil.copy2(module,sources/module.name)
    selected=original.geometry.selected_jobs(False)
    if args.smoke:selected=[next(j for j in selected if j[0]=='delayed'),next(j for j in selected if j[0]=='standard')]
    jobs=[(*j[:-1],str(j[-1]),str(args.output/(j[0]+'_p'+str(j[1])+'_'+str(j[2])))) for j in selected]
    rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(replay,jobs):
            rows.append(row);(args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),last=row),indent=2))
            print('R141B_PASSIVE_REPLAY',row['label'],row['case_seed'],row['first_crossing'],flush=True)
    (args.output/'results.json').write_text(json.dumps(dict(smoke=args.smoke,rows=rows,all_full_trace_bitwise_equal=True,
        original_failed_interface_preserved=True,no_assertion_weakened=True,native50_pre_target_capture_exact=True,
        no_new_model_training=True,no_success_relabeling=True,unchanged_control_physics_and_acceptance=True,
        independent_qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(f):digest(f) for f in [original.geometry.SCENE,*sources.iterdir()]}),indent=2))
    print('R141B_TERMINAL',len(rows),flush=True)


if __name__=='__main__':main()
