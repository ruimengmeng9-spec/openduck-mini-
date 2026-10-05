"""Repair historical teacher action interface; retain failed R126 unchanged."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import probe_getup_substep_contacts_r126 as original
from diagnostics import search_getup_case_teachers_r109 as teacher
from diagnostics.train_getup_joint_anchor_r102 import action_for
from diagnostics.getup_independent_native import digest

ROOT=original.ROOT
OUTPUT=ROOT/'outputs/getup_substep_contacts_r126b_left_20261005'


def replay(job):
    execute=original.previous.program.execute_program
    def historical(weights,current,k,profile,knots):
        frozen={n:weights['feedback_'+n] for n in ('hidden0_kernel','hidden0_bias','hidden1_kernel','hidden1_bias','mean_kernel','mean_bias')}
        base=np.zeros(10) if profile==0 else action_for(frozen,current,weights['feedback_anchors'][k],weights['feedback_gains'])
        return np.clip(base+teacher.knot_action(knots,k),-1.,1.)
    if job[0]=='teacher':original.previous.program.execute_program=historical
    try:return original.replay(job)
    finally:original.previous.program.execute_program=execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for file in (Path(__file__),Path(original.__file__),Path(teacher.__file__)):
        shutil.copy2(file,sources/file.name)
    selected=list(json.loads((original.geometry.OUTPUT/'results.json').read_text())['rows'])
    if args.smoke:
        selected=[r for r in selected if (r['label'],r['case_seed']) in [('teacher',773007),('r122_history',773004)]]
        assert len(selected)==2
    standard=next(r for r in json.loads((original.previous.OUTPUT/'results.json').read_text())['summaries']['snapshot']['all']['rows'] if r['case_seed'] is None)
    selected.append(dict(label='r122_snapshot',case_seed=None,trace_path=str(original.previous.OUTPUT/'snapshot/case_None/trajectory.npz'),
        saved_initial_hash=standard['initial_hash'],saved_substep_peaks=standard['peaks'],saved_valid=standard['valid'],saved_success=standard['success']))
    rows=[];jobs=[];reused=[]
    for r in selected:
        name=r['label']+'_'+str(r['case_seed']);before=original.OUTPUT/name;dest=args.output/name
        if not args.smoke and (before/'result.json').exists():
            row=json.loads((before/'result.json').read_text());assert all(row['full_control_trace_bitwise_equal'].values())
            assert row['trace_sha256']==digest(r['trace_path']) and row['initial_hash']==r['saved_initial_hash']
            shutil.copytree(before,dest);rows.append(row);reused.append(name)
        else:jobs.append((r['label'],r['case_seed'],r['trace_path'],r,str(dest)))
    (args.output/'contract.json').write_text(json.dumps(dict(smoke=args.smoke,seed=226,workers=args.workers,
        original_failed_run_preserved=True,historical_teacher_formula_after_home_restored=True,
        no_weakened_parity=True,no_physics_or_control_changes=True,closed_original_replays_reused=reused,
        no_new_model_training=True,reserved_qualification_never_loaded=True,
        hashes={str(f):digest(f) for f in [Path(__file__),Path(original.__file__),Path(teacher.__file__),original.previous.program.SCENE]}),indent=2))
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(replay,jobs):
            rows.append(row)
            (args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(selected),last=row),indent=2))
            print('R126B_REPLAY',row['label'],row['case_seed'],row['deepest_substep'],flush=True)
    result=dict(rows=rows,smoke=args.smoke,reused_original_closed_replays=reused,
        all_full_trace_bitwise_equal=all(all(r['full_control_trace_bitwise_equal'].values()) for r in rows),
        no_new_model_training=True,no_success_relabeling=True,not_causal_proof=True,
        independent_qualification_run=False,simulation_only=True,hardware_readiness=False,full_task_completed=False)
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print('R126B_TERMINAL',len(rows),flush=True)


if __name__=='__main__':main()
