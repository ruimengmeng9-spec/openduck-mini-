"""Append immutable closed smoke and startup only; never copy open episodes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_intervention_dynamics_r173 as task

def hashes(folder):
    return {str(p.relative_to(folder)):task.digest(p) for p in folder.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=task.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    bundle=task.ROOT/'tmp/getup_intervention_startup_r173_20261009.bundle'
    doc='GETUP_INTERVENTION_DYNAMICS_R173_20261009.md'
    assert not bundle.exists() and not (repo/doc).exists()
    smoke=json.loads((task.SMOKE/'results.json').read_text())
    startup=json.loads((task.OUTPUT/'startup_closed.json').read_text())
    assert smoke['smoke'] and smoke['terminal_result_saved'] and startup['independent_smoke_bitwise_equal']
    task.compare_smoke(task.OUTPUT)
    before_smoke=hashes(task.SMOKE)
    target=repo/'results'/task.SMOKE.name/'terminal_snapshot';assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(task.SMOKE,target)
    shutil.copy2(task.SMOKE.with_suffix('.log'),target/'process.log')
    task.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,smoke_only=True,full_task_completed=False,hardware_readiness=False))
    assert before_smoke==hashes(task.SMOKE)
    task.local.write_json(target/'artifact_hashes.json',hashes(target))
    formal=repo/'results'/task.OUTPUT.name/'startup_closed_01';assert not formal.exists()
    formal.mkdir(parents=True)
    files=['contract.json','startup_closed.json']
    dirs=['executed_sources','frozen','zero_parity','nonzero_smoke']
    before={}
    for name in files:
        path=task.OUTPUT/name;before[name]=task.digest(path);shutil.copy2(path,formal/name)
    for name in dirs:
        before[name]=hashes(task.OUTPUT/name);shutil.copytree(task.OUTPUT/name,formal/name)
    for name in files:assert before[name]==task.digest(task.OUTPUT/name)
    for name in dirs:assert before[name]==hashes(task.OUTPUT/name)
    logs=formal/'regression_logs';logs.mkdir()
    for stage in ('smoke','formal'):
        source=task.ROOT/f'tmp/getup_intervention_dynamics_r173_regression_{stage}_20261009.log'
        assert 'PASS' in source.read_text();shutil.copy2(source,logs/source.name)
    task.local.write_json(formal/'snapshot.json',dict(terminal_result_saved=False,training_programs_saved=0,
        formal_startup_full_attempts=6,all_arrays_hashes_peaks_match_independent_smoke=True,new_controller=False,
        full_task_completed=False,hardware_readiness=False,qualification_run=False,open_training_episodes_not_copied=True))
    task.local.write_json(formal/'artifact_hashes.json',hashes(formal))
    for folder in (target,formal):subprocess.run(['git','add','-f',str(folder.relative_to(repo))],cwd=repo,check=True)
    for name in ('train_getup_intervention_dynamics_r173.py','test_getup_intervention_dynamics_r173.py',
                 'launch_getup_intervention_dynamics_r173.py',Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(task.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R173 signed intervention regressions, independent complete smoke and closed startup; formal data and model still separate'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),parent=args.base,
        bundle=str(bundle),bytes=bundle.stat().st_size,sha256=task.digest(bundle),terminal_result_saved=False,training_programs_saved=0)),flush=True)

if __name__=='__main__':main()
