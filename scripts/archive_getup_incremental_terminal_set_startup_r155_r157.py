"""Append immutable R155/R156 terminal and R157 closed learner/smoke."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_incremental_residual_r155 as r155
from diagnostics import audit_getup_incremental_terminal_r156 as r156
from diagnostics import train_getup_set_decision_r157 as r157
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=r155.ROOT/'github/openduck-mini-';doc='GETUP_SET_DECISION_R155_R157_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    terminal=json.loads((r155.OUTPUT/'results.json').read_text());audit=json.loads((r156.OUTPUT/'results.json').read_text())
    smoke=json.loads((r157.SMOKE/'results.json').read_text())
    assert not terminal['smoke'] and len(json.loads((r155.OUTPUT/'training_closed.json').read_text())['history'])==6
    assert len(audit['full_pair_exact'])==25 and audit['distinct_programs']==61 and audit['read_only']
    assert smoke['smoke'] and not smoke['independent_qualification_run']
    assert len(list(r155.OUTPUT.rglob('trajectory.npz')))==1581
    for relative in ['model.npz',*[f'checkpoints/update_{n:05d}/learner.npz' for n in [250,500,750,1000]]]:
        with np.load(r157.SMOKE/'training'/relative,allow_pickle=False) as a,np.load(r157.OUTPUT/'training'/relative,allow_pickle=False) as b:
            assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    bundle=r155.ROOT/'tmp/getup_incremental_terminal_set_startup_r155_r157_20261006.bundle';assert not bundle.exists()
    saved=[]
    for source,readonly in [(r155.OUTPUT,False),(r156.OUTPUT,True),(r157.SMOKE,False)]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        assert before==hashes(source),'Closed source changed during archive'
        count=len(list(source.rglob('trajectory.npz')))
        r157.old.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,source=str(source),saved_trajectories=count,
            read_only=readonly,full_task_completed=False,hardware_readiness=False,independent_qualification_run=False,short_training_not_30s_acceptance=True))
        r157.old.local.write_json(target/'artifact_hashes.json',hashes(target));saved.append(dict(source=str(source),trajectories=count))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/r157.OUTPUT.name/'training_closed_01000_startup';assert not target.exists()
    target.mkdir(parents=True);immutable=[r157.OUTPUT/'training',r157.OUTPUT/'executed_sources',r157.OUTPUT/'contract.json']
    before={str(s):hashes(s) if s.is_dir() else digest(s) for s in immutable}
    for source in immutable:
        if source.is_dir():shutil.copytree(source,target/source.name)
        else:shutil.copy2(source,target/source.name)
    testlog=r157.ROOT/'tmp/getup_set_decision_r157_regression_20261006.log';shutil.copy2(testlog,target/'regression.log')
    assert before=={str(s):hashes(s) if s.is_dir() else digest(s) for s in immutable}
    r157.old.local.write_json(target/'snapshot.json',dict(terminal_result_saved=False,training_closed_update=1000,
        independent_smoke_saved=True,learner_and_model_bitwise_equal_to_smoke=True,formal_full_validation_pending=True,
        saved_trajectories=0,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
    r157.old.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_incremental_terminal_r156.py','train_getup_set_decision_r157.py','test_getup_set_decision_r157.py','launch_getup_set_decision_r157.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(r155.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R155 complete terminal, R156 direct-residual audit and R157 set-decision closed learner startup'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
