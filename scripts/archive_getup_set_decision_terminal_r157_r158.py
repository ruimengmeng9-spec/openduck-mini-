"""Append unique complete R157/R158 terminal, never replace startup evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import train_getup_set_decision_r157 as run
from diagnostics import audit_getup_set_decision_terminal_r158 as audit
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_SET_DECISION_TERMINAL_R157_R158_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    result=json.loads((run.OUTPUT/'results.json').read_text());analysis=json.loads((audit.OUTPUT/'results.json').read_text())
    assert not result['smoke'] and not result['original_development_gate'] and not result['independent_qualification_run']
    assert len(analysis['rows'])==25 and analysis['read_only'] and analysis['rescued']==[773007] and not analysis['regressed']
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==50
    bundle=run.ROOT/'tmp/getup_set_decision_terminal_r157_r158_20261006.bundle';assert not bundle.exists()
    for source,readonly in [(run.OUTPUT,False),(audit.OUTPUT,True)]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        assert before==hashes(source),'Closed source changed during archive'
        run.old.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,source=str(source),read_only=readonly,
            saved_trajectories=len(list(source.rglob('trajectory.npz'))),full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
        run.old.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_set_decision_terminal_r158.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R157 full-path 18/24 terminal and R158 exact paired decision audit'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size)),flush=True)

if __name__=='__main__':main()
