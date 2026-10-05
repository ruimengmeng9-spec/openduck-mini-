"""Append complete immutable R147, R148 and R149 terminal evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import train_getup_dynamic_gain_r147 as r147
from diagnostics import audit_getup_dynamic_terminal_r148 as r148
from diagnostics import probe_getup_fixed_dynamic_r149 as r149
from diagnostics.getup_independent_native import digest

def hashes(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=r147.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    doc='GETUP_DYNAMIC_TERMINAL_R147_R149_20261006.md'
    assert not (repo/doc).exists()
    result147=json.loads((r147.OUTPUT/'results.json').read_text())
    result149=json.loads((r149.OUTPUT/'results.json').read_text())
    audit=json.loads((r148.OUTPUT/'results.json').read_text())
    assert len(json.loads((r147.OUTPUT/'training_closed.json').read_text())['history'])==8
    assert not result149['smoke'] and not result147['smoke'] and audit['read_only']
    for result in [result147,result149]:assert not result['independent_qualification_run']
    copies=[(r147.OUTPUT,False),(r148.OUTPUT,True),(r149.OUTPUT,False),
        (r147.ROOT/'outputs/getup_fixed_dynamic_r149_smoke_20261006',False)]
    saved=[]
    for source,readonly in copies:
        before=hashes(source)
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        assert before==hashes(source),'Immutable terminal source changed during copy'
        trajectories=len(list(source.rglob('trajectory.npz')))
        r147.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,source=str(source),
            saved_trajectories=trajectories,read_only=readonly,full_task_completed=False,hardware_readiness=False,
            independent_qualification_run=False,short_training_not_30s_acceptance=True))
        r147.local.write_json(target/'artifact_hashes.json',hashes(target))
        saved.append(dict(source=str(source),trajectories=trajectories))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    names=['audit_getup_dynamic_terminal_r148.py','probe_getup_fixed_dynamic_r149.py',
        'launch_getup_fixed_dynamic_r149.py',Path(__file__).name]
    for name in names:
        dest=repo/'scripts'/name;assert not dest.exists()
        shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(r147.ROOT/'tmp'/doc,repo/doc)
    subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R147 full terminal, R148 causal audit and R149 fixed probe failures'],cwd=repo,check=True)
    bundle=r147.ROOT/'tmp/getup_dynamic_terminal_r147_r149_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        bundle=str(bundle),bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
