"""Append immutable complete diagonal-feedback terminal evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import train_getup_jointwise_dynamic_r151 as r151
from diagnostics import audit_getup_jointwise_terminal_r152 as r152
from diagnostics import probe_getup_fixed_dynamic_r153 as r153
from diagnostics.getup_independent_native import digest

def hashes(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=r151.ROOT/'github/openduck-mini-';doc='GETUP_JOINTWISE_TERMINAL_R151_R153_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    a=json.loads((r151.OUTPUT/'results.json').read_text());b=json.loads((r153.OUTPUT/'results.json').read_text())
    audit=json.loads((r152.OUTPUT/'results.json').read_text())
    assert len(json.loads((r151.OUTPUT/'training_closed.json').read_text())['history'])==6
    assert not a['smoke'] and not b['smoke'] and audit['read_only']
    assert not a['independent_qualification_run'] and not b['independent_qualification_run']
    assert len(audit['full_pair_exact'])==25 and audit['distinct_programs']==61
    bundle=r151.ROOT/'tmp/getup_jointwise_terminal_r151_r153_20261006.bundle';assert not bundle.exists()
    saved=[]
    for source,readonly in [(r151.OUTPUT,False),(r152.OUTPUT,True),(r153.OUTPUT,False),
                            (r151.ROOT/'outputs/getup_fixed_dynamic_r153_smoke_20261006',False)]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        assert before==hashes(source),'Closed source changed during archive'
        count=len(list(source.rglob('trajectory.npz')))
        r151.prior.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,source=str(source),saved_trajectories=count,
            read_only=readonly,full_task_completed=False,hardware_readiness=False,independent_qualification_run=False,
            short_training_not_30s_acceptance=True))
        r151.prior.local.write_json(target/'artifact_hashes.json',hashes(target));saved.append(dict(source=str(source),trajectories=count))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_jointwise_terminal_r152.py','probe_getup_fixed_dynamic_r153.py','launch_getup_fixed_dynamic_r153.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(r151.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R151 six-generation terminal, R152 audit and R153 fixed diagonal full-path probe'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
