"""Append closed R160 full terminal and R161 read-only evidence, no overwrites."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import train_getup_bilateral_modes_r160 as run
from diagnostics import audit_getup_bilateral_terminal_r161 as audit
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_BILATERAL_TERMINAL_R160_R161_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip() and not (repo/doc).exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text());ar=json.loads((audit.OUTPUT/'results.json').read_text())
    assert len(json.loads((run.OUTPUT/'training_closed.json').read_text())['history'])==6
    assert terminal['candidate']['successes']==18 and terminal['baseline']['successes']==18
    assert not terminal['original_development_gate'] and not terminal['independent_qualification_run']
    assert ar['read_only'] and len(ar['full_pair_exact'])==25 and ar['distinct_programs']==61
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==1581
    bundle=run.ROOT/'tmp/getup_bilateral_terminal_r160_r161_20261006.bundle';assert not bundle.exists();saved=[]
    write=run.prior.old.local.write_json
    for source,readonly in [(run.OUTPUT,False),(audit.OUTPUT,True)]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        if source==run.OUTPUT:shutil.copy2(run.ROOT/'tmp/getup_bilateral_modes_r160_regression_20261006.log',target/'regression.log')
        assert before==hashes(source),'Terminal source changed during archive'
        trajectories=len(list(source.rglob('trajectory.npz')))
        write(target/'snapshot.json',dict(terminal_result_saved=True,source=str(source),read_only=readonly,saved_trajectories=trajectories,
            full_task_completed=False,hardware_readiness=False,independent_qualification_run=False,short_training_not_30s_acceptance=True))
        write(target/'artifact_hashes.json',hashes(target));saved.append(dict(source=str(source),trajectories=trajectories))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_bilateral_terminal_r161.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R160 six-generation full terminal and R161 exact bilateral failure audit'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+a.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
