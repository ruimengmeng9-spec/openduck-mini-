"""Append R162 once-only frozen full contrast and R163 read-only terminal."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import probe_getup_fixed_bilateral_r162 as run
from diagnostics import audit_getup_fixed_bilateral_r163 as audit
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    repo=run.run.ROOT/'github/openduck-mini-';doc='GETUP_FIXED_BILATERAL_TERMINAL_R162_R163_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip() and not (repo/doc).exists()
    result=json.loads((run.OUTPUT/'results.json').read_text());ar=json.loads((audit.OUTPUT/'results.json').read_text())
    assert not result['smoke'] and not result['original_development_gate'] and not result['independent_qualification_run']
    assert result['candidate']['successes']==8 and result['baseline']['successes']==18 and result['candidate']['physical_failures']==2
    assert ar['read_only'] and len(ar['rows'])==25 and ar['rescued']==[769002] and len(ar['regressed'])==11
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==56
    smoke=run.run.ROOT/'outputs/getup_fixed_bilateral_r162_smoke_20261006'
    assert json.loads((smoke/'results.json').read_text())['smoke'] and len(list(smoke.rglob('trajectory.npz')))==6
    bundle=run.run.ROOT/'tmp/getup_fixed_bilateral_terminal_r162_r163_20261006.bundle';assert not bundle.exists();saved=[]
    write=run.run.prior.old.local.write_json
    for source,readonly in [(run.OUTPUT,False),(smoke,False),(audit.OUTPUT,True)]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        if not readonly:shutil.copy2(run.run.ROOT/'tmp/getup_fixed_bilateral_r162_regression_20261006.log',target/'regression.log')
        assert before==hashes(source),'Terminal source changed during archive'
        trajectories=len(list(source.rglob('trajectory.npz')));saved.append(dict(source=str(source),trajectories=trajectories))
        write(target/'snapshot.json',dict(terminal_result_saved=True,source=str(source),read_only=readonly,saved_trajectories=trajectories,
            full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
        write(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['probe_getup_fixed_bilateral_r162.py','launch_getup_fixed_bilateral_r162.py','audit_getup_fixed_bilateral_r163.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R162 frozen bilateral full-path failures and R163 exact paired causal signals'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+a.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
