"""Append only closed read-only R167, with immutable unique snapshot/bundle."""
import argparse
from pathlib import Path
import shutil
import subprocess
from diagnostics import audit_getup_expert_directions_r167 as run

def hashes(root):return {str(p.relative_to(root)):run.audit.digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=run.run.ROOT/'github/openduck-mini-';doc='GETUP_EXPERT_DIRECTIONS_R167_20261008.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    bundle=run.run.ROOT/'tmp/getup_expert_directions_r167_20261008.bundle';assert not bundle.exists()
    assert not (repo/doc).exists()
    for source,smoke,count in [(run.SMOKE,True,3),(run.OUTPUT,False,25)]:
        result=run.audit.read(source/'results.json');assert result['read_only'] and result['smoke']==smoke and result['cases']==count
        assert result['regression_assertions']==6 and result['source_hashes_unchanged'] and not result['new_controller_trained']
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target);shutil.copy2(source.with_suffix('.log'),target/'process.log')
        assert before==hashes(source)
        run.audit.write(target/'snapshot.json',dict(terminal_result_saved=True,read_only=True,source=str(source),cases=count,algebra_comparisons=count*4,
            new_dynamic_trajectories=0,new_controller_trained=False,full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
        run.audit.write(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in [Path(run.__file__).name,Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R167 same-state fixed-expert direction audit and unchanged original outcomes'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(__import__('json').dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bytes=bundle.stat().st_size,sha256=run.audit.digest(bundle))),flush=True)

if __name__=='__main__':main()
