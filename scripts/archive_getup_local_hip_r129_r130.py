"""Append immutable R129 terminal and only closed R130 evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.train_getup_local_hip_r130 import ROOT,OUTPUT
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    targets=[]
    audit=ROOT/'outputs/getup_head_coupling_r129_20261005';assert (audit/'results.json').exists()
    target=repo/'results'/audit.name/args.snapshot;assert not target.exists()
    shutil.copytree(audit,target);targets.append(target)
    for source in (ROOT/'outputs/getup_local_hip_r130_smoke_20261005',OUTPUT):
        assert source.exists()
        target=repo/'results'/source.name/args.snapshot;assert not target.exists();target.mkdir(parents=True);targets.append(target)
        shutil.copytree(source/'executed_sources',target/'executed_sources')
        shutil.copytree(source/'frozen',target/'frozen')
        for name in ('contract.json','results.json','training_closed.json'):
            if (source/name).exists():shutil.copy2(source/name,target/name)
        # A group result is written only after every trial in that group closes.
        for group in source.rglob('results.json'):
            if group.parent==source:continue
            shutil.copytree(group.parent,target/group.parent.relative_to(source))
        # Checkpoints close with a full history file; copy only those generations.
        for history in (source/'checkpoints').glob('generation_*/history.json') if (source/'checkpoints').exists() else ():
            shutil.copytree(history.parent,target/history.parent.relative_to(source))
        generations=[int(f.parent.name.split('_')[-1]) for f in (target/'checkpoints').glob('generation_*/history.json')] if (target/'checkpoints').exists() else []
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),
            closed_generation=max(generations,default=0),closed_trials=len(list(target.rglob('result.json'))),
            snapshot_only_not_live_run_status=True,hardware_readiness=False,full_task_completed=False),indent=2))
        if source.with_suffix('.log').exists():shutil.copy2(source.with_suffix('.log'),target/source.with_suffix('.log').name)
    names=['audit_getup_head_coupling_r129.py','train_getup_local_hip_r130.py','test_getup_local_hip_r130.py','launch_getup_local_hip_r130.py',Path(__file__).name]
    for name in names:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
    tests=subprocess.run([str(ROOT/'projects/Open_Duck_Playground/.venv/bin/python'),'-m','unittest','diagnostics.test_getup_local_hip_r130','-v'],cwd=ROOT/'projects/Open_Duck_Playground',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert tests.returncode==0
    (targets[1]/'archive_regression_tests.log').write_text(tests.stdout)
    for target in targets:
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    doc='GETUP_LOCAL_HIP_FEEDBACK_R129_R130_20261005.md';assert not (repo/doc).exists();shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+name for name in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R129 coupling audit and R130 closed simulation evidence'],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_local_hip_r129_r130_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
