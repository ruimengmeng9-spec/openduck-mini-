"""Append R118 completed evidence only; never overwrite earlier snapshots."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest
from diagnostics.probe_getup_teacher_neighborhood_r118 import ROOT,OUTPUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    target=repo/'results'/OUTPUT.name/args.snapshot;assert not target.exists();target.mkdir(parents=True)
    shutil.copytree(OUTPUT/'executed_sources',target/'executed_sources')
    for name in ('contract.json','pairs.json','own_replay.json','results.json'):
        if (OUTPUT/name).exists():shutil.copy2(OUTPUT/name,target/name)
    counts={}
    for group in ('own_replay','neighbors'):
        source=OUTPUT/group
        if not source.exists():continue
        count=0
        for marker in sorted(source.rglob('result.json')):
            shutil.copytree(marker.parent,target/group/marker.parent.relative_to(source));count+=1
        counts[group]=count
    shutil.copy2(OUTPUT.with_suffix('.log'),target/'parent_log_snapshot.log')
    cwd=ROOT/'projects/Open_Duck_Playground'
    tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_teacher_neighborhood_r118','-v'],
        cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'snapshot_regression_tests.log').write_text(tests.stdout);assert tests.returncode==0
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),completed_trial_counts=counts,
        diagnostic_development_only=True,deployed_nearest_neighbor_policy=False,full_task_completed=False,hardware_readiness=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=('probe_getup_teacher_neighborhood_r118.py','test_getup_teacher_neighborhood_r118.py','launch_getup_teacher_neighborhood_r118.py',Path(__file__).name)
    for name in names:
        src=Path(__file__).with_name(name);dst=repo/'scripts'/name
        if dst.exists():assert digest(dst)==digest(src)
        else:shutil.copy2(src,dst)
    doc='GETUP_TEACHER_NEIGHBORHOOD_R118_20261005.md';src=Path(__file__).with_name(doc)
    if (repo/doc).exists():assert digest(repo/doc)==digest(src)
    else:shutil.copy2(src,repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve local teacher program robustness audit R118 '+args.snapshot],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_teacher_neighborhood_r118_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        terminal_result_saved=(target/'results.json').exists(),completed_trial_counts=counts)),flush=True)


if __name__=='__main__':main()
