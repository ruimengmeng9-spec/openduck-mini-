"""Append safe completed artifacts, never overwrite an earlier R115 snapshot."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_expanded_teachers_r115 import ROOT,OUTPUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    target=repo/'results'/OUTPUT.name/args.snapshot;assert not target.exists();target.mkdir(parents=True)
    shutil.copytree(OUTPUT/'executed_sources',target/'executed_sources')
    for name in ('contract.json','frozen_profiles.npz','parity.json','coverage_before_search.json','results.json'):
        if (OUTPUT/name).exists():shutil.copy2(OUTPUT/name,target/name)
    terminal=(target/'results.json').exists()
    # Result marker is written after all trajectory and parameter files. Finished
    # trial directories are immutable, so live snapshots copy only those.
    counts={}
    for group in ('parity_zero','parity_predicted','expanded_r102','teachers','full_checks'):
        original=OUTPUT/group
        if not original.exists():continue
        markers=list(original.rglob('result.json')) if group not in ('parity_zero','parity_predicted') else [original/'result.json']
        count=0
        for marker in markers:
            if marker.exists():
                dest=target/group/marker.parent.relative_to(original)
                shutil.copytree(marker.parent,dest);count+=1
        counts[group]=count
        if (original/'results.json').exists():shutil.copy2(original/'results.json',target/group/'results.json')
    if terminal:
        for name in ('history.json','progress.json'):
            if (OUTPUT/name).exists():shutil.copy2(OUTPUT/name,target/name)
        if (OUTPUT/'checkpoints').exists():shutil.copytree(OUTPUT/'checkpoints',target/'checkpoints')
    shutil.copy2(OUTPUT.with_suffix('.log'),target/'parent_log_snapshot.log')
    cwd=ROOT/'projects/Open_Duck_Playground'
    tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_expanded_teachers_r115','-v'],
        cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'snapshot_regression_tests.log').write_text(tests.stdout);assert tests.returncode==0
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=terminal,completed_trial_counts=counts,
        teacher_data_only=True,unified_policy_success=False,full_task_completed=False,hardware_readiness=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=('search_getup_expanded_teachers_r115.py','test_getup_expanded_teachers_r115.py','launch_getup_expanded_teachers_r115.py',Path(__file__).name)
    for name in names:
        dst=repo/'scripts'/name;src=Path(__file__).with_name(name)
        if dst.exists():assert digest(dst)==digest(src)
        else:shutil.copy2(src,dst)
    doc='GETUP_EXPANDED_TEACHERS_R115_20261005.md';src=Path(__file__).with_name(doc)
    if (repo/doc).exists():assert digest(repo/doc)==digest(src)
    else:shutil.copy2(src,repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve expanded complete-fall offline teachers R115 '+args.snapshot],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_expanded_teachers_r115_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        terminal_result_saved=terminal,completed_trial_counts=counts)),flush=True)


if __name__=='__main__':main()
