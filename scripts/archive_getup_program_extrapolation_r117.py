"""Append immutable completed diagnostic trials and source failure records."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest
from diagnostics.probe_getup_program_extrapolation_r117 import ROOT,OUTPUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--snapshot',required=True);args=parser.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    target=repo/'results'/OUTPUT.name/args.snapshot;assert not target.exists();target.mkdir(parents=True)
    shutil.copytree(OUTPUT/'executed_sources',target/'executed_sources')
    for name in ('contract.json','parameter_audit.json','parity.json','progress.json','results.json'):
        if (OUTPUT/name).exists():shutil.copy2(OUTPUT/name,target/name)
    counts={}
    for group in ('original','no_nodes','no_feedback'):
        source=OUTPUT/group
        if not source.exists():continue
        count=0
        for case in sorted(source.glob('case_*')):
            if (case/'result.json').exists():shutil.copytree(case,target/group/case.name);count+=1
        counts[group]=count
        if (source/'results.json').exists():shutil.copy2(source/'results.json',target/group/'results.json')
    shutil.copy2(OUTPUT.with_suffix('.log'),target/'parent_log_snapshot.log')
    cwd=ROOT/'projects/Open_Duck_Playground'
    tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_program_extrapolation_r117','-v'],
        cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'snapshot_regression_tests.log').write_text(tests.stdout);assert tests.returncode==0
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),completed_trial_counts=counts,
        diagnostic_development_only=True,full_task_completed=False,hardware_readiness=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=('probe_getup_program_extrapolation_r117.py','test_getup_program_extrapolation_r117.py','launch_getup_program_extrapolation_r117.py',Path(__file__).name)
    for name in names:
        src=Path(__file__).with_name(name);dst=repo/'scripts'/name
        if dst.exists():assert digest(dst)==digest(src)
        else:shutil.copy2(src,dst)
    doc='GETUP_EXPANDED_PROGRAM_TERMINAL_R116_R117_20261005.md';src=Path(__file__).with_name(doc)
    if (repo/doc).exists():assert digest(repo/doc)==digest(src)
    else:shutil.copy2(src,repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R116 unseen failure and program component diagnosis R117 '+args.snapshot],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_program_extrapolation_r117_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        terminal_result_saved=(target/'results.json').exists(),completed_trial_counts=counts)),flush=True)


if __name__=='__main__':main()
