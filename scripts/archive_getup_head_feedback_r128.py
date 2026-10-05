import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.probe_getup_head_feedback_r128 import ROOT,OUTPUT
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    targets=[]
    for source in (ROOT/'outputs/getup_head_feedback_r128_smoke_20261005',OUTPUT):
        assert source.exists()
        if 'smoke' in source.name:assert (source/'results.json').exists()
        target=repo/'results'/source.name/args.snapshot;assert not target.exists();target.mkdir(parents=True);targets.append(target)
        shutil.copytree(source/'executed_sources',target/'executed_sources')
        for name in ('contract.json','progress.json','results.json'):
            if (source/name).exists():shutil.copy2(source/name,target/name)
        for marker in source.rglob('result.json'):
            shutil.copytree(marker.parent,target/marker.parent.relative_to(source))
        if source.with_suffix('.log').exists():shutil.copy2(source.with_suffix('.log'),target/source.with_suffix('.log').name)
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),
            closed_trials=len(list(target.rglob('result.json'))),not_training=True,no_candidate_promotion=True,
            hardware_readiness=False,full_task_completed=False),indent=2))
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=['probe_getup_head_feedback_r128.py','test_getup_head_feedback_r128.py','launch_getup_head_feedback_r128.py',Path(__file__).name]
    for name in names:
        dst=repo/'scripts'/name
        if dst.exists():assert digest(dst)==digest(Path(__file__).with_name(name))
        else:shutil.copy2(Path(__file__).with_name(name),dst)
    cwd=ROOT/'projects/Open_Duck_Playground'
    tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_head_feedback_r128','-v'],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert tests.returncode==0
    (targets[0]/'archive_regression_tests.log').write_text(tests.stdout)
    for target in targets:
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    doc='GETUP_HEAD_FEEDBACK_R128_20261005.md';dest=repo/doc
    if dest.exists():assert digest(dest)==digest(Path(__file__).with_name(doc))
    else:shutil.copy2(Path(__file__).with_name(doc),dest)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve bounded causal head feedback R128 '+args.snapshot],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_head_feedback_r128_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
