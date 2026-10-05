import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.train_getup_success_selector_r134 import ROOT,OUTPUT,source
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum();repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    targets=[]
    for original in (source.OUTPUT,ROOT/'outputs/getup_complementary_feedback_r133_smoke_20261006',ROOT/'outputs/getup_success_selector_r134_smoke_20261006',OUTPUT):
        assert original.exists();target=repo/'results'/original.name/args.snapshot;assert not target.exists();target.mkdir(parents=True);targets.append(target)
        shutil.copytree(original/'executed_sources',target/'executed_sources')
        if (original/'training').exists():
            assert (original/'training/training.json').exists();shutil.copytree(original/'training',target/'training')
        for name in ('contract.json','frozen_programs.npz','offline_success_sets.npz','manifest.json','results.json'):
            if (original/name).exists():shutil.copy2(original/name,target/name)
        # Results are atomically closed after the arrays are fully written.
        for marker in original.rglob('result.json'):
            shutil.copytree(marker.parent,target/marker.parent.relative_to(original))
        for group in original.rglob('results.json'):
            if group.parent!=original:shutil.copy2(group,target/group.relative_to(original))
        if original.with_suffix('.log').exists():shutil.copy2(original.with_suffix('.log'),target/original.with_suffix('.log').name)
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),closed_trials=len(list(target.rglob('result.json'))),
            no_independent_qualification=True,hardware_readiness=False,full_task_completed=False),indent=2))
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    names=['probe_getup_complementary_feedback_r133.py','test_getup_complementary_feedback_r133.py','launch_getup_complementary_feedback_r133.py',
        'train_getup_success_selector_r134.py','test_getup_success_selector_r134.py','launch_getup_success_selector_r134.py',Path(__file__).name]
    for name in names:
        target=repo/'scripts'/name
        if target.exists():assert digest(target)==digest(Path(__file__).with_name(name))
        else:shutil.copy2(Path(__file__).with_name(name),target)
    doc='GETUP_MULTI_SUCCESS_SELECTOR_R133_R134_20261006.md';target=repo/doc
    if target.exists():assert digest(target)==digest(Path(__file__).with_name(doc))
    else:shutil.copy2(Path(__file__).with_name(doc),target)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve complete complementary programs and success-set selector '+args.snapshot],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_success_selector_r133_r134_'+args.snapshot+'_20261006.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
