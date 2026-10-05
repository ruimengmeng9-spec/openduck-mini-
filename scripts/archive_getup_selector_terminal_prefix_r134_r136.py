import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.probe_getup_common_prefix_r136 import ROOT,OUTPUT
from diagnostics.train_getup_success_selector_r134 import OUTPUT as SELECTOR
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    repo=ROOT/'github/openduck-mini-';assert args.snapshot.replace('_','').isalnum()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert (SELECTOR/'results.json').exists()
    targets=[]
    for original in (SELECTOR,ROOT/'outputs/getup_success_choice_audit_r135_20261006',ROOT/'outputs/getup_common_prefix_r136_smoke_20261006',OUTPUT):
        assert original.exists();target=repo/'results'/original.name/args.snapshot;assert not target.exists();target.mkdir(parents=True);targets.append(target)
        for name in ('executed_sources','training'):
            if (original/name).exists():shutil.copytree(original/name,target/name)
        for name in ('contract.json','results.json','frozen_programs.npz'):
            if (original/name).exists():shutil.copy2(original/name,target/name)
        for marker in original.rglob('result.json'):shutil.copytree(marker.parent,target/marker.parent.relative_to(original))
        for group in original.rglob('results.json'):
            if group.parent!=original:shutil.copy2(group,target/group.relative_to(original))
        if original.with_suffix('.log').exists():shutil.copy2(original.with_suffix('.log'),target/original.with_suffix('.log').name)
        for src in original.glob('*.py'):shutil.copy2(src,target/src.name)
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),closed_trials=len(list(target.rglob('result.json'))),
            hardware_readiness=False,full_task_completed=False),indent=2))
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    names=['audit_getup_success_selector_r135.py','probe_getup_common_prefix_r136.py','test_getup_common_prefix_r136.py','launch_getup_common_prefix_r136.py',Path(__file__).name]
    for name in names:
        target=repo/'scripts'/name
        if target.exists():assert digest(target)==digest(Path(__file__).with_name(name))
        else:shutil.copy2(Path(__file__).with_name(name),target)
    doc='GETUP_SELECTOR_TERMINAL_PREFIX_R134_R136_20261006.md';target=repo/doc
    if target.exists():assert digest(target)==digest(Path(__file__).with_name(doc))
    else:shutil.copy2(Path(__file__).with_name(doc),target)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve selector terminal and common-prefix '+args.snapshot],cwd=repo,check=True)
    bundle=ROOT/'tmp'/('getup_selector_terminal_prefix_r134_r136_'+args.snapshot+'_20261006.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
