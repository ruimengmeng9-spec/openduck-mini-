"""Append complete R130, R131/R132 diagnostics and immutable evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.train_getup_local_hip_r130 import ROOT,OUTPUT
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args();repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    result=json.loads((OUTPUT/'results.json').read_text());assert result['gains']==[0.]*6 and not result['original_development_gate']
    targets=[]
    for source in (OUTPUT,ROOT/'outputs/getup_local_hip_audit_r131_20261006',ROOT/'outputs/getup_velocity_observability_r132_20261006'):
        assert (source/'results.json').exists()
        dest=repo/'results'/source.name/'terminal_snapshot';assert not dest.exists();shutil.copytree(source,dest);targets.append(dest)
        if source.with_suffix('.log').exists():shutil.copy2(source.with_suffix('.log'),dest/source.with_suffix('.log').name)
        (dest/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=True,full_task_completed=False,hardware_readiness=False),indent=2))
        (dest/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(dest)):digest(f) for f in dest.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    names=['audit_getup_local_hip_r131.py','probe_getup_velocity_observability_r132.py',Path(__file__).name]
    for name in names:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
    doc='GETUP_LOCAL_HIP_TERMINAL_R130_R133_20261006.md';assert not (repo/doc).exists();shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve complete R130 and causal diagnostics R131 R132'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_local_terminal_r130_r132_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
