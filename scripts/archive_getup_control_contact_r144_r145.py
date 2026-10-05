import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.audit_getup_control_contact_r144 import ROOT,OUTPUT as R144
from diagnostics.audit_getup_planned_contact_r145 import OUTPUT as R145
from diagnostics.getup_independent_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    targets=[]
    for source in (R144,R145,ROOT/'outputs/getup_control_contact_r144_smoke_20261006',ROOT/'outputs/getup_planned_contact_r145_smoke_20261006'):
        report=json.loads((source/'results.json').read_text())
        assert len(report['rows'])==(3 if report['smoke'] else 16) and report['regression_checks_passed']
        if source==R145:assert report['invalid_with_sampled_warning']==5 and report['valid_with_sampled_warning']==11
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        shutil.copytree(source,target)
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=True,
            read_only_diagnostic_not_training=True,not_new_strategy=True,full_task_completed=False),indent=2))
        (target/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
        targets.append(target)
    scripts=['audit_getup_control_contact_r144.py','audit_getup_planned_contact_r145.py',Path(__file__).name]
    for name in scripts:
        assert not (repo/'scripts'/name).exists();shutil.copy2(Path(__file__).with_name(name),repo/'scripts'/name)
    doc='GETUP_CONTROL_CONTACT_R144_R145_20261006.md';assert not (repo/doc).exists()
    shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+s for s in scripts]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(p.relative_to(repo)) for p in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve causal control contact and planned target diagnostics R144 R145'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_control_contact_r144_r145_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
