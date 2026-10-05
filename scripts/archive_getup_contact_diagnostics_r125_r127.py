"""Append immutable collision diagnostics, including original failed replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.audit_getup_self_geometry_r125 import ROOT
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--document',required=True);args=p.parse_args()
    repo=ROOT/'github/openduck-mini-';cwd=ROOT/'projects/Open_Duck_Playground'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    groups=['getup_self_geometry_r125_20261005','getup_substep_contacts_r126_smoke_20261005',
        'getup_substep_contacts_r126_left_20261005','getup_substep_contacts_r126b_smoke_20261005',
        'getup_substep_contacts_r126b_left_20261005','getup_contact_prestate_r127_20261005']
    targets=[]
    for name in groups:
        source=ROOT/'outputs'/name;assert source.exists()
        failed=name=='getup_substep_contacts_r126_left_20261005'
        if not failed:assert (source/'results.json').exists()
        else:assert not (source/'results.json').exists()
        target=repo/'results'/name/('failed_interface_snapshot' if failed else 'terminal_snapshot')
        assert not target.exists();shutil.copytree(source,target);targets.append(target)
        if source.with_suffix('.log').exists():shutil.copy2(source.with_suffix('.log'),target/source.with_suffix('.log').name)
        if failed:
            (target/'failure_record.json').write_text(json.dumps(dict(complete=False,
                reason='Teacher 773007 normalized residual mismatch at control 529; physical state and applied targets were equal. R126b restores historical teacher formula without weakening parity.',
                failed_run_preserved=True,no_success_claim=True),indent=2))
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=not failed,
            completed_replay_markers=len(list(source.rglob('result.json'))),no_new_training=True,
            independent_qualification_run=False,hardware_readiness=False,full_task_completed=False),indent=2))
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=['audit_getup_self_geometry_r125.py','test_getup_self_geometry_r125.py','probe_getup_substep_contacts_r126.py',
        'launch_getup_substep_contacts_r126.py','probe_getup_substep_contacts_r126b.py','launch_getup_substep_contacts_r126b.py',
        'audit_getup_contact_prestate_r127.py',Path(__file__).name]
    for name in names:
        src=Path(__file__).with_name(name);dst=repo/'scripts'/name;assert not dst.exists();shutil.copy2(src,dst)
    document=repo/args.document;assert document.name==args.document and not document.exists()
    shutil.copy2(Path(__file__).with_name(args.document),document)
    tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_self_geometry_r125','-v'],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert tests.returncode==0;(targets[0]/'archive_regression_tests.log').write_text(tests.stdout)
    # Hash the final snapshot including regression output; exclude hash manifest itself.
    for target in targets:
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    subprocess.run(['git','add',args.document,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve self collision forward and substep diagnostics R125-R127 including failed interface replay'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_contact_diagnostics_r125_r127_terminal_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
