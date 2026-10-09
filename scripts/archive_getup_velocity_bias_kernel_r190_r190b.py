"""Append the actual failed test and corrected offline terminal; no replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from diagnostics import probe_getup_centroidal_momentum_r188b as old


def inventory(root):
    return {str(p.relative_to(root)):old.prior.digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--base',required=True); args=parser.parse_args()
    root=old.ROOT; repo=root/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    processes=subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in c and any(s in c for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_velocity_bias_kernel_')) for c in processes)
    assert not any(('git push' in c) or ('/tmp/publish_getup_' in c and '--push' in c) for c in processes)
    assert shutil.disk_usage(root).free>10.25*1024**3
    for suffix,tests,passed,name in (('r190',19,False,'failure_snapshot_01'),('r190b',20,True,'terminal_snapshot')):
        source=root/'outputs'/f'getup_velocity_bias_kernel_{suffix}_20261010'
        row=json.loads((source/'results.json').read_text())
        assert row['terminal_result_saved'] and row['tests_passed']==passed and row['regression_tests']==tests
        assert row['regression_exit_code']==(0 if passed else 1) and row['new_dynamic_attempts']==0 and not row['new_controller_run']
        assert row['source_input_unchanged'] and row['compiled_model_unchanged']
        assert not row['full_task_completed'] and not row['hardware_readiness'] and not row['qualification_run']
        assert row['input_hashes_before']==row['input_hashes_after']
        for path,sha in row['input_hashes_before'].items(): assert old.prior.digest(Path(path))==sha
        contract=json.loads((source/'contract.json').read_text())
        manifests=((contract['source_manifest'],'executed_sources'),(json.loads((source/'test_source_manifest.json').read_text()),'test_executed_sources'))
        for manifest,folder in manifests:
            for path,info in manifest.items():
                assert old.prior.digest(Path(path))==info['sha256']==old.prior.digest(source/folder/info['copy'])
        if passed:
            assert row['first_failed_evidence_unchanged']
            for path,sha in row['first_failed_evidence_hashes_before'].items(): assert old.prior.digest(Path(path))==sha
        before=inventory(source); destination=repo/'results'/source.name/name
        assert not destination.exists(); shutil.copytree(source,destination)
        assert before==inventory(source)==inventory(destination)
        old.local.write_json(destination/'snapshot_metadata.json',dict(offline_kernel_validation_only=True,
            terminal_result_saved=True,new_dynamic_attempts=0,new_controller_run=False,
            full_task_completed=False,hardware_readiness=False,qualification_executed=False,
            original_failed_test_preserved=not passed,original_kernel_changed=False))
        old.local.write_json(destination/'artifact_manifest.json',inventory(destination))
        subprocess.run(['git','add','-f',str(destination.relative_to(repo))],cwd=repo,check=True)
    for name in ('velocity_bias_kernel_r190.py','test_velocity_bias_kernel_r190.py','validate_velocity_bias_kernel_r190.py',
                 'test_velocity_bias_kernel_r190b.py','validate_velocity_bias_kernel_r190b.py',Path(__file__).name):
        dest=repo/'scripts'/name; assert not dest.exists()
        shutil.copy2(Path(__file__).with_name(name),dest)
        assert old.prior.digest(Path(__file__).with_name(name))==old.prior.digest(dest)
        subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    doc='GETUP_VELOCITY_BIAS_KERNEL_R190_R190B_20261010.md'
    assert not (repo/doc).exists(); shutil.copy2(root/'tmp'/doc,repo/doc)
    subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R190 branch-coupling regression failure and corrected R190b offline velocity-bias contract'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__': main()
