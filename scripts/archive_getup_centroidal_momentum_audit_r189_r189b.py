"""Unique closed read-only evidence and original failed regression archive."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import audit_getup_centroidal_momentum_terminal_r189b as audit

def inventory(root):return {str(p.relative_to(root)):audit.digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=audit.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in p and any(s in p for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')) for p in processes)
    assert not any('git' in p and 'push' in p and 'server-publish' in p for p in processes)
    launcher=audit.read(audit.ROOT/'tmp/getup_centroidal_momentum_audit_r189b_launcher_result_20261010.json')
    assert launcher['natural_exit'] and launcher['exit_code']==0 and launcher['new_dynamic_trajectories']==0
    formal=audit.read(audit.OUTPUT/'results.json');smoke=audit.read(audit.SMOKE/'results.json')
    for result,n in ((formal,62),(smoke,6)):
        assert result['terminal_result_saved'] and result['all_scalar_and_limits_bitwise'] and result['source_evidence_unchanged']
        assert result['existing_trajectories_audited']==n and len(result['rows'])==n and len(result['terminal_pairs'])==25
        assert result['new_dynamic_trajectories']==0 and not result['original_IMU_double_request_reconstructed']
        assert not result['full_task_completed'] and not result['hardware_readiness'] and not result['qualification']
    assert formal['terminal_pairs']==smoke['terminal_pairs']
    for prior in smoke['rows']:
        assert prior==next(r for r in formal['rows'] if r['identity']==prior['identity'])
        ident=prior['identity']
        with np.load(audit.OUTPUT/'signals'/ident/'signals.npz',allow_pickle=False) as a,np.load(audit.SMOKE/'signals'/ident/'signals.npz',allow_pickle=False) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    for source,n in ((audit.SMOKE,6),(audit.OUTPUT,62)):
        tracked=audit.read(source/'source_hashes.json')
        assert tracked['unchanged'] and tracked['before']==tracked['after'] and tracked['compiled_model_unchanged']
        for p,sha in tracked['before'].items():assert audit.digest(p)==sha
        manifest=audit.read(source/'executed_sources_manifest.json')
        assert str(Path(audit.__file__).resolve()) in manifest
        for p,info in manifest.items():assert audit.digest(p)==info['sha256']==audit.digest(source/'executed_sources'/info['copy'])
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        before=inventory(source);shutil.copytree(source,target);assert before==inventory(source)==inventory(target)
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        for suffix in ('initial_regression','formal_regression','launcher'):
            log=audit.ROOT/'tmp'/f'getup_centroidal_momentum_audit_r189b_{suffix}_20261010.log';assert log.exists();shutil.copy2(log,target/log.name)
        shutil.copy2(audit.ROOT/'tmp/getup_centroidal_momentum_audit_r189b_launcher_result_20261010.json',target/'launcher_result.json')
        audit.write(target/'snapshot_metadata.json',dict(terminal_result_saved=True,read_only=True,existing_trajectories_audited=n,
         new_dynamic_trajectories=0,original_double_IMU_request_reconstructed=False,historical_gaps_recovered=False,
         full_task_completed=False,hardware_readiness=False,qualification_executed=False))
        audit.write(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    fail=repo/'results/getup_centroidal_momentum_audit_r189_20261010/failure_snapshot_01';assert not fail.exists();fail.mkdir(parents=True)
    for name in ('audit_getup_centroidal_momentum_terminal_r189.py','test_getup_centroidal_momentum_audit_r189.py','launch_getup_centroidal_momentum_audit_r189.py'):
        source=Path(audit.__file__).with_name(name);shutil.copy2(source,fail/name);assert audit.digest(source)==audit.digest(fail/name)
    for name in ('getup_centroidal_momentum_audit_r189_initial_regression_20261010.log','getup_centroidal_momentum_audit_r189_launcher_20261010.log'):
        shutil.copy2(audit.ROOT/'tmp'/name,fail/name)
    audit.write(fail/'failure_scope.json',dict(regressions_run=10,regressions_passed=9,regressions_failed=1,
     root_translation_matrix_max_roundoff=1.734723475976807e-17,existing_trajectory_audit_started=False,
     new_dynamic_trajectories=0,original_regression_startup_source_copy=False,
     archived_source_copies_are_post_exit_unchanged=True,physical_or_controller_rule_changed=False))
    audit.write(fail/'artifact_manifest.json',inventory(fail));subprocess.run(['git','add','-f',str(fail.relative_to(repo))],cwd=repo,check=True)
    names=('audit_getup_centroidal_momentum_terminal_r189.py','test_getup_centroidal_momentum_audit_r189.py','launch_getup_centroidal_momentum_audit_r189.py',
     'audit_getup_centroidal_momentum_terminal_r189b.py','test_getup_centroidal_momentum_audit_r189b.py','launch_getup_centroidal_momentum_audit_r189b.py',Path(__file__).name)
    for name in names:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    doc='GETUP_CENTROIDAL_MOMENTUM_AUDIT_R189_R189B_20261010.md';assert not (repo/doc).exists();shutil.copy2(audit.ROOT/'tmp'/doc,repo/doc)
    subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R189 failed regression and R189b independent closed momentum and target audit'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)

if __name__=='__main__':main()
