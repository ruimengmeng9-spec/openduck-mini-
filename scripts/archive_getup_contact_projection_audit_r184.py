"""Unique immutable R184 read-only terminal archive; no old evidence repair."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import audit_getup_contact_projection_terminal_r184 as audit


def inventory(path):
    return {str(p.relative_to(path)):audit.digest(p) for p in path.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=audit.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in p and any(s in p for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')) for p in processes)
    assert not any('git' in p and 'push' in p and 'server-publish' in p for p in processes)
    formal=audit.read(audit.OUTPUT/'results.json');smoke=audit.read(audit.SMOKE/'results.json')
    for result,n in ((formal,62),(smoke,6)):
        assert result['terminal_result_saved'] and result['all_scalar_bitwise'] and result['source_evidence_unchanged']
        assert result['existing_trajectories_audited']==n and len(result['rows'])==n and len(result['terminal_pairs'])==25
        assert result['new_dynamic_trajectories']==0 and not result['original_IMU_double_request_reconstructed']
        assert not result['full_task_completed'] and not result['hardware_readiness'] and not result['qualification']
    assert formal['terminal_pairs']==smoke['terminal_pairs']
    for prior in smoke['rows']:
        assert prior==next(row for row in formal['rows'] if row['identity']==prior['identity'])
        ident=prior['identity']
        with np.load(audit.OUTPUT/'signals'/ident/'signals.npz',allow_pickle=False) as a,np.load(audit.SMOKE/'signals'/ident/'signals.npz',allow_pickle=False) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    assert audit.digest(Path(audit.__file__))=='f9ddeb8a3454e124622aa9570efb15ae71718f1a73515a17dde65a6be7555da6'
    for source,n in ((audit.SMOKE,6),(audit.OUTPUT,62)):
        tracked=audit.read(source/'source_hashes.json')
        assert tracked['unchanged'] and tracked['before']==tracked['after']
        for path,sha in tracked['before'].items():assert audit.digest(path)==sha
        assert audit.digest(source/'executed_sources'/Path(audit.__file__).name)==audit.digest(Path(audit.__file__))
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        before=inventory(source);shutil.copytree(source,target)
        assert before==inventory(source)==inventory(target)
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        for suffix in ('initial_regression','formal_regression','launcher'):
            log=audit.ROOT/'tmp'/f'getup_contact_projection_audit_r184_{suffix}_20261010.log'
            assert log.exists();shutil.copy2(log,target/log.name)
        audit.write(target/'snapshot_metadata.json',dict(terminal_result_saved=True,read_only=True,existing_trajectories_audited=n,new_dynamic_trajectories=0,original_double_IMU_request_reconstructed=False,old_partial_frames_recovered=False,old_main_startup_capture_repaired=False,full_task_completed=False,hardware_readiness=False,qualification_executed=False))
        audit.write(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ('audit_getup_contact_projection_terminal_r184.py','test_getup_contact_projection_audit_r184.py','launch_getup_contact_projection_audit_r184.py',Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    doc='GETUP_CONTACT_PROJECTION_AUDIT_R184_20261010.md'
    assert not (repo/doc).exists();shutil.copy2(audit.ROOT/'tmp'/doc,repo/doc)
    subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R184 independent contact projection and limit decision audit'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
