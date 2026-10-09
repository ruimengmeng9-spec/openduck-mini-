"""Append immutable offline geometry contract and first failure, no dynamic data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from diagnostics import probe_getup_phase_coordinate_r185b as old


def inventory(path):
    return {str(p.relative_to(path)):old.prior.digest(p) for p in path.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=old.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in p and any(s in p for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_centroidal_')) for p in processes)
    assert not any('git' in p and 'push' in p and 'server-publish' in p for p in processes)
    for suffix,passed,n in [('r187',False,14),('r187b',True,15)]:
        source=old.ROOT/'outputs'/f'getup_centroidal_momentum_kernel_{suffix}_20261010'
        result=json.loads((source/'results.json').read_text())
        assert result['terminal_result_saved'] and result['offline_kernel_validation_only']
        assert result['tests_passed']==passed and result['regression_tests']==n
        assert result['new_dynamic_attempts']==0 and not result['new_controller_run']
        assert result['source_input_hashes_before']==result['source_input_hashes_after']
        for path,sha in result['source_input_hashes_before'].items():assert old.prior.digest(Path(path))==sha
        for p in (source/'executed_sources').iterdir():
            assert old.prior.digest(p)==old.prior.digest(Path(__file__).with_name(p.name))
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        before=inventory(source);shutil.copytree(source,target)
        assert before==inventory(source)==inventory(target)
        old.local.write_json(target/'snapshot_metadata.json',dict(terminal_result_saved=True,offline_only=True,
            tests_passed=passed,new_dynamic_attempts=0,new_controller_run=False,full_task_completed=False,
            hardware_readiness=False,qualification_executed=False,first_failure_preserved=True))
        old.local.write_json(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
        for stem in ('centroidal_momentum_kernel_','test_centroidal_momentum_kernel_','validate_centroidal_momentum_kernel_'):
            name=stem+suffix+'.py';dest=repo/'scripts'/name;assert not dest.exists()
            shutil.copy2(Path(__file__).with_name(name),dest)
            subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    dest=repo/'scripts'/Path(__file__).name;assert not dest.exists();shutil.copy2(__file__,dest)
    subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    doc='GETUP_CENTROIDAL_MOMENTUM_KERNEL_R187_R187B_20261010.md'
    assert not (repo/doc).exists();shutil.copy2(old.ROOT/'tmp'/doc,repo/doc)
    subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R187 first gyro mounting assertion failure and R187b causal momentum kernel validation'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
