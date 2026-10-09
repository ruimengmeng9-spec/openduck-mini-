"""Unique failure and closed fixed-rule snapshots, never overwrite evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import probe_getup_centroidal_momentum_r188b as run


def hashes(path):return {str(p.relative_to(path)):run.prior.digest(p) for p in path.rglob('*') if p.is_file()}


def verify_sources(source):
    contract=json.loads((source/'contract.json').read_text())
    manifests=[(source/'executed_sources',contract['sources'])]
    for path in source.glob('worker_source_manifest_*.json'):
        pid=path.stem.rsplit('_',1)[1]
        manifests.append((source/f'worker_executed_sources_{pid}',json.loads(path.read_text())))
    for folder,manifest in manifests:
        for actual,info in manifest.items():assert run.prior.digest(actual)==info['sha256']==run.prior.digest(folder/info['copy'])
    assert contract['model_files']==run.model_files(run.local.prior.program.SCENE)
    for path,sha in contract['hashes'].items():assert run.prior.digest(path)==sha
    assert json.loads((source/'input_hashes_after.json').read_text())['source_and_model_files_unchanged']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    commands=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in s and any(v in s for v in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')) for s in commands)
    assert not any('git' in s and 'push' in s and 'server-publish' in s for s in commands)
    launcher=run.ROOT/'outputs/getup_centroidal_momentum_launcher_r188c_20261010'
    assert json.loads((launcher/'result.json').read_text())['natural_exit']
    manifest=json.loads((launcher/'contract.json').read_text())['sources']
    for path,info in manifest.items():assert run.prior.digest(path)==info['sha256']==run.prior.digest(launcher/'executed_sources'/info['copy'])
    formal=json.loads((run.OUTPUT/'results.json').read_text());smoke=json.loads((run.SMOKE/'results.json').read_text())
    assert formal['terminal_result_saved'] and smoke['terminal_result_saved'] and formal['formal_dynamic_attempts']==56
    run.compare_smoke(run.OUTPUT)
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==56 and len(list(run.SMOKE.rglob('trajectory.npz')))==6
    pairings=[]
    for case in run.CASES:
        bpath=run.OUTPUT/'development_baseline'/f'case_{case}';cpath=run.OUTPUT/'development_candidate'/f'case_{case}';opath=run.selector.OUTPUT/'candidate'/f'case_{case}'
        b,c,o=[json.loads((p/'result.json').read_text()) for p in (bpath,cpath,opath)]
        assert b['initial_hash']==c['initial_hash']==o['initial_hash'] and b['peaks']==o['peaks']
        # Root arrays decoded only for isolated original whole-field equality.
        with np.load(bpath/'trajectory.npz',allow_pickle=False) as z,np.load(cpath/'trajectory.npz',allow_pickle=False) as y,np.load(opath/'trajectory.npz',allow_pickle=False) as x:
            be={k:bool(np.array_equal(z[k],x[k])) for k in x.files};assert all(be.values())
            ce={k:bool(np.array_equal(y[k],x[k])) for k in x.files}
        pairings.append(dict(case=case,initial_hash=b['initial_hash'],baseline_all_original_fields_equal=be,candidate_all_original_fields_equal=ce,baseline_success=b['success'],candidate_success=c['success'],candidate_valid=c['valid'],baseline_peaks=b['peaks'],candidate_peaks=c['peaks']))
    failed=run.ROOT/'outputs/getup_centroidal_momentum_r188_smoke_20261010'
    assert not (failed/'results.json').exists() and not list(failed.rglob('trajectory.npz'))
    failure_rows=[json.loads(p.read_text()) for p in failed.rglob('failure.json')]
    assert len(failure_rows)==3 and all(r['controls_completed']==0 and r['prepare_frames_saved']==80 for r in failure_rows)
    for source,phase,label in ((failed,'failure_snapshot_01','r188'),(run.SMOKE,'terminal_snapshot','r188b'),(run.OUTPUT,'terminal_snapshot','r188b')):
        verify_sources(source)
        target=repo/'results'/source.name/phase;assert not target.exists()
        before=hashes(source);shutil.copytree(source,target);assert before==hashes(source)==hashes(target)
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        for suffix in ('initial_regression','formal_regression','launcher'):
            log=run.ROOT/'tmp'/f'getup_centroidal_momentum_{label}_{suffix}_20261010.log'
            if log.exists():shutil.copy2(log,target/log.name)
        if source==failed:
            snapshot=dict(terminal_result_saved=False,implementation_failure=True,closed_recovery_attempts_saved=0,prepare_frames_per_dispatched_job=80,prepare_partial_saved=True,recovery_controls_completed=0,formal_started=False,no_claim_of_zero_physics_integration=True)
        else:
            snapshot=dict(terminal_result_saved=True,fixed_rule=True,learned_parameters=0,search_trials=0,saved_dynamic_attempts=6 if source==run.SMOKE else 56,independent_qualification_run=False)
        snapshot.update(full_task_completed=False,hardware_readiness=False,qualification_executed=False,kernel_unchanged=True)
        run.local.write_json(target/'snapshot.json',snapshot)
        if source==run.OUTPUT:
            run.local.write_json(target/'original_terminal_pairing.json',dict(pairings=pairings,root_arrays_only_separate_equality_check=True))
            run.local.write_json(target/'retrospective_summary.json',dict(not_independent_scalar_reconstruction=True,not_dynamic_counterfactual=True,rescued=[p['case'] for p in pairings if p['candidate_success'] and not p['baseline_success']],regressed=[p['case'] for p in pairings if p['baseline_success'] and not p['candidate_success']]))
        run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/launcher.name/'terminal_snapshot';assert not target.exists()
    before=hashes(launcher);shutil.copytree(launcher,target);assert before==hashes(launcher)==hashes(target)
    shutil.copy2(run.ROOT/'tmp/getup_centroidal_momentum_r188c_launcher_20261010.log',target/'process.log')
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    names=[f'{kind}_getup_centroidal_momentum_{label}.py' for label in ('r188','r188b') for kind in ('probe','test','launch')]+['launch_getup_centroidal_momentum_r188c.py','summarize_getup_centroidal_momentum_r188b.py',Path(__file__).name]
    for name in names:
        destination=repo/'scripts'/name;assert not destination.exists();shutil.copy2(Path(__file__).with_name(name),destination);subprocess.run(['git','add',str(destination.relative_to(repo))],cwd=repo,check=True)
    for name in ('GETUP_CENTROIDAL_MOMENTUM_PLAN_R188_20261010.md','GETUP_CENTROIDAL_MOMENTUM_TERMINAL_R188_R188B_20261010.md'):
        assert not (repo/name).exists();shutil.copy2(run.ROOT/'tmp'/name,repo/name);subprocess.run(['git','add',name],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R188 preparation capture failure and R188b fixed momentum full-episode terminal evidence'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
