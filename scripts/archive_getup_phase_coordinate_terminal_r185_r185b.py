"""Unique closed phase-coordinate evidence; preserve initialization failure."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import probe_getup_phase_coordinate_r185b as run


def hashes(path):return {str(p.relative_to(path)):run.prior.digest(p) for p in path.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')) for line in processes)
    assert not any('git' in line and 'push' in line and 'server-publish' in line for line in processes)
    result=json.loads((run.OUTPUT/'results.json').read_text());smoke=json.loads((run.SMOKE/'results.json').read_text())
    assert result['terminal_result_saved'] and smoke['terminal_result_saved'] and result['formal_dynamic_attempts']==56
    run.compare_smoke(run.OUTPUT)
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==56 and len(list(run.SMOKE.rglob('trajectory.npz')))==6
    pairings=[];signal_summary=[]
    for case in run.CASES:
        baseline=run.OUTPUT/'development_baseline'/f'case_{case}';candidate=run.OUTPUT/'development_candidate'/f'case_{case}'
        original=run.selector.OUTPUT/'candidate'/f'case_{case}'
        b,c,o=[json.loads((path/'result.json').read_text()) for path in (baseline,candidate,original)]
        assert b['initial_hash']==c['initial_hash']==o['initial_hash'] and b['peaks']==o['peaks']
        # Isolated whole-field parity only; root arrays never controller inputs.
        with np.load(baseline/'trajectory.npz',allow_pickle=False) as z,np.load(candidate/'trajectory.npz',allow_pickle=False) as y,np.load(original/'trajectory.npz',allow_pickle=False) as x:
            be={k:bool(np.array_equal(z[k],x[k])) for k in x.files};assert all(be.values())
            ce={k:bool(np.array_equal(y[k],x[k])) for k in x.files}
            request=y['phase_reference_shift_rad'];direct=y['same_state_pre_slew_direct_new_rad']
            post=y['planned_before_integration_rad']-y['same_state_unperturbed_planned_rad']
            first=np.flatnonzero(np.any(post!=0,axis=1))
            signal_summary.append(dict(case=case,raw_reference_shift_max_rad=float(np.abs(request).max()),phase_offset_max_controls=float(np.abs(y['phase_offset_controls']).max()),same_state_pre_slew_direct_max_rad=float(np.abs(direct).max()),same_state_post_slew_direct_max_rad=float(np.abs(post).max()),first_nonzero_direct_control=int(first[0]) if len(first) else None))
        pairings.append(dict(case=case,baseline_all_original_fields_equal=be,candidate_all_original_fields_equal=ce,initial_hash=b['initial_hash'],baseline_success=b['success'],candidate_success=c['success'],candidate_valid=c['valid'],candidate_peaks=c['peaks'],baseline_peaks=b['peaks']))
    failed=run.ROOT/'outputs/getup_phase_coordinate_r185_smoke_20261010'
    assert failed.exists() and not (failed/'results.json').exists() and not list(failed.rglob('result.json'))
    for source,phase,label in ((failed,'failure_snapshot_01','r185'),(run.SMOKE,'terminal_snapshot','r185b'),(run.OUTPUT,'terminal_snapshot','r185b')):
        target=repo/'results'/source.name/phase;assert not target.exists()
        before=hashes(source);shutil.copytree(source,target);assert before==hashes(source)==hashes(target)
        contract=json.loads((source/'contract.json').read_text())
        for path,sha in contract['hashes'].items():assert run.prior.digest(path)==sha
        main_name=f'probe_getup_phase_coordinate_{label}.py'
        assert run.prior.digest(source/'executed_sources'/main_name)==run.prior.digest(Path(__file__).with_name(main_name))
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        for suffix in ('initial_regression','formal_regression','launcher'):
            log=run.ROOT/'tmp'/f'getup_phase_coordinate_{label}_{suffix}_20261010.log'
            if log.exists():shutil.copy2(log,target/log.name)
        if source==failed:
            run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=False,execution_initialization_failure=True,zero_parity_jobs_dispatched=3,closed_dynamic_attempts_saved=0,recovery_step_target_not_reached=True,preparation_reset_was_invoked=True,preparation_frames_not_saved=True,partial_frames_saved=False,no_claim_of_zero_physics_integration=True,formal_started=False,full_task_completed=False,hardware_readiness=False))
        else:
            run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,fixed_rule=True,learned_parameters=0,saved_dynamic_attempts=6 if source==run.SMOKE else 56,training_generations=0,full_task_completed=False,hardware_readiness=False,qualification_executed=False))
            if source==run.OUTPUT:
                run.local.write_json(target/'original_terminal_pairing.json',dict(pairings=pairings,root_arrays_only_separate_equality_check=True))
                run.local.write_json(target/'retrospective_summary.json',dict(signal_summary=signal_summary,not_independent_scalar_reconstruction=True,raw_reference_shift_is_not_executed_extra=True,not_dynamic_counterfactual=True,rescued=[p['case'] for p in pairings if p['candidate_success'] and not p['baseline_success']],regressed=[p['case'] for p in pairings if p['baseline_success'] and not p['candidate_success']]))
        run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ('probe_getup_phase_coordinate_r185.py','test_getup_phase_coordinate_r185.py','launch_getup_phase_coordinate_r185.py','probe_getup_phase_coordinate_r185b.py','test_getup_phase_coordinate_r185b.py','launch_getup_phase_coordinate_r185b.py','inspect_getup_phase_coordinate_r185b.py','summarize_getup_phase_coordinate_terminal_r185b.py',Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    for name in ('GETUP_PHASE_COORDINATE_PLAN_R185_20261010.md','GETUP_PHASE_COORDINATE_TERMINAL_R185_R185B_20261010.md'):
        assert not (repo/name).exists();shutil.copy2(run.ROOT/'tmp'/name,repo/name);subprocess.run(['git','add',name],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R185 snapshot loading failure and R185b fixed local phase terminal evidence'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
