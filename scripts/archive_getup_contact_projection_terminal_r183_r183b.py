"""Unique closed evidence archive; preserve original assertion and data gap."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import probe_getup_contact_projection_r183b as run


def hashes(path):return {str(p.relative_to(path)):run.prior.digest(p) for p in path.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    processes=subprocess.check_output(['ps','-u',str(__import__('os').getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')) for line in processes),'Wait natural exit; no signals'
    result=json.loads((run.OUTPUT/'results.json').read_text());smoke=json.loads((run.SMOKE/'results.json').read_text())
    assert result['terminal_result_saved'] and smoke['terminal_result_saved'] and result['formal_dynamic_attempts']==56
    run.compare_smoke(run.OUTPUT)
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==56 and len(list(run.SMOKE.rglob('trajectory.npz')))==6
    pairings=[]
    signal_summary=[]
    for case in run.CASES:
        baseline=run.OUTPUT/'development_baseline'/f'case_{case}';candidate=run.OUTPUT/'development_candidate'/f'case_{case}'
        original=run.selector.OUTPUT/'candidate'/f'case_{case}'
        b=json.loads((baseline/'result.json').read_text());c=json.loads((candidate/'result.json').read_text());o=json.loads((original/'result.json').read_text())
        assert b['initial_hash']==c['initial_hash']==o['initial_hash'] and b['peaks']==o['peaks']
        with np.load(baseline/'trajectory.npz',allow_pickle=False) as z,np.load(candidate/'trajectory.npz',allow_pickle=False) as y,np.load(original/'trajectory.npz',allow_pickle=False) as x:
            be={k:bool(np.array_equal(z[k],x[k])) for k in x.files};assert all(be.values())
            ce={k:bool(np.array_equal(y[k],x[k])) for k in x.files}
        pairings.append(dict(case=case,baseline_all_original_fields_equal=be,candidate_all_original_fields_equal=ce,initial_hash=b['initial_hash'],baseline_success=b['success'],candidate_success=c['success'],candidate_valid=c['valid']))
        with np.load(candidate/'trajectory.npz',allow_pickle=False) as z:
            raw_double=np.all(z['observations'][:529,48:50]>0,axis=1)
            active=z['contact_projection_active'][:529].astype(bool)
            extra=np.max(np.abs(z['projection_extra_rad'][:529]),axis=1)
            changed=np.flatnonzero(extra>0)
            signal_summary.append(dict(case=case,raw_double_contact_recovery_controls=int(np.count_nonzero(raw_double)),feasible_active_controls=int(np.count_nonzero(active)),maximum_projection_extra_rad=float(extra.max()),first_nonzero_added_target_control=int(changed[0]) if len(changed) else None))
    failed=run.ROOT/'outputs/getup_contact_projection_r183_smoke_20261010'
    assert failed.exists() and not (failed/'results.json').exists() and not list(failed.rglob('result.json'))
    for source,phase in ((failed,'failure_snapshot_01'),(run.SMOKE,'terminal_snapshot'),(run.OUTPUT,'terminal_snapshot')):
        target=repo/'results'/source.name/phase;assert not target.exists()
        before=hashes(source);shutil.copytree(source,target);assert before==hashes(source)==hashes(target)
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        label='r183' if source==failed else 'r183b'
        for suffix in ('initial_regression','formal_regression','launcher'):
            log=run.ROOT/'tmp'/f'getup_contact_projection_{label}_{suffix}_20261010.log'
            if log.exists():shutil.copy2(log,target/log.name)
        if source==failed:
            run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=False,execution_assertion_failure=True,dynamic_startup_jobs_dispatched=3,closed_dynamic_attempts_saved=0,partial_frames_saved=False,missing_partial_frame_evidence_not_reconstructed=True,formal_started=False,physical_labels_not_reclassified=True,full_task_completed=False,hardware_readiness=False))
        else:
            run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,fixed_rule=True,learned_parameters=0,saved_dynamic_attempts=6 if source==run.SMOKE else 56,training_generations=0,full_task_completed=False,hardware_readiness=False,qualification_executed=False))
            if source==run.OUTPUT:
                run.local.write_json(target/'original_terminal_pairing.json',dict(pairings=pairings,root_arrays_only_separate_equality_check=True))
                run.local.write_json(target/'retrospective_summary.json',dict(signal_summary=signal_summary,not_independent_scalar_reconstruction=True,source_capture_limitation='Main module was not copied into executed_sources at startup; exact uploaded local source and current server source are saved retrospectively, not relabeled as startup capture.',rescued=[p['case'] for p in pairings if p['candidate_success'] and not p['baseline_success']],regressed=[p['case'] for p in pairings if p['baseline_success'] and not p['candidate_success']]))
        run.local.write_json(target/'artifact_hashes.json',hashes(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ('probe_getup_contact_projection_r183.py','test_getup_contact_projection_r183.py','launch_getup_contact_projection_r183.py','probe_getup_contact_projection_r183b.py','test_getup_contact_projection_r183b.py','launch_getup_contact_projection_r183b.py','inspect_getup_contact_projection_r183b.py','summarize_getup_contact_projection_terminal_r183b.py',Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add',str(dest.relative_to(repo))],cwd=repo,check=True)
    for name in ('GETUP_CONTACT_PROJECTION_PLAN_R183_20261010.md','GETUP_CONTACT_PROJECTION_TERMINAL_R183_R183B_20261010.md'):
        assert not (repo/name).exists();shutil.copy2(run.ROOT/'tmp'/name,repo/name);subprocess.run(['git','add',name],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R183 planner assertion and R183b fixed paired-contact projection terminal evidence'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
