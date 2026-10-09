"""Unique failure and closed fixed-rule snapshots, never overwrite evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import probe_getup_velocity_bias_r191 as run


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
    assert shutil.disk_usage(run.ROOT).free>10*1024**3
    assert sum(p.stat().st_size for root in (run.OUTPUT,run.SMOKE,run.ROOT/'outputs/getup_velocity_bias_launcher_r191_20261010') for p in root.rglob('*') if p.is_file())*2 < 2*1024**3
    commands=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in s and any(v in s for v in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')) for s in commands)
    assert not any('git' in s and 'push' in s and 'server-publish' in s for s in commands)
    launcher=run.ROOT/'outputs/getup_velocity_bias_launcher_r191_20261010'
    assert json.loads((launcher/'result.json').read_text())['natural_exit']
    for phase in ('initial','formal'):
        data=launcher/phase;tm=json.loads((data/'test_source_manifest.json').read_text())
        for path,info in tm.items():assert run.prior.digest(path)==info['sha256']==run.prior.digest(data/'executed_sources'/info['copy'])
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

    for source in (run.SMOKE,run.OUTPUT):
        verify_sources(source)
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        before=hashes(source);shutil.copytree(source,target);assert before==hashes(source)==hashes(target)
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,fixed_rule=True,learned_parameters=0,search_trials=0,saved_dynamic_attempts=6 if source==run.SMOKE else 56,full_task_completed=False,hardware_readiness=False,qualification_executed=False,kernel_unchanged=True))
        if source==run.OUTPUT:
            statistics=[]
            for row in formal['candidate']['rows']:
                case=row['case_seed'];path=source/'development_candidate'/f'case_{case}'/'trajectory.npz'
                with np.load(path,allow_pickle=False) as z:
                    # Scalar whitelist only; root arrays above were equality only.
                    raw=z['velocity_bias_request_rad'];error=z['velocity_bias_error_nm']
                    post=z['planned_before_integration_rad']-z['same_state_unperturbed_planned_rad']
                    pre=z['same_state_pre_slew_direct_new_rad'];idx=np.flatnonzero(np.any(post!=0,axis=1))
                    assert len(raw)==row['controls']
                    statistics.append(dict(case=case,controls=row['controls'],success=row['success'],valid=row['valid'],peaks=row['peaks'],
                        first_post_slew_difference_control=None if not len(idx) else int(idx[0]),
                        maximum_raw_rad=float(np.max(np.abs(raw))),maximum_error_nm=float(np.max(np.abs(error))),
                        maximum_pre_slew_difference_rad=float(np.max(np.abs(pre))),maximum_post_slew_difference_rad=float(np.max(np.abs(post))),
                        maximum_combined_14_joint_correction_rad=row['maximum_combined_14_joint_correction_rad']))
            run.local.write_json(target/'original_terminal_pairing.json',dict(pairings=pairings,root_arrays_only_separate_equality_check=True))
            run.local.write_json(target/'retrospective_summary.json',dict(not_independent_scalar_reconstruction=True,not_dynamic_counterfactual=True,
                rescued=[p['case'] for p in pairings if p['candidate_success'] and not p['baseline_success']],
                regressed=[p['case'] for p in pairings if p['baseline_success'] and not p['candidate_success']],
                candidate_statistics=statistics))
        run.local.write_json(target/'artifact_hashes.json',hashes(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/launcher.name/'terminal_snapshot';assert not target.exists()
    before=hashes(launcher);shutil.copytree(launcher,target);assert before==hashes(launcher)==hashes(target)
    shutil.copy2(run.ROOT/'tmp/getup_velocity_bias_r191_launcher_20261010.log',target/'process.log')
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    names=['probe_getup_velocity_bias_r191.py','test_getup_velocity_bias_r191.py','launch_getup_velocity_bias_r191.py','publish_getup_velocity_bias_terminal_r191_20261010.py',Path(__file__).name]
    for name in names:
        destination=repo/'scripts'/name;assert not destination.exists();shutil.copy2(Path(__file__).with_name(name),destination);subprocess.run(['git','add',str(destination.relative_to(repo))],cwd=repo,check=True)
    for name in ('GETUP_VELOCITY_BIAS_PLAN_R191_20261010.md','GETUP_VELOCITY_BIAS_TERMINAL_R191_20261010.md'):
        assert not (repo/name).exists();shutil.copy2(run.ROOT/'tmp'/name,repo/name);subprocess.run(['git','add',name],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R191 fixed velocity bias full-episode terminal evidence and original gates'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
