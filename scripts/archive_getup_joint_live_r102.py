"""New immutable live snapshot, closed generation only; never restart training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    source=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    progress=json.loads((source/'progress.json').read_text());n=progress['completed_generations']
    history=json.loads((source/'history.json').read_text())[:n]
    assert len(history)==n and history[-1]['generation']==n
    baseline=history[0]['reports'][0]
    matches=[r for g in history for gains,r in zip(g['candidates'],g['reports'])
        if np.array_equal(gains,progress['best_gains'])]
    assert matches and all(r==matches[0] for r in matches)
    selected=matches[0];base_rows={r['case_seed']:r for r in baseline['rows']}
    assert all(r['initial_hash']==base_rows[r['case_seed']]['initial_hash'] for r in selected['rows'])
    comparison=dict(closed_generation=n,training_only=True,continuous_30s_validation_pending=True,
        baseline_successes=baseline['successes'],selected_successes=selected['successes'],
        selected_nominal_success=selected['nominal_success'],selected_physical_failures=selected['physical_failures'],
        regressions=[r['case_seed'] for r in selected['rows'] if base_rows[r['case_seed']]['success'] and not r['success']],
        recoveries=[r['case_seed'] for r in selected['rows'] if not base_rows[r['case_seed']]['success'] and r['success']],
        all_initial_hashes_match=True,selected=selected,simulation_only=True,hardware_readiness=False,
        full_task_completed=False)
    target=REPO/'results'/source.name/f'closed_generation_{n:04d}';target.mkdir(exist_ok=False)
    for i in range(1,n+1):
        name=f'checkpoint_{i:04d}.npz';shutil.copy2(source/name,target/name)
    for name,data in [('progress.json',progress),('history.json',history),('training_comparison.json',comparison)]:
        (target/name).write_text(json.dumps(data,indent=2))
    scope=dict(simulation_only=True,full_task_completed=False,hardware_readiness=False,
        closed_generation=n,short_tail_training_labels_only=True,development_qualification_pending=True,
        immutable_training_sources_unmodified=True,no_new_experiment_launched=True)
    (target/'archive_scope.json').write_text(json.dumps(scope,indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f)
        for f in target.rglob('*') if f.is_file()},indent=2))
    name=Path(__file__).name;shutil.copy2(Path(__file__),REPO/'scripts'/name)
    subprocess.run(['git','add','scripts/'+name],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m',f'Archive R102 closed generation {n} and paired short-tail comparison'],cwd=REPO,check=True)
    bundle=ROOT/f'tmp/getup_joint_live_r102_g{n:04d}_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print(json.dumps(dict(bundle=str(bundle),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        generation=n,successes=selected['successes'],physical_failures=selected['physical_failures'],
        regressions=comparison['regressions'],recoveries=comparison['recoveries'])),flush=True)


if __name__=='__main__':main()
