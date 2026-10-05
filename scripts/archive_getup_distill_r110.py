"""Append a unique completed-file snapshot; never overwrite earlier evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'
RUN=ROOT/'outputs/getup_distill_r110_left_20261005'
SMOKE=ROOT/'outputs/getup_distill_r110_smoke_20261005'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    smoke=json.loads((SMOKE/'results.json').read_text());assert smoke['nominal']['nominal_success']
    dest=REPO/'results'/RUN.name/args.snapshot;assert not dest.exists();dest.mkdir(parents=True)
    cwd=ROOT/'projects/Open_Duck_Playground'
    tested=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_distill_r110','-v'],
        cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (dest/'snapshot_regression_tests.log').write_text(tested.stdout)
    assert tested.returncode==0
    shutil.copytree(SMOKE,dest/'independent_smoke');shutil.copy2(SMOKE.with_suffix('.log'),dest/'smoke_parent.log')
    shutil.copytree(RUN/'executed_sources',dest/'executed_sources')
    for name in ('contract.json','progress.json','results.json'):
        if (RUN/name).exists():shutil.copy2(RUN/name,dest/name)
    training=RUN/'training';target=dest/'training';target.mkdir();closed=[]
    for report in sorted(training.glob('report_*.json')):
        update=int(report.stem.split('_')[-1]);number=f'{update:05d}'
        assert json.loads(report.read_text())['model_sha256']==digest(training/f'update_{number}.npz')
        for filename in (report.name,f'update_{number}.npz',f'learner_{number}.msgpack',f'rng_{number}.json'):
            shutil.copy2(training/filename,target/filename)
        closed.append(update)
    for name in ('data_manifest.json','zero_initial_actor.npz','history.json','progress.json','results.json'):
        if (training/name).exists():shutil.copy2(training/name,target/name)
    complete={}
    for group in sorted(RUN.iterdir()):
        if not group.is_dir() or not (group.name=='baseline' or group.name.startswith('development_') or group.name.startswith('qualification_')):continue
        saved=[]
        for case in sorted(group.glob('case_*')):
            if not (case/'result.json').exists():continue
            row=json.loads((case/'result.json').read_text());assert (case/'trajectory.npz').exists()
            shutil.copytree(case,dest/group.name/case.name);saved.append(row)
        complete[group.name]=saved
        if (group/'results.json').exists():shutil.copy2(group/'results.json',dest/group.name/'results.json')
    for name in ('frozen_candidate.npz','frozen_candidate.json'):
        if (RUN/name).exists():shutil.copy2(RUN/name,dest/name)
    shutil.copy2(RUN.with_suffix('.log'),dest/'parent_log_snapshot.log')
    snapshot=dict(closed_training_updates=closed,completed_trials=complete,terminal_result_saved=(dest/'results.json').exists(),
        snapshot_only=True,simulation_only=True,hardware_readiness=False,full_task_completed=False)
    (dest/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
    (dest/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(dest)):digest(f) for f in dest.rglob('*') if f.is_file()},indent=2))
    for name in ('train_getup_distill_r110.py','test_getup_distill_r110.py','launch_getup_distill_r110.py',Path(__file__).name):
        destination=REPO/'scripts'/name
        source=Path(__file__).with_name(name)
        if destination.exists():assert digest(destination)==digest(source)
        else:shutil.copy2(source,destination)
    doc='GETUP_CAUSAL_DISTILLATION_R110_20261005.md'
    if (REPO/doc).exists():assert digest(REPO/doc)==digest(Path(__file__).with_name(doc))
    else:shutil.copy2(Path(__file__).with_name(doc),REPO/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in ('train_getup_distill_r110.py','test_getup_distill_r110.py','launch_getup_distill_r110.py',Path(__file__).name)]],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(dest.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve causal-context get-up distillation R110 '+args.snapshot],cwd=REPO,check=True)
    bundle=ROOT/'tmp'/('getup_distill_r110_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        closed_training_updates=closed,completed_trial_counts={k:len(v) for k,v in complete.items()},terminal_result_saved=snapshot['terminal_result_saved'])),flush=True)


if __name__=='__main__':main()
