"""Append immutable closed R113 evidence; simulation only, no credentials."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'
FORMAL=ROOT/'outputs/getup_program_r113_left_20261005'
SMOKE=ROOT/'outputs/getup_program_r113_smoke_20261005'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    assert (SMOKE/'results.json').exists()
    target=REPO/'results'/FORMAL.name/args.snapshot;assert not target.exists()
    target.mkdir(parents=True)
    shutil.copytree(SMOKE,target/'independent_smoke');shutil.copy2(SMOKE.with_suffix('.log'),target/'smoke_parent.log')
    if FORMAL.exists():
        shutil.copytree(FORMAL/'executed_sources',target/'executed_sources')
        for name in ('contract.json','progress.json','results.json','frozen_candidate.npz','frozen_candidate.json'):
            if (FORMAL/name).exists():shutil.copy2(FORMAL/name,target/name)
        training=FORMAL/'training';dest=target/'training';dest.mkdir();closed=[]
        for report in sorted(training.glob('report_*.json')):
            update=int(report.stem.split('_')[-1]);number=f'{update:05d}'
            assert json.loads(report.read_text())['model_sha256']==digest(training/f'update_{number}.npz')
            for name in (report.name,f'update_{number}.npz',f'learner_{number}.msgpack',f'rng_{number}.json'):
                shutil.copy2(training/name,dest/name)
            closed.append(update)
        for name in ('data_manifest.json','initial_encoder.npz','history.json','progress.json','results.json'):
            if (training/name).exists():shutil.copy2(training/name,dest/name)
        completed={}
        for group in sorted(FORMAL.iterdir()):
            if not group.is_dir() or not (group.name.startswith('development_') or group.name.startswith('qualification_')):continue
            saved=[]
            for case in sorted(group.glob('case_*')):
                if (case/'result.json').exists():
                    shutil.copytree(case,target/group.name/case.name);saved.append(json.loads((case/'result.json').read_text()))
            completed[group.name]=saved
            if (group/'results.json').exists():shutil.copy2(group/'results.json',target/group.name/'results.json')
        shutil.copy2(FORMAL.with_suffix('.log'),target/'parent_log_snapshot.log')
    else:closed=[];completed={}
    cwd=ROOT/'projects/Open_Duck_Playground'
    test=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_program_r113','-v'],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'snapshot_regression_tests.log').write_text(test.stdout);assert test.returncode==0
    (target/'snapshot.json').write_text(json.dumps(dict(closed_updates=closed,completed_trial_counts={k:len(v) for k,v in completed.items()},
        terminal_result_saved=(target/'results.json').exists(),full_task_completed=False,hardware_readiness=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=('train_getup_program_r113.py','test_getup_program_r113.py','launch_getup_program_r113.py',Path(__file__).name)
    for name in names:
        src=Path(__file__).with_name(name);dst=REPO/'scripts'/name
        if dst.exists():assert digest(dst)==digest(src)
        else:shutil.copy2(src,dst)
    doc='GETUP_STRUCTURED_PROGRAM_R113_20261005.md'
    assert not (REPO/doc).exists();shutil.copy2(Path(__file__).with_name(doc),REPO/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve structured causal teacher-program distillation R113 '+args.snapshot],cwd=REPO,check=True)
    bundle=ROOT/'tmp'/('getup_program_r113_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),closed_updates=closed,
        completed_trials={k:len(v) for k,v in completed.items()},terminal_result_saved=(target/'results.json').exists())),flush=True)


if __name__=='__main__':main()
