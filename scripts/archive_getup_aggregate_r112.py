"""Append terminal R110, immutable R111 and a bounded closed R112 snapshot."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'
R110=ROOT/'outputs/getup_distill_r110_left_20261005'
R111=ROOT/'outputs/getup_distill_audit_r111_20261005'
R112=ROOT/'outputs/getup_aggregate_r112_left_20261005'
SMOKE=ROOT/'outputs/getup_aggregate_r112_smoke_20261005'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--snapshot',required=True);args=p.parse_args()
    assert args.snapshot.replace('_','').isalnum()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    target=REPO/'results'/R112.name/args.snapshot;assert not target.exists();target.mkdir(parents=True)
    # R110 is terminal and its original evidence is never edited.
    assert (R110/'results.json').exists() and not json.loads((R110/'results.json').read_text())['independent_qualification_run']
    terminal=REPO/'results'/R110.name/'terminal_snapshot'
    if not terminal.exists():
        shutil.copytree(R110,terminal);shutil.copy2(R110.with_suffix('.log'),terminal/'parent.log')
        (terminal/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(terminal)):digest(f) for f in terminal.rglob('*') if f.is_file()},indent=2))
    audited=REPO/'results'/R111.name/'terminal_snapshot'
    if not audited.exists():shutil.copytree(R111,audited)
    shutil.copytree(SMOKE,target/'independent_smoke');shutil.copy2(SMOKE.with_suffix('.log'),target/'smoke_parent.log')
    shutil.copytree(R112/'executed_sources',target/'executed_sources')
    for name in ('contract.json','progress.json','results.json','frozen_candidate.npz','frozen_candidate.json'):
        if (R112/name).exists():shutil.copy2(R112/name,target/name)
    training=R112/'training';dest=target/'training';dest.mkdir();closed=[]
    for report in sorted(training.glob('report_*.json')):
        update=int(report.stem.split('_')[-1]);number=f'{update:05d}'
        assert json.loads(report.read_text())['model_sha256']==digest(training/f'update_{number}.npz')
        for name in (report.name,f'update_{number}.npz',f'learner_{number}.msgpack',f'rng_{number}.json'):
            shutil.copy2(training/name,dest/name)
        closed.append(update)
    for name in ('data_manifest.json','warm_actor.npz','history.json','progress.json','results.json'):
        if (training/name).exists():shutil.copy2(training/name,dest/name)
    completed={}
    for group in sorted(R112.iterdir()):
        if not group.is_dir() or not (group.name.startswith('development_') or group.name.startswith('qualification_')):continue
        saved=[]
        for case in sorted(group.glob('case_*')):
            if (case/'result.json').exists():
                shutil.copytree(case,target/group.name/case.name);saved.append(json.loads((case/'result.json').read_text()))
        completed[group.name]=saved
        if (group/'results.json').exists():shutil.copy2(group/'results.json',target/group.name/'results.json')
    cwd=ROOT/'projects/Open_Duck_Playground'
    test=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_aggregate_r112','-v'],
        cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'snapshot_regression_tests.log').write_text(test.stdout);assert test.returncode==0
    shutil.copy2(R112.with_suffix('.log'),target/'parent_log_snapshot.log')
    (target/'snapshot.json').write_text(json.dumps(dict(closed_updates=closed,completed_trials=completed,
        terminal_result_saved=(target/'results.json').exists(),full_task_completed=False,hardware_readiness=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    for name in ('audit_getup_distill_r111.py','train_getup_aggregate_r112.py','test_getup_aggregate_r112.py','launch_getup_aggregate_r112.py',Path(__file__).name):
        dst=REPO/'scripts'/name
        if dst.exists():assert digest(dst)==digest(Path(__file__).with_name(name))
        else:shutil.copy2(Path(__file__).with_name(name),dst)
    doc='GETUP_VISITED_STATE_AGGREGATION_R111_R112_20261005.md'
    if (REPO/doc).exists():assert digest(REPO/doc)==digest(Path(__file__).with_name(doc))
    else:shutil.copy2(Path(__file__).with_name(doc),REPO/doc)
    subprocess.run(['git','add',doc,'scripts/audit_getup_distill_r111.py','scripts/train_getup_aggregate_r112.py',
        'scripts/test_getup_aggregate_r112.py','scripts/launch_getup_aggregate_r112.py','scripts/'+Path(__file__).name],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(terminal.relative_to(REPO)),str(audited.relative_to(REPO)),str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve complete distillation failure audit and offline aggregation R112 '+args.snapshot],cwd=REPO,check=True)
    bundle=ROOT/'tmp'/('getup_aggregate_r112_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        closed_updates=closed,completed_trials={k:len(v) for k,v in completed.items()},terminal_result_saved=(target/'results.json').exists())),flush=True)


if __name__=='__main__':main()
