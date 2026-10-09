"""Append closed R181/R182 evidence only, never modify old sources/archives."""
import argparse
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_full_episode_pg_r181 as run
from diagnostics import audit_getup_full_episode_pg_terminal_r182 as audit


def hashes(path):
    return {str(p.relative_to(path)):run.prior.digest(p) for p in path.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_FULL_EPISODE_PG_TERMINAL_R181_R182_20261010.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    terminal=audit.read(run.OUTPUT/'results.json');closed=audit.read(run.OUTPUT/'training_closed.json')
    assert terminal['terminal_result_saved'] and terminal['formal_dynamic_attempts']==358 and closed['updates']==4
    results=audit.read(audit.OUTPUT/'results.json');smoke=audit.read(audit.SMOKE/'results.json')
    assert results['terminal_result_saved'] and results['scalar_attempts']==358 and len(results['learner_updates'])==4 and results['source_hashes_unchanged']
    assert results['programs'][:4]==smoke['programs'] and results['terminal_pairing']==smoke['terminal_pairing']
    for case in (None,769002,773004):
        with np.load(audit.OUTPUT/'probe_signals'/f'case_{case}'/'signals.npz',allow_pickle=False) as a,np.load(audit.SMOKE/'probe_signals'/f'case_{case}'/'signals.npz',allow_pickle=False) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    run.compare_smoke(run.OUTPUT)
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==358
    assert not (repo/doc).exists()
    for source in (run.OUTPUT,audit.OUTPUT,audit.SMOKE):
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        before=hashes(source);shutil.copytree(source,target);assert before==hashes(source)==hashes(target)
        if source.with_suffix('.log').exists():shutil.copy2(source.with_suffix('.log'),target/'process.log')
        if source==run.OUTPUT:
            for name in ('getup_full_episode_pg_r181_initial_regression_20261010.log','getup_full_episode_pg_r181_formal_regression_20261010.log'):
                shutil.copy2(run.ROOT/'tmp'/name,target/name)
            run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,training_updates_saved=4,saved_formal_dynamic_attempts=358,training_attempts=200,independent_smoke_attempts=8,independent_smoke_already_immutable_in_startup_commit=args.base,full_task_completed=False,hardware_readiness=False,qualification_executed=False))
        else:
            shutil.copy2(run.ROOT/'outputs/getup_full_episode_pg_terminal_audit_r182_regression_20261010.log',target/'regression.log')
            run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,read_only=True,scalar_attempts=8 if source==audit.SMOKE else 358,terminal_pairs=25,learner_updates=0 if source==audit.SMOKE else 4,new_dynamic_replays=0,full_task_completed=False,hardware_readiness=False,qualification_executed=False))
        run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ('audit_getup_full_episode_pg_terminal_r182.py','launch_getup_full_episode_pg_audit_r182.py','inspect_getup_full_episode_pg_terminal_r182.py',Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R181 full-episode terminal failures and R182 complete scalar-gradient-Adam audit'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),flush=True)


if __name__=='__main__':main()
