"""Append only closed independent smoke and formal startup to clean repo."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import train_getup_full_episode_pg_r181 as run

def hashes(path):return {str(p.relative_to(path)):run.prior.digest(p) for p in path.rglob('*') if p.is_file()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_FULL_EPISODE_PG_R181_20261010.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    receipt=json.loads((run.ROOT/'tmp/getup_publication_r179_r180_terminal_20261010_result.json').read_text());assert receipt['independent_remote_verification'] and receipt['published_terminal']==args.base
    assert not (repo/doc).exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());startup=json.loads((run.OUTPUT/'startup_closed.json').read_text())
    assert smoke['smoke'] and smoke['terminal_result_saved'] and startup['independent_smoke_bitwise_equal'] and not startup['terminal_result_saved'];run.compare_smoke(run.OUTPUT)
    for source,target in [(run.SMOKE,repo/'results'/run.SMOKE.name/'terminal_snapshot'),(run.OUTPUT,repo/'results'/run.OUTPUT.name/'startup_closed_01')]:
        assert not target.exists();target.mkdir(parents=True)
        for name in ['executed_sources','frozen','zero_parity','nonzero_smoke','exploration_smoke']:
            before=hashes(source/name);shutil.copytree(source/name,target/name);assert hashes(target/name)==before and hashes(source/name)==before
        for name in ['contract.json','startup_closed.json']:shutil.copy2(source/name,target/name)
        if source==run.SMOKE:shutil.copy2(source/'results.json',target/'results.json');shutil.copy2(source.with_suffix('.log'),target/'process.log')
        for name,dest in [('getup_full_episode_pg_r181_initial_regression_20261010.log','initial_regression.log'),('getup_full_episode_pg_r181_formal_regression_20261010.log','formal_regression.log')]:shutil.copy2(run.ROOT/'tmp'/name,target/dest)
        run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=source==run.SMOKE,training_updates_saved=0,saved_complete_or_failed_dynamic_attempts=8,formal_startup_closed=source==run.OUTPUT,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['train_getup_full_episode_pg_r181.py','test_getup_full_episode_pg_r181.py','launch_getup_full_episode_pg_r181.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R181 whole-episode policy-gradient contract and closed full-path startup'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),startup_only=True,terminal_result_saved=False,training_updates_saved=0)),flush=True)

if __name__=='__main__':main()
