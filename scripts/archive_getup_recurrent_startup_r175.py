"""Append unique closed smoke/startup only, without copying open training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import train_getup_recurrent_readout_r175 as run
from diagnostics.getup_independent_native import digest


def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_RECURRENT_READOUT_R175_20261009.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());startup=json.loads((run.OUTPUT/'startup_closed.json').read_text())
    assert smoke['smoke'] and smoke['terminal_result_saved'] and startup['independent_smoke_bitwise_equal'] and not startup['terminal_result_saved']
    assert all(all(row['complete_R157_parity'].values()) for row in smoke['parity']['rows'])
    run.compare_smoke(run.OUTPUT)
    for source,target in [(run.SMOKE,repo/'results'/run.SMOKE.name/'terminal_snapshot'),(run.OUTPUT,repo/'results'/run.OUTPUT.name/'startup_closed_01')]:
        assert not target.exists();target.mkdir(parents=True)
        names=['executed_sources','frozen','zero_parity','nonzero_smoke']
        before={name:hashes(source/name) for name in names}
        for name in names:
            shutil.copytree(source/name,target/name);assert hashes(target/name)==before[name]
        assert all(hashes(source/name)==before[name] for name in names)
        for name in ['contract.json','startup_closed.json']:shutil.copy2(source/name,target/name)
        if source==run.SMOKE:
            shutil.copy2(source/'results.json',target/'results.json');shutil.copy2(source.with_suffix('.log'),target/'process.log')
        for name,dest in [('getup_recurrent_readout_r175_initial_regression_20261009.log','initial_regression.log'),('getup_recurrent_readout_r175_formal_regression_20261009.log','formal_regression.log')]:
            shutil.copy2(run.ROOT/'tmp'/name,target/dest)
        run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=source==run.SMOKE,training_generations_saved=0,formal_startup_closed=source==run.OUTPUT,
            saved_complete_trajectories=6,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        run.local.write_json(target/'artifact_hashes.json',hashes(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['train_getup_recurrent_readout_r175.py','test_getup_recurrent_readout_r175.py','launch_getup_recurrent_readout_r175.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R175 recurrent coordinated sensor readout regressions and closed startup'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),startup_only=True,training_generations_saved=0,terminal_result_saved=False)),flush=True)


if __name__=='__main__':main()
