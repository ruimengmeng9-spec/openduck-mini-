"""Append closed independent smoke/startup and preserved unit failure, not live training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from diagnostics import train_getup_expert_mixture_r168 as run
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_EXPERT_MIXTURE_R168_20261008.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip() and not (repo/doc).exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    startup=json.loads((run.OUTPUT/'startup_closed.json').read_text());assert not startup['terminal_result_saved'] and startup['independent_smoke_bitwise_equal']
    assert all(all(r['complete_R157_parity'].values()) for r in smoke['parity']['rows'])
    bundle=run.ROOT/'tmp/getup_expert_mixture_startup_r168_20261008.bundle';assert not bundle.exists()
    target=repo/'results'/run.SMOKE.name/'terminal_snapshot';assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    before=hashes(run.SMOKE);shutil.copytree(run.SMOKE,target);assert before==hashes(run.SMOKE)
    shutil.copy2(run.SMOKE.with_suffix('.log'),target/'process.log')
    run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,smoke=True,saved_complete_trajectories=6,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
    run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/run.OUTPUT.name/'startup_closed_01';assert not target.exists();target.mkdir(parents=True)
    for name in ['executed_sources','frozen','zero_parity','nonzero_smoke']:
        before=hashes(run.OUTPUT/name);shutil.copytree(run.OUTPUT/name,target/name);assert before==hashes(run.OUTPUT/name)
    for name in ['contract.json','startup_closed.json']:shutil.copy2(run.OUTPUT/name,target/name)
    shutil.copy2(run.ROOT/'tmp/getup_expert_mixture_r168_regression_20261008.log',target/'regression.log')
    run.compare_smoke(run.OUTPUT)
    # Preserve other actually imported diagnostic dependency sources as evidence,
    # without changing sources used by the running parent or workers.
    dependencies=target/'imported_diagnostic_dependencies';dependencies.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,dependencies/Path(f).name)
    run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=False,training_generations_saved=0,formal_startup_closed=True,saved_complete_trajectories=6,
        full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
    run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    source=run.ROOT/'tmp/getup_expert_mixture_r168_regression_failure_01';target=repo/'results/getup_expert_mixture_r168_regression_failure_01/preserved_failure'
    assert source.exists() and not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    before=hashes(source);shutil.copytree(source,target);assert before==hashes(source)
    run.local.write_json(target/'snapshot.json',dict(unit_test_rounding_failure=True,no_dynamic_replay=True,physics_acceptance_unchanged=True))
    run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['train_getup_expert_mixture_r168.py','test_getup_expert_mixture_r168.py','launch_getup_expert_mixture_r168.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R168 causal frozen-expert output mixture regressions and closed full startup'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bytes=bundle.stat().st_size,bundle_sha256=digest(bundle))),flush=True)

if __name__=='__main__':main()
