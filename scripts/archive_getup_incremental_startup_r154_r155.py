"""Append new causal algebra audit and closed startup, never open training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_incremental_residual_r155 as run
from diagnostics import audit_getup_error_products_r154 as audit
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_INCREMENTAL_RESIDUAL_R154_R155_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    result=json.loads((audit.OUTPUT/'results.json').read_text());assert result['read_only'] and len(result['rows'])==25
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    startup=json.loads((run.OUTPUT/'startup_closed.json').read_text());assert not startup['terminal_result_saved']
    for group in [smoke['parity'],startup['parity']]:
        for row in group['rows']:assert all(row['frozen_R134_trace_bitwise_equal'].values())
    bundle=run.ROOT/'tmp/getup_incremental_startup_r154_r155_20261006.bundle';assert not bundle.exists()
    for source in [audit.OUTPUT,run.SMOKE]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target);assert before==hashes(source)
        run.prior.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,read_only=source==audit.OUTPUT,
            source=str(source),saved_trajectories=len(list(source.rglob('trajectory.npz'))),full_task_completed=False,hardware_readiness=False))
        run.prior.local.write_json(target/'artifact_hashes.json',hashes(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/run.OUTPUT.name/'startup_closed_01';assert not target.exists();target.mkdir(parents=True)
    for name in ['executed_sources','frozen','zero_parity','nonzero_smoke']:shutil.copytree(run.OUTPUT/name,target/name)
    for name in ['contract.json','startup_closed.json']:shutil.copy2(run.OUTPUT/name,target/name)
    # Independently verified smoke and formal startup must match every saved field.
    for group in ['zero_parity','nonzero_smoke']:
        for folder in (target/group).glob('case_*'):
            with np.load(folder/'trajectory.npz',allow_pickle=False) as x,np.load(run.SMOKE/group/folder.name/'trajectory.npz',allow_pickle=False) as y:
                assert all(np.array_equal(x[k],y[k]) for k in x.files)
            formal=json.loads((folder/'result.json').read_text());independent=json.loads((run.SMOKE/group/folder.name/'result.json').read_text())
            assert formal['initial_hash']==independent['initial_hash'] and formal['peaks']==independent['peaks']
    run.prior.local.write_json(target/'snapshot.json',dict(terminal_result_saved=False,training_generations_saved=0,
        formal_startup_closed=True,saved_complete_trajectories=6,full_task_completed=False,hardware_readiness=False))
    run.prior.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_error_products_r154.py','train_getup_incremental_residual_r155.py','test_getup_incremental_residual_r155.py',
                 'launch_getup_incremental_residual_r155.py','archive_getup_incremental_live_r155.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R154 read-only error-product audit and R155 causal direct-residual startup'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size)),flush=True)

if __name__=='__main__':main()
