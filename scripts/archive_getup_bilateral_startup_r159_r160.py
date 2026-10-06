"""Append immutable R159 terminal and R160 independently closed startup evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_bilateral_modes_r160 as run
from diagnostics import audit_getup_bilateral_coupling_r159 as audit
from diagnostics.getup_independent_native import digest

def hashes(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_BILATERAL_MODES_R159_R160_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    ar=json.loads((audit.OUTPUT/'results.json').read_text());assert ar['read_only'] and len(ar['rows'])==6
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    startup=json.loads((run.OUTPUT/'startup_closed.json').read_text());assert not startup['terminal_result_saved']
    for group in [smoke['parity'],startup['parity']]:
        assert len(group['rows'])==3
        assert all(all(r['frozen_R157_trace_bitwise_equal'].values()) for r in group['rows'])
    bundle=run.ROOT/'tmp/getup_bilateral_startup_r159_r160_20261006.bundle';assert not bundle.exists()
    write=run.prior.old.local.write_json
    for source in [audit.OUTPUT,run.SMOKE]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target);assert before==hashes(source)
        write(target/'snapshot.json',dict(terminal_result_saved=True,read_only=source==audit.OUTPUT,
            source=str(source),saved_trajectories=len(list(source.rglob('trajectory.npz'))),full_task_completed=False,hardware_readiness=False))
        write(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/run.OUTPUT.name/'startup_closed_01';assert not target.exists();target.mkdir(parents=True)
    for name in ['executed_sources','frozen','zero_parity','nonzero_smoke']:
        before=hashes(run.OUTPUT/name);shutil.copytree(run.OUTPUT/name,target/name);assert before==hashes(run.OUTPUT/name)
    for name in ['contract.json','startup_closed.json']:shutil.copy2(run.OUTPUT/name,target/name)
    shutil.copy2(run.ROOT/'tmp/getup_bilateral_modes_r160_regression_20261006.log',target/'regression.log')
    for group in ['zero_parity','nonzero_smoke']:
        for folder in (target/group).glob('case_*'):
            with np.load(folder/'trajectory.npz',allow_pickle=False) as x,np.load(run.SMOKE/group/folder.name/'trajectory.npz',allow_pickle=False) as y:
                assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            formal=json.loads((folder/'result.json').read_text());ind=json.loads((run.SMOKE/group/folder.name/'result.json').read_text())
            assert formal['initial_hash']==ind['initial_hash'] and formal['peaks']==ind['peaks']
    write(target/'snapshot.json',dict(terminal_result_saved=False,training_generations_saved=0,formal_startup_closed=True,
        saved_complete_trajectories=6,full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
    write(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_bilateral_coupling_r159.py','train_getup_bilateral_modes_r160.py','test_getup_bilateral_modes_r160.py',
                 'launch_getup_bilateral_modes_r160.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R159 bilateral coupling audit and R160 causal bilateral feedback startup'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+a.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),
        bundle_sha256=digest(bundle),bytes=bundle.stat().st_size)),flush=True)

if __name__=='__main__':main()
