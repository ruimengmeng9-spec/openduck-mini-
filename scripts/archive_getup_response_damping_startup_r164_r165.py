"""Append immutable R164/R164b audits and closed R165 full startup, not training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_velocity_damping_r165 as run
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_RESPONSE_STAGES_DAMPING_R164_R165_20261006.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip() and not (repo/doc).exists()
    for name in ['getup_response_stages_r164_20261006','getup_response_stages_r164b_20261006']:
        result=json.loads((run.ROOT/'outputs'/name/'results.json').read_text());assert result['read_only'] and result['pairs']==58
    filter_report=json.loads((run.ROOT/'outputs/getup_response_filter_parity_r164b_20261006/results.json').read_text());assert len(filter_report['checks'])==62
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    startup=json.loads((run.OUTPUT/'startup_closed.json').read_text());assert not startup['terminal_result_saved']
    assert all(all(r['complete_R157_parity'].values()) for r in smoke['parity']['rows'])
    assert all(all(r['complete_R157_parity'].values()) for r in startup['parity']['rows'])
    bundle=run.ROOT/'tmp/getup_response_damping_startup_r164_r165_20261006.bundle';assert not bundle.exists()
    write=run.local.write_json;saved=[]
    names=['getup_response_stages_r164_20261006','getup_response_stages_r164_smoke_20261006',
        'getup_response_stages_r164b_20261006','getup_response_stages_r164b_smoke_20261006','getup_response_filter_parity_r164b_20261006',run.SMOKE.name]
    for name in names:
        source=run.ROOT/'outputs'/name;before=hashes(source);target=repo/'results'/name/'terminal_snapshot';assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        assert before==hashes(source)
        if source==run.SMOKE:shutil.copy2(run.ROOT/'tmp/getup_velocity_damping_r165_independent_regression_20261006.log',target/'regression.log')
        elif name=='getup_response_stages_r164b_20261006':shutil.copy2(run.ROOT/'tmp/getup_response_stages_r164_regression_20261006.log',target/'algebra_regression.log')
        trajectories=len(list(source.rglob('trajectory.npz')));saved.append(dict(source=str(source),trajectories=trajectories))
        write(target/'snapshot.json',dict(source=str(source),terminal_result_saved=True,read_only=source!=run.SMOKE,saved_trajectories=trajectories,
            R164_unused_truth_field_loader_caveat_preserved=name.startswith('getup_response_stages_r164_'),full_task_completed=False,hardware_readiness=False))
        write(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/run.OUTPUT.name/'startup_closed_01';assert not target.exists();target.mkdir(parents=True)
    for name in ['executed_sources','frozen','zero_parity','nonzero_smoke']:
        before=hashes(run.OUTPUT/name);shutil.copytree(run.OUTPUT/name,target/name);assert before==hashes(run.OUTPUT/name)
    for name in ['contract.json','startup_closed.json']:shutil.copy2(run.OUTPUT/name,target/name)
    shutil.copy2(run.ROOT/'tmp/getup_velocity_damping_r165_regression_20261006.log',target/'regression.log')
    for group in ['zero_parity','nonzero_smoke']:
        for folder in (target/group).glob('case_*'):
            with np.load(folder/'trajectory.npz',allow_pickle=False) as x,np.load(run.SMOKE/group/folder.name/'trajectory.npz',allow_pickle=False) as y:
                assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            formal=json.loads((folder/'result.json').read_text());ind=json.loads((run.SMOKE/group/folder.name/'result.json').read_text())
            assert formal['initial_hash']==ind['initial_hash'] and formal['peaks']==ind['peaks']
    write(target/'snapshot.json',dict(terminal_result_saved=False,training_generations_saved=0,formal_startup_closed=True,saved_complete_trajectories=6,
        full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
    write(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['audit_getup_response_stages_r164.py','test_getup_response_stages_r164.py','audit_getup_response_stages_r164b.py',
        'verify_getup_response_field_filter_r164b.py','train_getup_velocity_damping_r165.py','test_getup_velocity_damping_r165.py','launch_getup_velocity_damping_r165.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R164 bounded stage audits and R165 sign-constrained full-path damping startup'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+a.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
