"""Append only closed R170 independent smoke and formal startup, never open trials."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from diagnostics import train_getup_tracking_memory_r170 as run
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_TRACKING_MEMORY_R170_20261008.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip() and not (repo/doc).exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke'] and smoke['terminal_result_saved']
    startup=json.loads((run.OUTPUT/'startup_closed.json').read_text());assert not startup['terminal_result_saved'] and startup['independent_smoke_bitwise_equal']
    assert all(all(r['complete_R157_parity'].values()) for r in smoke['parity']['rows'])
    for name in ('zero_parity','nonzero_smoke'):
        for row in (smoke['parity'] if name=='zero_parity' else smoke['nonzero'])['rows']:
            case=row['case_seed'];fresh=run.OUTPUT/name/f'case_{case}';old=run.SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(old/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            r=json.loads((fresh/'result.json').read_text());assert r['initial_hash']==row['initial_hash'] and r['peaks']==row['peaks']
    bundle=run.ROOT/'tmp/getup_tracking_memory_startup_r170_20261008.bundle';assert not bundle.exists()
    target=repo/'results'/run.SMOKE.name/'terminal_snapshot';assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    before=hashes(run.SMOKE);shutil.copytree(run.SMOKE,target);assert before==hashes(run.SMOKE)
    shutil.copy2(run.SMOKE.with_suffix('.log'),target/'process.log')
    run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,smoke=True,saved_complete_trajectories=6,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
    run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    target=repo/'results'/run.OUTPUT.name/'startup_closed_01';assert not target.exists();target.mkdir(parents=True)
    for name in ['executed_sources','frozen','zero_parity','nonzero_smoke']:
        before=hashes(run.OUTPUT/name);shutil.copytree(run.OUTPUT/name,target/name);assert before==hashes(run.OUTPUT/name)
    for name in ['contract.json','startup_closed.json']:shutil.copy2(run.OUTPUT/name,target/name)
    for name,dest in [('getup_tracking_memory_r170_initial_regression_20261008.log','initial_regression.log'),('getup_tracking_memory_r170_regression_20261008.log','regression.log')]:
        shutil.copy2(run.ROOT/'tmp'/name,target/dest)
    dependencies=target/'imported_diagnostic_dependencies';dependencies.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,dependencies/Path(f).name)
    run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=False,training_generations_saved=0,formal_startup_closed=True,saved_complete_trajectories=6,
        full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
    run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ['train_getup_tracking_memory_r170.py','test_getup_tracking_memory_r170.py','launch_getup_tracking_memory_r170.py',Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R170 causal target tracking memory regressions and closed full startup'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bytes=bundle.stat().st_size,bundle_sha256=digest(bundle))),flush=True)

if __name__=='__main__':main()
