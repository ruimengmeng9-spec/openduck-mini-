"""Append unique R172 failure and R172b closed offline learning evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_sensor_dynamics_r172b as task

def hashes(root):
    return {str(p.relative_to(root)):task.sha(p) for p in root.rglob('*') if p.is_file()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=task.ROOT/'github/openduck-mini-';doc='GETUP_SENSOR_DYNAMICS_R172_20261008.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    r=json.loads((task.OUTPUT/'results.json').read_text());s=json.loads((task.SMOKE/'results.json').read_text())
    assert r['terminal_result_saved'] and r['source_hashes_unchanged'] and r['new_dynamic_trajectories']==0 and r['input_trajectories']==100 and r['training_trajectories']==63 and r['training_transitions']==143514
    assert s['smoke'] and s['terminal_result_saved'] and s['input_trajectories']==6
    # Closed solver reproducibility, not a policy or physical safety test.
    with np.load(task.OUTPUT/'checkpoint_closed_01/normal_equations.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.linalg.solve(z['gram']+task.RIDGE*np.eye(342),z['rhs']),z['weights'])
    bundle=task.ROOT/'tmp/getup_sensor_dynamics_r172_20261008.bundle'
    assert not bundle.exists() and not (repo/doc).exists()
    saved=[]
    sources=[(task.ROOT/'outputs/getup_sensor_dynamics_r172_smoke_20261008',False),(task.SMOKE,True),(task.OUTPUT,True)]
    for source,closed in sources:
        before=hashes(source);target=repo/'results'/source.name/('terminal_snapshot' if closed else 'failure_snapshot_01')
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        log=Path(str(source)+'.log');shutil.copy2(log,target/'process.log')
        assert before==hashes(source)
        task.write(target/'snapshot.json',dict(terminal_result_saved=closed,failed_before_model_training=not closed,offline_only=True,new_dynamic_trajectories=0,new_controller=False,full_task_completed=False,hardware_readiness=False,qualification_run=False))
        task.write(target/'artifact_hashes.json',hashes(target));saved.append(dict(source=str(source),files=len(before),terminal=closed))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in ('train_getup_sensor_dynamics_r172.py','launch_getup_sensor_dynamics_r172.py','train_getup_sensor_dynamics_r172b.py','launch_getup_sensor_dynamics_r172b.py',Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(task.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R172 logged-action rounding failure and R172b held-case held-program sensor transition model; no new controller'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),parent=args.base,bundle=str(bundle),bytes=bundle.stat().st_size,sha256=task.sha(bundle),saved=saved)),flush=True)

if __name__=='__main__':main()
