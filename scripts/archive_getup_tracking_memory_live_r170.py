"""Append only specified closed R170 generations; never copy open candidates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.train_getup_tracking_memory_r170 import ROOT,OUTPUT,local
from diagnostics.getup_independent_native import digest

def hashes(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--generation',type=int,required=True);a=p.parse_args()
    assert 1<=a.generation<=3;repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==a.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    cp=OUTPUT/'checkpoints'/f'generation_{a.generation:04d}';history=json.loads((cp/'history.json').read_text());assert len(history)==a.generation
    locations=sorted({s for h in history for s in h['closed_trial_directories']});before={str(cp):hashes(cp)}
    for location in locations:
        folder=Path(location);assert folder.is_relative_to(OUTPUT/'training');report=json.loads((folder/'results.json').read_text());assert len(report['rows'])==25
        assert all(r['full_path'] and (folder/f'case_{r["case_seed"]}'/'trajectory.npz').is_file() for r in report['rows'])
        assert all(r['controls']==2279 or not r['valid'] for r in report['rows'])
        before[location]=hashes(folder)
    target=repo/'results'/OUTPUT.name/f'closed_generation_{a.generation:04d}';assert not target.exists()
    bundle=ROOT/'tmp'/f'getup_tracking_memory_live_r170_g{a.generation:04d}_20261008.bundle';assert not bundle.exists()
    script=repo/'scripts'/Path(__file__).name
    if script.exists():assert digest(script)==digest(__file__)
    target.mkdir(parents=True);shutil.copytree(cp,target/'checkpoint')
    for location in locations:
        src=Path(location);dest=target/src.relative_to(OUTPUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(src,dest)
    shutil.copy2(OUTPUT/'contract.json',target/'contract.json')
    assert all(h==hashes(Path(s)) for s,h in before.items()),'Closed source changed during archive'
    local.write_json(target/'snapshot.json',dict(closed_generation=a.generation,distinct_closed_programs=len(locations),actual_full_path_training_attempts=25*len(locations),
        selected_parameters=history[-1]['selected_parameters'],all_training_uses_original_30s_acceptance=True,terminal_result_saved=False,
        same_development_cases_not_independent_qualification=True,no_expanded_or_independent_run=True,full_task_completed=False,hardware_readiness=False))
    local.write_json(target/'artifact_hashes.json',hashes(target))
    if not script.exists():shutil.copy2(__file__,script)
    subprocess.run(['git','add','scripts/'+script.name],cwd=repo,check=True);subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m',f'Preserve R170 closed generation {a.generation}, complete-path training attempts and all failures'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+a.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bundle_sha256=digest(bundle),
        bytes=bundle.stat().st_size,closed_generation=a.generation,distinct_programs=len(locations),attempts=25*len(locations))),flush=True)

if __name__=='__main__':main()
