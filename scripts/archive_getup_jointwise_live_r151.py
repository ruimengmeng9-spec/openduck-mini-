"""Append immutable closed diagonal-feedback generations, never open trials."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.train_getup_jointwise_dynamic_r151 import ROOT,OUTPUT
from diagnostics.getup_independent_native import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--generation',type=int,required=True)
    args=p.parse_args();assert 1<=args.generation<=6;repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    checkpoint=OUTPUT/'checkpoints'/f'generation_{args.generation:04d}'
    history=json.loads((checkpoint/'history.json').read_text());assert len(history)==args.generation
    before={str(f):digest(f) for f in checkpoint.iterdir() if f.is_file()}
    locations=sorted({path for h in history for path in h['closed_trial_directories']})
    for location in locations:
        folder=Path(location);assert folder.is_relative_to(OUTPUT/'training')
        report=json.loads((folder/'results.json').read_text());assert len(report['rows'])==25
        assert all((folder/f'case_{r["case_seed"]}'/'trajectory.npz').is_file() for r in report['rows'])
    target=repo/'results'/OUTPUT.name/f'closed_generation_{args.generation:04d}';assert not target.exists()
    bundle=ROOT/'tmp'/f'getup_jointwise_live_r151_g{args.generation:04d}_20261006.bundle';assert not bundle.exists()
    script=repo/'scripts'/Path(__file__).name
    if script.exists():assert digest(script)==digest(__file__)
    target.mkdir(parents=True);shutil.copytree(checkpoint,target/'checkpoint')
    for location in locations:
        source=Path(location);dest=target/source.relative_to(OUTPUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,dest)
    shutil.copy2(OUTPUT/'contract.json',target/'contract.json')
    assert before=={str(f):digest(f) for f in checkpoint.iterdir() if f.is_file()}
    (target/'snapshot.json').write_text(json.dumps(dict(closed_generation=args.generation,distinct_closed_programs=len(locations),short_rollouts=25*len(locations),
        selected_parameters=history[-1]['selected_parameters'],training_short_label_not_30s_acceptance=True,
        terminal_result_saved=False,no_expanded_or_independent_run=True,full_task_completed=False,hardware_readiness=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    if not script.exists():shutil.copy2(__file__,script)
    subprocess.run(['git','add','scripts/'+script.name],cwd=repo,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m',f'Preserve R151 closed generation {args.generation} and all causal feedback failures'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bundle_sha256=digest(bundle),bundle_bytes=bundle.stat().st_size,closed_generation=args.generation,distinct_programs=len(locations),rollouts=25*len(locations))),flush=True)

if __name__=='__main__':main()
