"""Append one immutable closed training generation, never open candidates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.train_getup_dynamic_gain_r147 import ROOT,OUTPUT
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--generation',type=int,required=True)
    args=p.parse_args();assert 1<=args.generation<=8;repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    checkpoint=OUTPUT/'checkpoints'/f'generation_{args.generation:04d}'
    history=json.loads((checkpoint/'history.json').read_text());assert len(history)==args.generation
    before={str(p):digest(p) for p in checkpoint.iterdir() if p.is_file()}
    locations=sorted({path for h in history for path in h['closed_trial_directories']})
    for location in locations:
        folder=Path(location);assert folder.is_relative_to(OUTPUT/'training')
        report=json.loads((folder/'results.json').read_text());assert len(report['rows'])==25
        assert all((folder/f'case_{r["case_seed"]}'/'trajectory.npz').exists() for r in report['rows'])
    target=repo/'results'/OUTPUT.name/f'closed_generation_{args.generation:04d}';assert not target.exists()
    target.mkdir(parents=True)
    shutil.copytree(checkpoint,target/'checkpoint')
    for location in locations:
        source=Path(location);dest=target/source.relative_to(OUTPUT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,dest)
    shutil.copy2(OUTPUT/'contract.json',target/'contract.json')
    assert before=={str(p):digest(p) for p in checkpoint.iterdir() if p.is_file()}
    info=dict(closed_generation=args.generation,distinct_closed_programs=len(locations),saved_short_rollouts=25*len(locations),
        best=history[-1]['reports'][max(range(len(history[-1]['reports'])),key=lambda i:(history[-1]['reports'][i]['nominal_success'],history[-1]['reports'][i]['physical_failures']==0,history[-1]['reports'][i]['successes'],-history[-1]['reports'][i]['physical_failures'],history[-1]['reports'][i]['return_sum']))],
        parameters=history[-1]['selected_parameters'],training_short_label_not_30s_acceptance=True,
        terminal_result_saved=False,no_expanded_or_independent_run=True,
        startup_parent_commit='e04b9e627aafb4835e009a2e893c39c56f4e0940',full_task_completed=False,hardware_readiness=False)
    (target/'snapshot.json').write_text(json.dumps(info,indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
    script=repo/'scripts'/Path(__file__).name
    if script.exists():assert digest(script)==digest(Path(__file__))
    else:shutil.copy2(__file__,script)
    subprocess.run(['git','add','scripts/'+script.name],cwd=repo,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m',f'Preserve R147 closed generation {args.generation} and all short failures'],cwd=repo,check=True)
    bundle=ROOT/'tmp'/f'getup_dynamic_live_r147_g{args.generation:04d}_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),closed_generation=args.generation,distinct_programs=len(locations),rollouts=25*len(locations))),flush=True)


if __name__=='__main__':main()
