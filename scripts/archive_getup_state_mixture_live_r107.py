"""Append one closed R107 generation without pretending the run is complete."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'
RUN=ROOT/'outputs/getup_state_mixture_r107_left_20261005'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--generation',required=True,type=int)
    args=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    progress=json.loads((RUN/'progress.json').read_text());generation=progress['completed_generations']
    assert generation==args.generation and generation>0
    history=json.loads((RUN/'history.json').read_text())[:generation]
    assert len(history)==generation and history[-1]['generation']==generation
    for key in ('best_parameters','mean','std'):
        assert key in history[-1]
    assert history[-1]['best_parameters']==progress['best_parameters']
    destination=REPO/'results'/RUN.name/f'closed_generation_{generation:04d}'
    destination.mkdir(parents=True,exist_ok=False)
    for k in range(1,generation+1):shutil.copy2(RUN/f'checkpoint_{k:04d}.npz',destination/f'checkpoint_{k:04d}.npz')
    (destination/'progress.json').write_text(json.dumps(progress,indent=2))
    (destination/'history.json').write_text(json.dumps(history,indent=2))
    (destination/'archive_scope.json').write_text(json.dumps(dict(closed_generations=generation,
        training_tail_label_s=1,complete_development_not_saved=True,independent_qualification_not_saved=True,
        simulation_only=True,hardware_readiness=False,full_task_completed=False),indent=2))
    (destination/'artifact_hashes.json').write_text(json.dumps({f.name:digest(f) for f in destination.iterdir() if f.is_file()},indent=2))
    name=Path(__file__).name;shutil.copy2(Path(__file__),REPO/'scripts'/name)
    subprocess.run(['git','add','scripts/'+name],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(destination.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m',f'Archive R107 closed generation {generation} with candidate outcomes and RNG'],cwd=REPO,check=True)
    bundle=ROOT/f'tmp/getup_state_mixture_r107_g{generation:04d}_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print('R107_CLOSED_BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),flush=True)


if __name__=='__main__':main()
