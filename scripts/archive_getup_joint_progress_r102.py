"""Append a coherent closed-generation snapshot without editing live training."""
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'
BASE='17419b4be26537efb2b216d6192ac930d236b2be'


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    source=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    progress=json.loads((source/'progress.json').read_text());n=progress['completed_generations']
    history=json.loads((source/'history.json').read_text())[:n]
    assert len(history)==n and history[-1]['generation']==n
    target=REPO/'results'/source.name/f'closed_generation_{n:04d}';target.mkdir(exist_ok=False)
    for i in range(1,n+1):
        name=f'checkpoint_{i:04d}.npz';shutil.copy2(source/name,target/name)
    for name,data in [('progress.json',progress),('history.json',history)]:
        (target/name).write_text(json.dumps(data,indent=2))
    shutil.copytree(source/'zero_parity',target/'zero_parity')
    scope=dict(simulation_only=True,full_task_completed=False,hardware_readiness=False,
        closed_generation=n,short_tail_training_labels_only=True,development_qualification_pending=True)
    (target/'archive_scope.json').write_text(json.dumps(scope,indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f)
        for f in target.rglob('*') if f.is_file()},indent=2))
    name=Path(__file__).name;shutil.copy2(Path(__file__),REPO/'scripts'/name)
    subprocess.run(['git','add','scripts/'+name],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m',f'Save R102 closed generation {n} feedback gains and candidate reports'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_joint_progress_r102_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+BASE],cwd=REPO,check=True)
    print('R102_PROGRESS_BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':main()
