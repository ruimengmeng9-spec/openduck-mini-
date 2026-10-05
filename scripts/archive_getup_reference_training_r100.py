"""Append closed formal R100 model groups; never modify live experiment files."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from diagnostics.archive_getup_reference_r100 import snapshot,sha

ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
BASE='ebe940179f244508e6bea3ea14b34a8f84db8e19'


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    source=ROOT/'outputs/getup_reference_residual_r100_left_20261005'
    summary=json.loads((source/'training/training_summary.json').read_text())
    assert summary['completed_iterations']==64
    target=REPO/'results/getup_reference_residual_r100_left_20261005/formal_training_snapshot'
    target.mkdir(exist_ok=False)
    snapshot(source/'training',target/'training')
    for step in (16,32,48,64):
        prefix=f'checkpoint_{step:04d}'
        for file in (source/'training').glob(prefix+'.*'):shutil.copy2(file,target/'training'/file.name)
    shutil.copytree(source/'baseline_development',target/'baseline_development')
    shutil.copy2(source/'training.log',target/'training.log')
    scope=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        formal_training_completed=True,development_validation_pending=True,
        snapshot_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (target/'archive_scope.json').write_text(json.dumps(scope,indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):sha(f)
        for f in target.rglob('*') if f.is_file()},indent=2))
    doc='GETUP_FULL_REFERENCE_R99_R100_20261005.md'
    shutil.copy2(PROJECT/doc,REPO/doc)
    script=Path(__file__).name;shutil.copy2(Path(__file__),REPO/'scripts'/script)
    subprocess.run(['git','add',doc,'scripts/'+script],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m','Save R100 completed PPO checkpoints and exact full-fall baseline before validation'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_reference_r100_training_20261005.bundle'
    assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+BASE],cwd=REPO,check=True)
    print('R100_TRAINING_BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':main()
