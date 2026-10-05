"""Append completed negative evidence and bounded R102 startup snapshot."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
BASE='24139362314444fc3968fbf4b4b47f3fce942afd'


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    r100=ROOT/'outputs/getup_reference_residual_r100_left_20261005'
    final=json.loads((r100/'results.json').read_text())
    target=REPO/'results/getup_reference_residual_r100_left_20261005/development_final'
    target.mkdir(exist_ok=False);shutil.copy2(r100/'results.json',target/'results.json')
    for n in (16,32,48,64):
        d=r100/f'checkpoint_{n:04d}_development'
        assert (d/'results.json').exists();shutil.copytree(d,target/d.name)
    for name in ('getup_residual_audit_r101_20261005','getup_nominal_anchor_r101_left_20261005',
                 'getup_nominal_anchor_r101b_left_20261005','getup_joint_anchor_r102_smoke_20261005'):
        source=ROOT/'outputs'/name
        if 'anchor_r101_left' not in name:assert (source/'results.json').exists()
        destination=REPO/'results'/name;assert not destination.exists()
        shutil.copytree(source,destination)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,destination/'parent.log')
        if 'anchor_r101_left' in name:
            (destination/'archive_scope.json').write_text(json.dumps(dict(
                outcome='failed nominal exact-zero numerical assertion; original evidence preserved',
                full_task_completed=False),indent=2))
            numeric=subprocess.check_output([str(PROJECT/'.venv/bin/python'),'-m',
                'diagnostics.inspect_anchor_numerics_r101'],cwd=PROJECT,text=True)
            (destination/'numerical_diagnostic.json').write_text(json.dumps(json.loads(numeric),indent=2))
    source=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    destination=REPO/'results'/source.name/'startup_snapshot';destination.mkdir(parents=True,exist_ok=False)
    for name in ('contract.json','anchor.npz'):
        shutil.copy2(source/name,destination/name)
    shutil.copytree(source/'executed_sources',destination/'executed_sources')
    for file in source.glob('checkpoint_*.npz'):shutil.copy2(file,destination/file.name)
    for name in ('progress.json','history.json','results.json'):
        if (source/name).exists():
            data=json.loads((source/name).read_text())
            (destination/name).write_text(json.dumps(data,indent=2))
    shutil.copy2(source.with_suffix('.log'),destination/'parent.log')
    (destination/'archive_scope.json').write_text(json.dumps(dict(
        simulation_only=True,hardware_readiness=False,full_task_completed=False,
        snapshot_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        snapshot_only=True,formal_training_terminal_not_claimed=True),indent=2))
    sources=['audit_getup_residual_r101.py','probe_getup_anchor_r101.py','probe_getup_anchor_r101b.py',
        'inspect_anchor_numerics_r101.py','train_getup_joint_anchor_r102.py','test_getup_joint_anchor_r102.py',
        Path(__file__).name]
    for name in sources:shutil.copy2(PROJECT/'diagnostics'/name,REPO/'scripts'/name)
    for name in ('launch_getup_anchor_r101.sh','launch_getup_anchor_r101b.sh','launch_getup_joint_anchor_r102.sh'):
        shutil.copy2(PROJECT/'scripts'/name,REPO/'scripts'/name)
    doc='GETUP_JOINT_ANCHOR_R101_R102_20261005.md';shutil.copy2(PROJECT/doc,REPO/doc)
    paths=[target,*[REPO/'results'/name for name in ('getup_residual_audit_r101_20261005',
        'getup_nominal_anchor_r101_left_20261005','getup_nominal_anchor_r101b_left_20261005',
        'getup_joint_anchor_r102_smoke_20261005','getup_joint_anchor_r102_left_20261005')]]
    manifest={str(f.relative_to(REPO)):digest(f) for path in paths for f in path.rglob('*') if f.is_file()}
    (destination/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run(['git','add',doc,*['scripts/'+name for name in sources],
        'scripts/launch_getup_anchor_r101.sh','scripts/launch_getup_anchor_r101b.sh',
        'scripts/launch_getup_joint_anchor_r102.sh'],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',*[str(p.relative_to(REPO)) for p in paths]],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m',
        'Preserve R100 failed validation and R101 anchoring evidence; start R102 joint feedback training'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_joint_anchor_r102_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+BASE],cwd=REPO,check=True)
    print('R102_BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':main()
