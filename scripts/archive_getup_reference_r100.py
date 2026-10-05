"""Append R98 final evidence, R99 audit and explicitly bounded R100 snapshot."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
BASE='d524250e48e400324455aa29b79f8a7af32184a8'


def sha(file):return hashlib.sha256(file.read_bytes()).hexdigest()


def json_snapshot(source,target):
    for _ in range(20):
        try:
            data=source.read_bytes();json.loads(data)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);return
        except json.JSONDecodeError:time.sleep(.1)
    raise RuntimeError('Torn JSON snapshot: '+str(source))


def snapshot(source,target):
    target.mkdir(parents=True,exist_ok=False)
    for name in ('experiment.json','controller_contract.json','training_history.json',
                 'training_summary.json','results.json','comparison_partial.json'):
        if (source/name).exists():json_snapshot(source/name,target/name)
    if (source/'executed_sources').exists():shutil.copytree(source/'executed_sources',target/'executed_sources')
    for label in ('initial','final'):
        # Only complete model groups; a model currently being exported is excluded.
        if not (source/f'{label}.onnx').exists():continue
        for file in source.glob(label+'.*'):
            if file.suffix=='.json':json_snapshot(file,target/file.name)
            else:shutil.copy2(file,target/file.name)
    closed=[f for f in source.glob('checkpoint_*.onnx') if f.with_suffix('.rng.json').exists()]
    if closed:
        prefix=sorted(closed)[-1].stem
        for file in source.glob(prefix+'.*'):shutil.copy2(file,target/file.name)
    if (source/'episodes.json').exists():
        data=json.loads((source/'episodes.json').read_text())
        with gzip.open(target/'episodes.json.gz','wt') as stream:json.dump(data,stream)


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    final=REPO/'results/getup_mixed_lr_r98_20261005/final_outcome'
    final.mkdir(exist_ok=False)
    r98=ROOT/'outputs/getup_mixed_lr_r98_20261005'
    for name in ('results.json','experiment.json','comparison_partial.json'):json_snapshot(r98/name,final/name)
    for label in ('a_legacy_lr','b_corrected_lr'):
        snapshot(r98/label,final/label)
        shutil.copytree(r98/f'{label}_development',final/f'{label}_development')
    r99=ROOT/'outputs/getup_path_audit_r99_20261005'
    shutil.copytree(r99,REPO/'results'/r99.name)
    r100=ROOT/'outputs/getup_reference_residual_r100_left_20261005'
    destination=REPO/'results'/r100.name
    snapshot(r100,destination)
    for name in ('tests.log','smoke.log','training.log'):
        if (r100/name).exists():shutil.copy2(r100/name,destination/name)
    for child in ('smoke','training','baseline_development'):
        source=r100/child
        if not source.exists():continue
        if child=='baseline_development':
            (destination/child).mkdir()
            if (source/'results.json').exists():
                shutil.copytree(source,destination/child,dirs_exist_ok=True)
        else:snapshot(source,destination/child)
    for name in ('audit_getup_path_r99.py','getup_reference_env_r100.py',
                 'train_getup_reference_r100.py','run_getup_reference_r100.py','test_getup_reference_r100.py'):
        assert not (REPO/'diagnostics'/name).exists()
        shutil.copy2(PROJECT/'diagnostics'/name,REPO/'diagnostics'/name)
    script='launch_getup_reference_r100.sh';shutil.copy2(PROJECT/'scripts'/script,REPO/'scripts'/script)
    script='archive_getup_reference_r100.py';shutil.copy2(PROJECT/'diagnostics'/script,REPO/'scripts'/script)
    document='GETUP_FULL_REFERENCE_R99_R100_20261005.md'
    shutil.copy2(PROJECT/document,REPO/document)
    for directory in (final,REPO/'results'/r99.name,destination):
        scope=dict(snapshot_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   source_originals_preserved=True,simulation_only=True,hardware_readiness=False,
                   full_task_completed=False,r100_snapshot_not_final=directory==destination)
        (directory/'archive_scope.json').write_text(json.dumps(scope,indent=2))
        manifest={str(f.relative_to(directory)):sha(f) for f in directory.rglob('*') if f.is_file()}
        (directory/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run(['git','add',document,'diagnostics/audit_getup_path_r99.py',
        'diagnostics/getup_reference_env_r100.py','diagnostics/train_getup_reference_r100.py',
        'diagnostics/run_getup_reference_r100.py','diagnostics/test_getup_reference_r100.py',
        'scripts/launch_getup_reference_r100.sh','scripts/archive_getup_reference_r100.py'],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(final.relative_to(REPO)),
                    'results/'+r99.name,'results/'+r100.name],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m',
                    'Save R98 full-fall failures and start audited full-reference residual PPO R100'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_reference_r100_started_20261005.bundle'
    assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+BASE],cwd=REPO,check=True)
    print('R100_BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),flush=True)


if __name__=='__main__':main()
