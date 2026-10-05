"""Publish an explicit bounded snapshot, never modify the source experiment."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/data/shijinsheng/open_duck')
SOURCE=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
RUN=ROOT/'outputs/getup_mixed_lr_r98_20261005'
BASE='477c972563b803617efe76f5378c29af5035dbd3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_json(path,target):
    # Running trainer writes its JSON nonatomically; never publish a torn read.
    for _ in range(20):
        try:
            data=Path(path).read_bytes()
            json.loads(data)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            return
        except json.JSONDecodeError:
            time.sleep(.1)
    raise RuntimeError(f'Cannot obtain valid JSON snapshot: {path}')


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    destination=REPO/'results'/RUN.name
    destination.mkdir(parents=True,exist_ok=False)
    new_files=('train_getup_mixed_ppo_r98.py','getup_mixed_curriculum_env_r98.py',
               'test_getup_mixed_r98.py','validate_getup_mixed_r98.py','run_getup_mixed_r98.py')
    for name in new_files:
        assert not (REPO/'diagnostics'/name).exists()
        shutil.copy2(SOURCE/'diagnostics'/name,REPO/'diagnostics'/name)
    # Source evidence for Oct 2 defects; not installed over the training code.
    evidence=destination/'oct02_source_evidence'
    evidence.mkdir()
    for name in ('train_getup_prefix_ppo.py','getup_mixed_curriculum_env.py','getup_prefix_library_env.py'):
        shutil.copy2(SOURCE/'diagnostics'/name,evidence/name)
    for old_name in ('getup_mixed_p05_smoke_20261002b','getup_mixed_p05_extended_20261002',
                     'getup_prefix_stage3_smoke_20261002'):
        old=ROOT/'training'/old_name
        target=destination/'historical_training'/old_name
        target.mkdir(parents=True)
        for name in ('controller_contract.json','training_summary.json','final.msgpack','final.onnx'):
            shutil.copy2(old/name,target/name)
        if (old/'episodes.json').exists():
            with gzip.open(target/'episodes.json.gz','wb') as stream:
                stream.write((old/'episodes.json').read_bytes())
    for old_name in ('getup_mixed_p05_eval_20261002','getup_prefix_stage3_eval_20261002',
                     'getup_prefix_state_audit_r73_20261002'):
        shutil.copytree(ROOT/'outputs'/old_name,destination/'historical_outputs'/old_name)
    prior=ROOT/'outputs/getup_vertical_imu_r97_left_20261001'
    target=destination/'r97_final'
    target.mkdir()
    for name in ('results.json','contract.json','checkpoint.npz','search_history.json'):
        shutil.copy2(prior/name,target/name)
    shutil.copytree(RUN/'executed_sources',destination/'runner_started_sources')
    for name in ('historical_audit.json','experiment.json'):
        copy_json(RUN/name,destination/name)
    shutil.copy2(RUN/'tests.log',destination/'tests_at_start.log')
    final_tests=SOURCE/'diagnostics/r98_full_regression_20261005.txt'
    shutil.copy2(final_tests,destination/'full_regression_23_tests.log')
    for label in ('smoke','a_legacy_lr','b_corrected_lr'):
        directory=RUN/label
        if not directory.exists():continue
        target=destination/label
        target.mkdir()
        for name in ('controller_contract.json','training_history.json','training_summary.json'):
            if (directory/name).exists():copy_json(directory/name,target/name)
        if (directory/'executed_sources').exists():
            shutil.copytree(directory/'executed_sources',target/'executed_sources')
        for name in ('initial.msgpack','initial.onnx'):
            if (directory/name).exists():shutil.copy2(directory/name,target/name)
        checkpoints=sorted(directory.glob('checkpoint_*.msgpack'))
        plain=[f for f in checkpoints if not f.name.endswith('.learner.msgpack')]
        if plain:
            selected=plain[-1].stem
            for file in directory.glob(selected+'.*'):
                shutil.copy2(file,target/file.name)
    cpu=ROOT/'outputs/getup_mixed_cpu_smoke_r98_20261005'
    shutil.copytree(cpu,destination/'cpu_smoke_exit_check')
    scope=dict(server_run=str(RUN),server_originals_preserved=True,snapshot_only=True,
        full_task_completed=False,hardware_readiness=False,
        source_repository_dirty_edits_untouched=True,
        runner_used_gpu_before_later_cpu_option=True,
        halted_process_count=0,qualification_pending=True)
    (destination/'archive_scope.json').write_text(json.dumps(scope,indent=2))
    manifest={str(f.relative_to(destination)):sha(f) for f in destination.rglob('*') if f.is_file()}
    (destination/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2))
    document='GETUP_MIXED_PPO_AUDIT_R98_20261005.md'
    shutil.copy2(SOURCE/document,REPO/document)
    script='archive_getup_mixed_r98.py'
    shutil.copy2(SOURCE/'diagnostics'/script,REPO/'scripts'/script)
    subprocess.run(['git','add',document,'scripts/'+script,
        *['diagnostics/'+name for name in new_files]],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(destination.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m',
        'Audit mixed getup false positives and launch corrected R98 learning rate comparison'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_mixed_r98_started_20261005.bundle'
    assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+BASE],cwd=REPO,check=True)
    print('R98_BUNDLE',bundle,flush=True)
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True),flush=True)


if __name__=='__main__':
    main()
