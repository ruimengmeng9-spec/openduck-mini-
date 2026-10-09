"""Append closed R173 and R174 evidence only; never overwrite executed archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data/shijinsheng/open_duck')
REPO = ROOT / 'github/openduck-mini-'
FORMAL = ROOT / 'outputs/getup_intervention_dynamics_r173_20261009'
AUDIT = ROOT / 'outputs/getup_intervention_dynamics_audit_r174_20261009'
AUDIT_SMOKE = ROOT / 'outputs/getup_intervention_dynamics_audit_r174_smoke_20261009'
DOC = 'GETUP_INTERVENTION_TERMINAL_R173_R174_20261009.md'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def hashes(folder):
    assert not any(p.is_symlink() for p in folder.rglob('*'))
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=REPO, text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    args = p.parse_args()
    assert git('rev-parse', 'HEAD') == args.base
    assert not git('status', '--porcelain')
    running = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(x in line for x in ('train_getup_', 'audit_getup_', 'probe_getup_')) for line in running)
    assert shutil.disk_usage(ROOT).free > 4 * 1024**3
    bundle = ROOT / 'tmp/getup_intervention_terminal_r173_r174_20261009.bundle'
    targets = [REPO / 'results' / source.name / 'terminal_snapshot' for source in (FORMAL, AUDIT, AUDIT_SMOKE)]
    assert not bundle.exists() and not (REPO / DOC).exists()
    assert all(not target.exists() for target in targets)
    for name in (Path(__file__).name, 'audit_getup_intervention_dynamics_r174.py'):
        assert not (REPO / 'scripts' / name).exists()

    result = read(FORMAL / 'results.json')
    closed = read(FORMAL / 'training_closed.json')
    data = read(FORMAL / 'training_data_closed.json')
    audit = read(AUDIT / 'results.json')
    smoke = read(AUDIT_SMOKE / 'results.json')
    assert result['terminal_result_saved'] and result['source_hashes_unchanged']
    assert not result['new_controller'] and result['current_best_unified_successes'] == 18
    assert closed['data_attempts'] == 175 and closed['startup_attempts'] == 6 and closed['all_failures_saved']
    assert data['programs'] == 7 and data['attempts'] == 175 and data['common_state_pairs'] == 75
    assert len(list((FORMAL / 'episodes').glob('program_*/case_*/trajectory.npz'))) == 175
    assert len(list((FORMAL / 'zero_parity').glob('case_*/trajectory.npz'))) == 3
    assert len(list((FORMAL / 'nonzero_smoke').glob('case_*/trajectory.npz'))) == 3
    assert [(r['successes'], r['physical_failures']) for r in result['intervention_reports']] == [(18, 0), (6, 0), (8, 3), (8, 1), (9, 0), (5, 4), (8, 2)]
    for program in range(7):
        checkpoint = read(FORMAL / 'checkpoints' / f'program_{program:02d}' / 'closed.json')
        assert checkpoint['program'] == program and checkpoint['closed_episodes'] == 25
        assert checkpoint['report'] == result['intervention_reports'][program]
    learner = result['learner']
    assert learner == closed['learner'] and learner['training_transitions'] == 230078
    assert learner['training_trajectories'] == 101 and learner['evaluation_trajectories'] == 165
    assert sha(FORMAL / 'learner_closed_01/model.npz') == learner['model_sha256']
    assert learner['invalid_excluded'] == 10
    for path, digest in read(FORMAL / 'contract.json')['hashes'].items():
        assert sha(Path(path)) == digest
    for source, report, pairs in ((AUDIT, audit, 75), (AUDIT_SMOKE, smoke, 9)):
        assert report['read_only'] and report['terminal_result_saved'] and report['source_hashes_unchanged']
        assert report['pairs'] == pairs and report['new_dynamic_trajectories'] == 0 and not report['new_controller']
        assert len(read(source / 'pairs.json')) == pairs
        for path, digest in read(source / 'source_hashes.json').items():
            assert sha(Path(path)) == digest
    assert audit['smoke_rows_exact'] and audit['pair_summary']['all']['nonzero_applied_pairs'] == 72

    all_sources = []
    for source, target in zip((FORMAL, AUDIT, AUDIT_SMOKE), targets):
        before = hashes(source)
        log = source.with_suffix('.log')
        log_hash = sha(log)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        assert hashes(target) == before
        shutil.copy2(log, target / 'process.log')
        assert sha(target / 'process.log') == log_hash
        assert hashes(source) == before and sha(log) == log_hash
        if source == FORMAL:
            logs = target / 'regression_logs'
            logs.mkdir()
            for stage in ('smoke', 'formal'):
                regression = ROOT / f'tmp/getup_intervention_dynamics_r173_regression_{stage}_20261009.log'
                assert 'PASS' in regression.read_text()
                shutil.copy2(regression, logs / regression.name)
        write(target / 'snapshot.json', dict(terminal_result_saved=True,
              formal_dynamic_attempts=181 if source == FORMAL else 0,
              original_data_attempts=175 if source == FORMAL else 0,
              read_only=source != FORMAL, new_controller=False,
              full_task_completed=False, hardware_readiness=False, qualification_run=False,
              historical_progress_is_not_terminal_status=True, source_hashes_unchanged=True))
        write(target / 'source_file_hashes.json', before)
        write(target / 'artifact_hashes.json', hashes(target))
        assert all(p.stat().st_size < 100 * 1024**2 for p in target.rglob('*') if p.is_file())
        subprocess.run(['git', 'add', '-f', str(target.relative_to(REPO))], cwd=REPO, check=True)
        all_sources.append((source, before, log, log_hash))

    for name in (Path(__file__).name, 'audit_getup_intervention_dynamics_r174.py'):
        dest = REPO / 'scripts' / name
        shutil.copy2(Path(__file__).with_name(name), dest)
        subprocess.run(['git', 'add', 'scripts/' + name], cwd=REPO, check=True)
    shutil.copy2(ROOT / 'tmp' / DOC, REPO / DOC)
    subprocess.run(['git', 'add', DOC], cwd=REPO, check=True)
    assert all(hashes(source) == before and sha(log) == log_hash for source, before, log, log_hash in all_sources)
    subprocess.run(['git', '-c', 'gc.auto=0', 'commit', '--quiet', '-m',
                    'Preserve R173 complete intervention attempts and learner, R174 causal pair audit; no recovery policy promotion'], cwd=REPO, check=True)
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD', '^' + args.base], cwd=REPO, check=True)
    assert not git('status', '--porcelain')
    print(json.dumps(dict(head=git('rev-parse', 'HEAD'), parent=args.base, bundle=str(bundle),
                         bytes=bundle.stat().st_size, sha256=sha(bundle), terminal_result_saved=True,
                         current_best_unified_successes=18, new_controller=False)), flush=True)


if __name__ == '__main__':
    main()
