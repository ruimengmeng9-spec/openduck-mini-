"""Append immutable completed R119 evidence to the clean independent publisher."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_robust_neighborhood_r119 import ROOT, OUTPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    assert args.snapshot.replace('_', '').isalnum()
    repo = ROOT / 'github/openduck-mini-'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == args.base
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    terminal = (OUTPUT / 'results.json').exists()
    history = json.loads((OUTPUT / 'history.json').read_text()) if (OUTPUT / 'history.json').exists() else []
    closed = history[-1]['generation'] if history else 0
    target = repo / 'results' / OUTPUT.name / args.snapshot
    assert not target.exists()
    target.mkdir(parents=True)
    shutil.copytree(OUTPUT / 'executed_sources', target / 'executed_sources')
    for name in ('contract.json', 'frozen_profiles.npz', 'fixed_comparisons.json'):
        if (OUTPUT / name).exists():
            shutil.copy2(OUTPUT / name, target / name)
    if terminal:
        shutil.copy2(OUTPUT / 'results.json', target / 'results.json')
    (target / 'closed_history.json').write_text(json.dumps(history, indent=2))
    # Closed generation checkpoint files are never updated by training.
    for file in (OUTPUT / 'checkpoints').rglob('generation_*.npz') if (OUTPUT / 'checkpoints').exists() else []:
        if int(file.stem.split('_')[-1]) <= closed:
            dest = target / file.relative_to(OUTPUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    counts = {}
    for group in ('fixed_comparisons', 'short_trials', 'full_checks', 'final_common_programs'):
        source = OUTPUT / group
        if not source.exists():
            continue
        count = 0
        for marker in sorted(source.rglob('result.json')):
            if group == 'short_trials' and not terminal:
                gen = int(marker.relative_to(source).parts[0].split('_')[-1])
                if gen > closed:
                    continue
            # result.json is written last, so its trajectory and parameters are complete.
            shutil.copytree(marker.parent, target / group / marker.parent.relative_to(source))
            count += 1
        counts[group] = count
    smoke = ROOT / 'outputs/getup_robust_neighborhood_r119_smoke_20261005'
    assert json.loads((smoke / 'results.json').read_text())['parity_passed']
    shutil.copytree(smoke, target / 'independent_smoke')
    shutil.copy2(OUTPUT.with_suffix('.log'), target / 'parent_log_snapshot.log')
    shutil.copy2(OUTPUT.with_name(OUTPUT.name + '_tests.log'), target / 'prelaunch_tests.log')
    cwd = ROOT / 'projects/Open_Duck_Playground'
    tests = subprocess.run([str(cwd / '.venv/bin/python'), '-m', 'unittest',
        'diagnostics.test_getup_robust_neighborhood_r119', '-v'], cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (target / 'snapshot_regression_tests.log').write_text(tests.stdout)
    assert tests.returncode == 0
    (target / 'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=terminal,
        completed_generations=closed, completed_trial_counts=counts, teacher_data_only=True,
        unified_policy_success=False, independent_qualification_run=False,
        full_task_completed=False, hardware_readiness=False), indent=2))
    (target / 'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)): digest(f)
        for f in target.rglob('*') if f.is_file()}, indent=2))
    names = ('search_getup_robust_neighborhood_r119.py', 'test_getup_robust_neighborhood_r119.py',
        'launch_getup_robust_neighborhood_r119.py', Path(__file__).name)
    for name in names:
        src, dst = Path(__file__).with_name(name), repo / 'scripts' / name
        if dst.exists():
            assert digest(dst) == digest(src)
        else:
            shutil.copy2(src, dst)
    doc = 'GETUP_ROBUST_NEIGHBORHOOD_R119_20261005.md'
    src, dst = Path(__file__).with_name(doc), repo / doc
    if dst.exists():
        assert digest(dst) == digest(src)
    else:
        shutil.copy2(src, dst)
    subprocess.run(['git', 'add', doc, *['scripts/' + n for n in names]], cwd=repo, check=True)
    subprocess.run(['git', 'add', '-f', str(target.relative_to(repo))], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'gc.auto=0', 'commit', '--quiet', '-m',
        'Preserve common multi-start teacher search R119 ' + args.snapshot], cwd=repo, check=True)
    bundle = ROOT / 'tmp' / ('getup_robust_neighborhood_r119_' + args.snapshot + '_20261005.bundle')
    assert not bundle.exists()
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD', '^' + args.base], cwd=repo, check=True)
    print(json.dumps(dict(bundle=str(bundle), head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), terminal_result_saved=terminal,
        completed_generations=closed, completed_trial_counts=counts)), flush=True)


if __name__ == '__main__':
    main()
