"""Preserve R120 terminal or completed partial evidence without overwrite."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.probe_getup_common_program_r120 import ROOT, OUTPUT
from diagnostics.getup_independent_native import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    assert args.snapshot.replace('_', '').isalnum()
    repo = ROOT / 'github/openduck-mini-'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == args.base
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    target = repo / 'results' / OUTPUT.name / args.snapshot
    assert not target.exists()
    target.mkdir(parents=True)
    shutil.copytree(OUTPUT / 'executed_sources', target / 'executed_sources')
    for name in ('contract.json', 'frozen_common_program.npz', 'parity.json', 'results.json'):
        if (OUTPUT / name).exists():
            shutil.copy2(OUTPUT / name, target / name)
    counts = {}
    for group in ('parity', 'development'):
        source = OUTPUT / group
        count = 0
        for marker in sorted(source.rglob('result.json')):
            shutil.copytree(marker.parent, target / group / marker.parent.relative_to(source))
            count += 1
        counts[group] = count
    shutil.copy2(OUTPUT.with_suffix('.log'), target / 'parent_log_snapshot.log')
    shutil.copy2(OUTPUT.with_name(OUTPUT.name + '_tests.log'), target / 'prelaunch_tests.log')
    cwd = ROOT / 'projects/Open_Duck_Playground'
    tests = subprocess.run([str(cwd / '.venv/bin/python'), '-m', 'unittest',
        'diagnostics.test_getup_common_program_r120', '-v'], cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (target / 'snapshot_regression_tests.log').write_text(tests.stdout)
    assert tests.returncode == 0
    (target / 'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target / 'results.json').exists(),
        completed_trial_counts=counts, independent_qualification_run=False, hardware_readiness=False,
        full_task_completed=False), indent=2))
    (target / 'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)): digest(f)
        for f in target.rglob('*') if f.is_file()}, indent=2))
    names = ('probe_getup_common_program_r120.py', 'test_getup_common_program_r120.py',
        'launch_getup_common_program_r120.py', Path(__file__).name)
    for name in names:
        src, dst = Path(__file__).with_name(name), repo / 'scripts' / name
        if dst.exists():
            assert digest(dst) == digest(src)
        else:
            shutil.copy2(src, dst)
    doc = 'GETUP_COMMON_PROGRAM_R119_R120_20261005.md'
    src, dst = Path(__file__).with_name(doc), repo / doc
    assert not dst.exists()
    shutil.copy2(src, dst)
    subprocess.run(['git', 'add', doc, *['scripts/' + n for n in names]], cwd=repo, check=True)
    subprocess.run(['git', 'add', '-f', str(target.relative_to(repo))], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'gc.auto=0', 'commit', '--quiet', '-m',
        'Preserve frozen common-program validation R120 ' + args.snapshot], cwd=repo, check=True)
    bundle = ROOT / 'tmp' / ('getup_common_program_r120_' + args.snapshot + '_20261005.bundle')
    assert not bundle.exists()
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD', '^' + args.base], cwd=repo, check=True)
    print(json.dumps(dict(bundle=str(bundle), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=repo, text=True).strip(), terminal_result_saved=(target / 'results.json').exists(),
        completed_trial_counts=counts)), flush=True)


if __name__ == '__main__':
    main()
