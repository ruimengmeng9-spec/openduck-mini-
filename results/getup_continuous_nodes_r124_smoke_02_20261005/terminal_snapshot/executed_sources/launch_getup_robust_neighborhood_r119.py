"""Launch a fresh bounded simulation task, never send process signals."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.search_getup_robust_neighborhood_r119 import ROOT, OUTPUT


def main():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().split(b'\0')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        # Refuse concurrent get-up diagnostics, leave every process untouched.
        modules = [c.decode(errors='replace') for c in command if c.startswith(b'diagnostics.')]
        if any('getup' in c and c != 'diagnostics.launch_getup_robust_neighborhood_r119' for c in modules):
            raise RuntimeError('Another get-up diagnostic still active: ' + str(proc.name))
    cwd = ROOT / 'projects/Open_Duck_Playground'
    log = OUTPUT.with_suffix('.log')
    assert not OUTPUT.exists() and not log.exists()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    tests_log = OUTPUT.with_name(OUTPUT.name + '_tests.log')
    with tests_log.open('xb') as stream:
        tests = subprocess.run([str(cwd / '.venv/bin/python'), '-m', 'unittest',
            'diagnostics.test_getup_robust_neighborhood_r119', '-v'], cwd=cwd, env=env,
            stdout=stream, stderr=subprocess.STDOUT)
    assert tests.returncode == 0, 'Regression tests failed; no training launched'
    cmd = [str(cwd / '.venv/bin/python'), '-u', '-m', 'diagnostics.search_getup_robust_neighborhood_r119']
    with log.open('xb') as stream:
        child = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=child.pid, command=cmd, log=str(log), simulation_only=True)))


if __name__ == '__main__':
    main()
