"""Fresh bounded simulation experiment, no hardware or process signals."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.train_getup_history_program_r122 import ROOT,OUTPUT


def main():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        modules=[c.decode(errors='replace') for c in args if c.startswith(b'diagnostics.')]
        if any('getup' in c and c!='diagnostics.launch_getup_history_program_r122' for c in modules):
            raise RuntimeError('Other get-up diagnostic active '+proc.name)
    cwd=ROOT/'projects/Open_Duck_Playground';log=OUTPUT.with_suffix('.log')
    assert not OUTPUT.exists() and not log.exists()
    smoke=json.loads((ROOT/'outputs/getup_history_program_r122_smoke_20261005/results.json').read_text())
    assert smoke['smoke'] and len(smoke['interface_trials'])==2
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    with OUTPUT.with_name(OUTPUT.name+'_tests.log').open('xb') as stream:
        test=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_history_program_r122','-v'],
            cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    assert test.returncode==0
    cmd=[str(cwd/'.venv/bin/python'),'-u','-m','diagnostics.train_getup_history_program_r122']
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=cmd,log=str(log),simulation_only=True)))


if __name__=='__main__':main()
