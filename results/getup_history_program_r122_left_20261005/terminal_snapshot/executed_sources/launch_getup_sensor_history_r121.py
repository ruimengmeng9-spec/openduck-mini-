"""Bounded simulation data audit only, no process signals or hardware."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.probe_getup_sensor_history_r121 import ROOT,OUTPUT


def main():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        modules=[c.decode(errors='replace') for c in command if c.startswith(b'diagnostics.')]
        if any('getup' in c and c!='diagnostics.launch_getup_sensor_history_r121' for c in modules):
            raise RuntimeError('Other get-up diagnostic still active: '+str(proc.name))
    cwd=ROOT/'projects/Open_Duck_Playground';log=OUTPUT.with_suffix('.log')
    assert not OUTPUT.exists() and not log.exists()
    smoke=ROOT/'outputs/getup_sensor_history_r121_smoke_20261005/results.json'
    assert json.loads(smoke.read_text())['full_original_teacher_bitwise_equal']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    with OUTPUT.with_name(OUTPUT.name+'_tests.log').open('xb') as stream:
        tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_sensor_history_r121','-v'],
            cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    assert tests.returncode==0
    cmd=[str(cwd/'.venv/bin/python'),'-u','-m','diagnostics.probe_getup_sensor_history_r121']
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=cmd,log=str(log),simulation_only=True)))


if __name__=='__main__':main()
