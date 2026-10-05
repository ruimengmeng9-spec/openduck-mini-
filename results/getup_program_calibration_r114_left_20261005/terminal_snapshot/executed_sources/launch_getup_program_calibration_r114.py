"""Single fresh simulation-only launch, no process signals."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.train_getup_program_calibration_r114 import ROOT,OUTPUT


def main():
    related=('diagnostics.train_getup_program_r113','diagnostics.train_getup_program_calibration_r114')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        assert not any(name.encode() in command for name in related),'Related experiment still active'
    log=OUTPUT.with_suffix('.log');assert not OUTPUT.exists() and not log.exists()
    cwd=ROOT/'projects/Open_Duck_Playground';env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    cmd=[str(cwd/'.venv/bin/python'),'-u','-m',related[-1]]
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=cmd,log=str(log),simulation_only=True)))


if __name__=='__main__':main()
