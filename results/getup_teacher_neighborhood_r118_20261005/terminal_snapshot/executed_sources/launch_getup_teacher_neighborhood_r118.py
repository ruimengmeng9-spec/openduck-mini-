"""Fresh bounded simulation diagnostic, no hardware or process signals."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.probe_getup_teacher_neighborhood_r118 import ROOT,OUTPUT


def main():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        assert not any(n.encode() in command for n in ('diagnostics.probe_getup_program_extrapolation_r117','diagnostics.probe_getup_teacher_neighborhood_r118')),'Related experiment still active'
    cwd=ROOT/'projects/Open_Duck_Playground';log=OUTPUT.with_suffix('.log');assert not OUTPUT.exists() and not log.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    cmd=[str(cwd/'.venv/bin/python'),'-u','-m','diagnostics.probe_getup_teacher_neighborhood_r118']
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=cmd,log=str(log),simulation_only=True)))


if __name__=='__main__':main()
