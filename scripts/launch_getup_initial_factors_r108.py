"""Launch one bounded simulation-only R108 diagnosis after parity smoke."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.probe_getup_anchor_r101 import ROOT


def main():
    smoke=ROOT/'outputs/getup_initial_factors_r108_smoke_20261005/results.json'
    report=json.loads(smoke.read_text())
    assert report['smoke'] and len(report['parity'])==4
    assert all(r['original_full_path_bitwise_parity'] for r in report['parity'])
    module='diagnostics.probe_getup_initial_factors_r108'
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
            except (PermissionError,FileNotFoundError,ProcessLookupError):continue
            if module.encode() in cmd:raise RuntimeError('R108 process still running; do not duplicate')
    output=ROOT/'outputs/getup_initial_factors_r108_left_20261005'
    assert not output.exists()
    logfile=output.with_suffix('.log')
    assert not logfile.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    cwd=ROOT/'projects/Open_Duck_Playground'
    with logfile.open('xb') as log:
        child=subprocess.Popen([str(cwd/'.venv/bin/python'),'-u','-m',module,'--output',str(output),'--workers','6'],
            cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,module=module,output=str(output),log=str(logfile),
        simulation_only=True,diagnostic_only=True)))


if __name__=='__main__':main()
