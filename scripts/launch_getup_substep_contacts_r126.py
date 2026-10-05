"""Launch bounded simulation replay only after independent parity smoke."""
import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.probe_getup_substep_contacts_r126 import ROOT, OUTPUT


def main():
    cwd=ROOT/'projects/Open_Duck_Playground'
    smoke=ROOT/'outputs/getup_substep_contacts_r126_smoke_20261005/results.json'
    r=json.loads(smoke.read_text());assert r['smoke'] and r['all_full_trace_bitwise_equal'] and len(r['rows'])==4
    assert not OUTPUT.exists()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        assert b'diagnostics.probe_getup_substep_contacts_r126' not in args, 'Existing same replay'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    log=OUTPUT.with_suffix('.log')
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.probe_getup_substep_contacts_r126','--workers','6'],cwd=cwd,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,log=str(log),simulation_replay_only=True,no_new_training=True)),flush=True)


if __name__=='__main__':main()
