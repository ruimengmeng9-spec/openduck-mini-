import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.probe_getup_head_feedback_r128 import ROOT,OUTPUT


def main():
    smoke=json.loads((ROOT/'outputs/getup_head_feedback_r128_smoke_20261005/results.json').read_text())
    assert smoke['smoke'] and len(smoke['rows'])==4
    assert all(all(r['original_complete_trace_bitwise_equal'].values()) for r in smoke['rows'] if r['gain']==0. or r['case_seed'] is None)
    assert not OUTPUT.exists()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:argv=(path/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert b'diagnostics.probe_getup_head_feedback_r128' not in argv
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    with OUTPUT.with_suffix('.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.probe_getup_head_feedback_r128','--workers','4'],cwd=ROOT/'projects/Open_Duck_Playground',env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,log=str(OUTPUT.with_suffix('.log')),simulation_only=True,no_new_training=True)),flush=True)


if __name__=='__main__':main()
