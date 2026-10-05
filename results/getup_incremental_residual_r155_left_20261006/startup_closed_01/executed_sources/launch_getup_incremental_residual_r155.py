import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.train_getup_incremental_residual_r155 import ROOT,OUTPUT,SMOKE

def main():
    result=json.loads((SMOKE/'results.json').read_text());assert result['smoke']
    assert (SMOKE/'tests.log').read_text().strip().endswith('OK')
    for row in result['parity']['rows']:assert all(row['frozen_R134_trace_bitwise_equal'].values())
    nominal=next(r for r in result['nonzero']['rows'] if r['case_seed'] is None)
    assert nominal['success'] and all(nominal['frozen_R134_trace_bitwise_equal'].values())
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'diagnostics.train_getup_incremental_residual_r155' in argv:raise RuntimeError('Existing R155; wait for natural exit, no duplicate')
    log=OUTPUT.with_suffix('.log');assert not OUTPUT.exists() and not log.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    command=[sys.executable,'-u','-m','diagnostics.train_getup_incremental_residual_r155']
    with log.open('x') as stream:process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command=command,log=str(log))))

if __name__=='__main__':main()
