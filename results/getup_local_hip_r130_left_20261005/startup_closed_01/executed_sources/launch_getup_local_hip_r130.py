"""Launch only after independent complete smoke; never signal existing tasks."""
import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.train_getup_local_hip_r130 import ROOT,OUTPUT


def main():
    smoke=ROOT/'outputs/getup_local_hip_r130_smoke_20261005/results.json'
    report=json.loads(smoke.read_text());assert report['smoke']
    for row in report['parity']['rows']:
        assert all(row['original_complete_or_prefix_trace_bitwise_equal'].values())
    nominal=[r for r in report['nonzero']['rows'] if r['case_seed'] is None][0]
    assert nominal['success'] and nominal['valid'] and all(nominal['original_complete_or_prefix_trace_bitwise_equal'].values())
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'diagnostics.train_getup_local_hip_r130' in argv:raise RuntimeError('Existing R130 task: no duplicate launch')
    assert not OUTPUT.exists()
    log=OUTPUT.with_suffix('.log');assert not log.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    command=[sys.executable,'-u','-m','diagnostics.train_getup_local_hip_r130','--generations','8','--population','12','--workers','6']
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command=command,log=str(log),smoke=str(smoke))))


if __name__=='__main__':main()
