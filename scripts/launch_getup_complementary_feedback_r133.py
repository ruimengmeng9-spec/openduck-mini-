import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.probe_getup_complementary_feedback_r133 import ROOT,OUTPUT


def main():
    smoke=json.loads((ROOT/'outputs/getup_complementary_feedback_r133_smoke_20261006/results.json').read_text());assert smoke['smoke']
    for report in smoke['reports']:
        row=[r for r in report['rows'] if r['case_seed'] is None][0]
        assert row['success'] and all(row['original_complete_or_prefix_trace_bitwise_equal'].values())
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'diagnostics.probe_getup_complementary_feedback_r133' in argv:raise RuntimeError('Existing R133; no duplicate launch')
    log=OUTPUT.with_suffix('.log');assert not OUTPUT.exists() and not log.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    command=[sys.executable,'-u','-m','diagnostics.probe_getup_complementary_feedback_r133','--workers','6']
    with log.open('x') as stream:process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command=command,log=str(log))))


if __name__=='__main__':main()
