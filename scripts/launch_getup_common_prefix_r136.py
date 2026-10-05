import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.probe_getup_common_prefix_r136 import ROOT,OUTPUT


def main():
    r=json.loads((ROOT/'outputs/getup_common_prefix_r136_smoke_20261006/results.json').read_text());assert r['smoke']
    for delay,groups in r['reports'].items():
        for group in groups:
            for row in group['rows']:
                if delay=='0':assert all(row['frozen_program_complete_parity'].values())
                if delay=='10':assert all(row['common_ten_control_prefix_bitwise_equal'].values())
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'diagnostics.probe_getup_common_prefix_r136' in argv:raise RuntimeError('Existing R136; no duplicate')
    log=OUTPUT.with_suffix('.log');assert not OUTPUT.exists() and not log.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    cmd=[sys.executable,'-u','-m','diagnostics.probe_getup_common_prefix_r136','--workers','6']
    with log.open('x') as stream:process=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command=cmd,log=str(log))))


if __name__=='__main__':main()
