"""Launch the once-only fixed complete development after smoke naturally exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics.probe_getup_fixed_bilateral_r162 import OUTPUT,run

def main():
    smoke=run.ROOT/'outputs/getup_fixed_bilateral_r162_smoke_20261006'
    data=json.loads((smoke/'results.json').read_text());assert data['smoke']
    assert len(data['checks'])==6 and all(all(c['fields'].values()) for c in data['checks'])
    assert (run.ROOT/'tmp/getup_fixed_bilateral_r162_regression_20261006.log').read_text().strip().endswith('OK')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'-m' in argv:
            i=argv.index(b'-m')
            if len(argv)>i+1 and argv[i+1].startswith((b'diagnostics.train_getup_',b'diagnostics.probe_getup_')):
                raise RuntimeError('Existing simulation or smoke must naturally exit; no duplicate launch')
    log=OUTPUT.with_suffix('.log');assert not OUTPUT.exists() and not log.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    command=[sys.executable,'-u','-m','diagnostics.probe_getup_fixed_bilateral_r162']
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command=command,log=str(log))),flush=True)

if __name__=='__main__':main()
