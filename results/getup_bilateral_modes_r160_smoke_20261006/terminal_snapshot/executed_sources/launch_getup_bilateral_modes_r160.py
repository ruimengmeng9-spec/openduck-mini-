"""Single launch after independent complete smoke natural exit, no signals."""
import json
import os
import subprocess
import sys
from diagnostics import train_getup_bilateral_modes_r160 as run

def main():
    lines=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(t in line for t in ('train_getup_','probe_getup_')) for line in lines),'A getup simulation diagnostic is still active'
    assert not run.OUTPUT.exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke'] and len(smoke['parity']['rows'])==3 and len(smoke['nonzero']['rows'])==3
    assert smoke['parity']['nominal_success'] and not smoke['parity']['physical_failures'] and smoke['nonzero']['nominal_success']
    assert all(all(r['frozen_R157_trace_bitwise_equal'].values()) for r in smoke['parity']['rows'])
    assert all(r['actual_left_extra_nonzero'] and r['actual_right_extra_nonzero'] for r in smoke['nonzero']['rows'] if r['case_seed'] is not None)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    regression=run.ROOT/'tmp/getup_bilateral_modes_r160_regression_20261006.log';assert not regression.exists()
    with regression.open('w') as f:subprocess.run([sys.executable,'-m','diagnostics.test_getup_bilateral_modes_r160'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    log=run.OUTPUT.with_suffix('.log');assert not log.exists()
    with log.open('w') as f:p=subprocess.Popen([sys.executable,'-u','-m','diagnostics.train_getup_bilateral_modes_r160'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print(json.dumps(dict(pid=p.pid,command='diagnostics.train_getup_bilateral_modes_r160',log=str(log))),flush=True)

if __name__=='__main__':main()
