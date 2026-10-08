"""Formal launch after independent complete smoke naturally exits; no signals."""
import json
import os
import subprocess
import sys
import numpy as np
from diagnostics import train_getup_tracking_memory_r170 as run

def main():
    lines=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
    assert not any(' -m diagnostics.' in l and any(t in l for t in ('train_getup_','probe_getup_','audit_getup_')) for l in lines),'A getup diagnostic is active; wait without signals'
    assert not run.OUTPUT.exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    assert len(smoke['parity']['rows'])==3 and len(smoke['nonzero']['rows'])==3
    assert smoke['parity']['nominal_success'] and not smoke['parity']['physical_failures'] and smoke['nonzero']['nominal_success']
    assert all(all(r['complete_R157_parity'].values()) for r in smoke['parity']['rows'])
    for case in [None,769002,773004]:
        row=next(r for r in smoke['nonzero']['rows'] if r['case_seed']==case)
        with np.load(run.SMOKE/'nonzero_smoke'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
            assert np.abs(z['tracking_memory_state']).max()<=1.
            assert not np.any(z['tracking_memory_extra_rad'][:,np.asarray(row['groups'])==2])
            assert not np.any(z['tracking_memory_extra_rad'][0])
            if case is not None:assert np.any(z['tracking_memory_extra_rad'])
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    regression=run.ROOT/'tmp/getup_tracking_memory_r170_regression_20261008.log';assert not regression.exists()
    with regression.open('w') as f:subprocess.run([sys.executable,'-m','diagnostics.test_getup_tracking_memory_r170'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    log=run.OUTPUT.with_suffix('.log');assert not log.exists()
    with log.open('w') as f:p=subprocess.Popen([sys.executable,'-u','-m','diagnostics.train_getup_tracking_memory_r170'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print(json.dumps(dict(pid=p.pid,command='diagnostics.train_getup_tracking_memory_r170',log=str(log))),flush=True)

if __name__=='__main__':main()
