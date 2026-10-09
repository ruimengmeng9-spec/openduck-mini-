"""Unique formal launch after completed independent smoke; no process signals."""
import json
import os
import shutil
import subprocess
import sys
from diagnostics import train_getup_recurrent_readout_r175 as run


def main():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid,ppid,args'],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_')) for line in lines),'A getup diagnostic is active; wait without signals'
    assert not run.OUTPUT.exists() and shutil.disk_usage(run.ROOT).free>12*1024**3
    smoke=json.loads((run.SMOKE/'results.json').read_text())
    assert smoke['smoke'] and smoke['terminal_result_saved'] and len(smoke['parity']['rows'])==3 and len(smoke['nonzero']['rows'])==3
    assert smoke['parity']['nominal_success'] and not smoke['parity']['physical_failures'] and smoke['nonzero']['nominal_success']
    assert all(all(row['complete_R157_parity'].values()) for row in smoke['parity']['rows'])
    assert all(row['physics_unchanged'] for key in ('parity','nonzero') for row in smoke[key]['rows'])
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    log=run.ROOT/'tmp/getup_recurrent_readout_r175_formal_regression_20261009.log';assert not log.exists()
    with log.open('w') as f:subprocess.run([sys.executable,'-m','diagnostics.test_getup_recurrent_readout_r175'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    log=run.OUTPUT.with_suffix('.log');assert not log.exists()
    with log.open('w') as f:
        p=subprocess.Popen([sys.executable,'-u','-m','diagnostics.train_getup_recurrent_readout_r175'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print(json.dumps(dict(pid=p.pid,command='diagnostics.train_getup_recurrent_readout_r175',log=str(log))),flush=True)


if __name__=='__main__':main()
