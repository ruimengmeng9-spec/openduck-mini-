"""Launch once after independent complete smoke naturally exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
from diagnostics import train_getup_set_decision_r157 as run

def main():
    lines=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(t in line for t in ('train_getup_','probe_getup_')) for line in lines),'Another getup simulation diagnostic still active'
    assert not run.OUTPUT.exists()
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    assert all(len(v['rows'])==3 and not v['physical_failures'] and v['nominal_success'] for v in smoke['reports'].values())
    assert all(all(r['selected_fixed_R133_program_bitwise_equal'].values()) for v in smoke['reports'].values() for r in v['rows'])
    log=run.ROOT/'tmp/getup_set_decision_r157_regression_20261006.log';assert not log.exists()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    with log.open('w') as f:subprocess.run([sys.executable,'-m','diagnostics.test_getup_set_decision_r157'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    parent_log=run.OUTPUT.with_suffix('.log');assert not parent_log.exists()
    with parent_log.open('w') as f:
        process=subprocess.Popen([sys.executable,'-u','-m','diagnostics.train_getup_set_decision_r157','--workers','6'],
            stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command='diagnostics.train_getup_set_decision_r157 --workers 6',log=str(parent_log))))

if __name__=='__main__':main()
