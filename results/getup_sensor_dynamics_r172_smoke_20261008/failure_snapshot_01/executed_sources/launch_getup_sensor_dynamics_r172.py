"""Exclusive finite offline identification launcher; no hardware or signals."""
import os
from pathlib import Path
import subprocess
import sys
from diagnostics import train_getup_sensor_dynamics_r172 as task

def main():
    assert not task.OUTPUT.exists() and not task.SMOKE.exists()
    ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[line for line in ps.splitlines() if ' -m diagnostics.' in line and any(s in line for s in ('train_getup_','audit_getup_','probe_getup_')) and 'launch_getup_sensor_dynamics_r172' not in line]
    assert not active,active
    task.regressions()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    for smoke,path in ((True,task.SMOKE),(False,task.OUTPUT)):
        cmd=[sys.executable,'-u','-m','diagnostics.train_getup_sensor_dynamics_r172','--output',str(path)]
        if smoke:cmd.append('--smoke')
        with Path(str(path)+'.log').open('x') as log:
            completed=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        assert completed.returncode==0,(completed.returncode,str(path)+'.log')
        assert (path/'results.json').exists()
        print('R172_NATURAL_EXIT',str(path),flush=True)

if __name__=='__main__':main()
