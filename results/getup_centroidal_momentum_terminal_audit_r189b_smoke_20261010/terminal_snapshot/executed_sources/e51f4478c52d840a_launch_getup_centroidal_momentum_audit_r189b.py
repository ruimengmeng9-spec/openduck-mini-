"""Unique independent read-only smoke then formal audit; no signals."""
import json
import os
import shutil
import subprocess
import sys
from diagnostics import audit_getup_centroidal_momentum_terminal_r189b as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}

def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(name in line for name in ('train_getup_','probe_getup_','audit_getup_','validate_centroidal')) for line in lines),'Existing task; do not duplicate or signal'

def tests(label):
    log=run.ROOT/'tmp'/('getup_centroidal_momentum_audit_r189b_'+label+'_regression_20261010.log')
    with log.open('x') as f:subprocess.run([sys.executable,'-m','diagnostics.test_getup_centroidal_momentum_audit_r189b'],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    idle();assert not run.OUTPUT.exists() and not run.SMOKE.exists()
    assert shutil.disk_usage(run.ROOT).free>(10+.25)*1024**3
    tests('initial')
    with run.SMOKE.with_suffix('.log').open('x') as f:subprocess.run([sys.executable,'-u','-m','diagnostics.audit_getup_centroidal_momentum_terminal_r189b','--smoke'],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    idle();assert run.read(run.SMOKE/'results.json')['all_scalar_and_limits_bitwise']
    tests('formal');idle()
    with run.OUTPUT.with_suffix('.log').open('x') as f:subprocess.run([sys.executable,'-u','-m','diagnostics.audit_getup_centroidal_momentum_terminal_r189b'],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=dict(natural_exit=True,exit_code=0,new_dynamic_trajectories=0,full_task_completed=False)
    run.write(run.ROOT/'tmp/getup_centroidal_momentum_audit_r189b_launcher_result_20261010.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle()
        with (run.ROOT/'tmp/getup_centroidal_momentum_audit_r189b_launcher_20261010.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_centroidal_momentum_audit_r189b'],env=ENV,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,read_only=True,new_dynamic_trajectories=0)),flush=True)
    else:assert not sys.argv[1:];main()
