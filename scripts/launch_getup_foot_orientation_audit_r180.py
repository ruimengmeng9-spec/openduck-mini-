"""Finite read-only audit pipeline, no dynamic replay or process signals."""
import os
import subprocess
import sys
from diagnostics import audit_getup_foot_orientation_terminal_r180 as audit

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}

def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_')) for line in lines),'Existing task active; no signals or duplicate audit'

def main():
    idle();assert not audit.SMOKE.exists() and not audit.OUTPUT.exists()
    for smoke in (True,False):
        idle();output=audit.SMOKE if smoke else audit.OUTPUT
        command=[sys.executable,'-u','-m','diagnostics.audit_getup_foot_orientation_terminal_r180']+(['--smoke'] if smoke else [])
        with output.with_suffix('.log').open('x') as stream:subprocess.run(command,env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('R180_READ_ONLY_SMOKE_AND_FORMAL_NATURAL_EXIT',flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();log=audit.run.ROOT/'tmp/getup_foot_orientation_audit_r180_launcher_20261010.log'
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_foot_orientation_audit_r180'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=ENV,start_new_session=True)
        print(child.pid,flush=True)
    else:
        assert not sys.argv[1:];main()
