"""Unique read-only smoke then formal audit; never launch dynamic training."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
PYTHON=PROJECT/'.venv/bin/python'
BASE=ROOT/'outputs/getup_full_episode_pg_terminal_audit_r182'


def main():
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    processes=subprocess.check_output(['ps','-u','shijinsheng','-ww','-o','args='],text=True)
    active=[line for line in processes.splitlines() if 'python' in line and '-m diagnostics.' in line and any(s in line for s in ('train_getup','probe_getup','audit_getup','launch_getup')) and 'launch_getup_full_episode_pg_audit_r182' not in line]
    assert not active,active
    assert shutil.disk_usage(ROOT).free>10*1024**3
    for suffix in ('_smoke_20261010','_20261010'):assert not Path(str(BASE)+suffix).exists()
    log=Path(str(BASE)+'_regression_20261010.log');assert not log.exists()
    with log.open('x') as stream:
        result=subprocess.run([str(PYTHON),'-m','unittest','diagnostics.test_getup_full_episode_pg_r181'],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0
    for smoke in (True,False):
        suffix='_smoke_20261010' if smoke else '_20261010';log=Path(str(BASE)+suffix+'.log');assert not log.exists()
        args=[str(PYTHON),'-u','-m','diagnostics.audit_getup_full_episode_pg_terminal_r182']+(['--smoke'] if smoke else [])
        with log.open('x') as stream:result=subprocess.run(args,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,(args,result.returncode)
        print('R182_NATURAL_READ_ONLY_EXIT',smoke,flush=True)


if __name__=='__main__':main()
