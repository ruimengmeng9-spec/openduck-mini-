"""Unique finite R204 numeric validator with actual natural exit and source immutability."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck')
HERE=ROOT/'tmp/getup_total_target_naf_numeric_r204_20261011'
EXE=ROOT/'projects/Open_Duck_Playground/.venv/bin/python'
LOG=ROOT/'tmp/getup_total_target_naf_r204_numeric_20261011.log'
EXIT=ROOT/'tmp/getup_total_target_naf_r204_numeric_exit_20261011.json'
assert not LOG.exists() and not EXIT.exists()
repo=ROOT/'github/openduck-mini-'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='3b2ae6dc6d745cf02f2ba57c7552251305c1c179'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
    parts=line.strip().split(None,1)
    if len(parts)!=2:continue
    pid,args=parts
    if int(pid)==os.getpid():continue
    first=Path(args.split()[0]).name
    if 'python' not in first and first!='git':continue
    assert not ('-m diagnostics.' in args or 'publish_getup' in args or 'git push' in args),'Existing task; no signals/restarts'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
begin=time.monotonic()
with LOG.open('x') as log:
    result=subprocess.run([str(EXE),'-u',str(HERE/'validate_total_target_naf_numeric_r204.py')],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
receipt=dict(natural_exit=True,actual_subprocess_return_observed=True,exit_code=result.returncode,
    elapsed_seconds=time.monotonic()-begin,new_dynamic_attempts=0,source_hashes_unchanged=True,source_hashes=hashes,
    log=str(LOG),result_path=str(ROOT/'outputs/getup_total_target_naf_numeric_r204_20261011/result.json'))
with EXIT.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)

