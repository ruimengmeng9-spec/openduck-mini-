"""Unique R205 numeric run with true subprocess exit, idle and immutable-source guards."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck')
HERE=ROOT/'tmp/getup_total_target_cacla_numeric_r205_20261011'
EXE=ROOT/'projects/Open_Duck_Playground/.venv/bin/python'
LOG=ROOT/'tmp/getup_total_target_cacla_r205_numeric_20261011.log'
EXIT=ROOT/'tmp/getup_total_target_cacla_r205_numeric_exit_20261011.json'
assert not LOG.exists() and not EXIT.exists()
repo=ROOT/'github/openduck-mini-'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='d485cfc1c8eb31a4762ad5b711f8ae4d6fde6866'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
    parts=line.strip().split(None,1)
    if len(parts)!=2:continue
    pid,args=parts
    if int(pid)==os.getpid():continue
    first=Path(args.split()[0]).name
    if 'python' not in first and first!='git':continue
    assert not ('-m diagnostics.' in args or 'publish_getup' in args or 'git push' in args or
                ('train_getup' in args and ' -c ' not in args)),'Existing task; no signals/restarts'
for d in (ROOT/'outputs',ROOT/'tmp',ROOT/'projects/Open_Duck_Playground/diagnostics'):
    for p in d.iterdir():
        m=re.search('r([0-9]{3})(?:[^0-9]|$)',p.name,re.I)
        if m and 205<=int(m.group(1))<=299:
            assert p in (HERE,Path(__file__).resolve()),'Unexpected current or higher task; do not duplicate'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
ownsha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
begin=time.monotonic()
with LOG.open('x',encoding='utf-8') as log:
    result=subprocess.run([str(EXE),'-u',str(HERE/'validate_total_target_cacla_numeric_r205.py')],
        cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
unchanged=hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file()}
ownunchanged=ownsha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt=dict(natural_exit=True,actual_subprocess_return_observed=True,exit_code=result.returncode,
    elapsed_seconds=time.monotonic()-begin,new_dynamic_attempts=0,source_hashes_unchanged=unchanged,
    wrapper_hash_unchanged=ownunchanged,wrapper_sha256=ownsha,source_hashes=hashes,log=str(LOG),
    result_path=str(ROOT/'outputs/getup_total_target_cacla_numeric_r205_20261011/result.json'))
with EXIT.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
assert unchanged and ownunchanged
raise SystemExit(result.returncode)

