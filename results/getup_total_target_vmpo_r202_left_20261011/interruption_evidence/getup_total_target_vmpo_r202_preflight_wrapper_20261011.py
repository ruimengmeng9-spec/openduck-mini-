"""Unique offline actual-interface preflight; no new integration or old source capture."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
OUT=ROOT/'outputs/getup_total_target_vmpo_r202_preflight_20261011'
EXIT=ROOT/'tmp/getup_total_target_vmpo_r202_preflight_exit_20261011.json'
EXPECTED='9b4edd92c86f0c689f15f904c0298cae52f17b82'
assert not OUT.exists() and not EXIT.exists()
names=['train_getup_total_target_vmpo_r202.py','total_target_vmpo_kernel_r202.py','test_getup_total_target_vmpo_r202.py','launch_getup_total_target_vmpo_r202.py']
sources=[PROJECT/'diagnostics'/n for n in names]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
for p in sources:ast.parse(p.read_text())
assert subprocess.check_output(['git','-C',str(ROOT/'github/openduck-mini-'),'rev-parse','HEAD'],text=True).strip()==EXPECTED
assert not subprocess.check_output(['git','-C',str(ROOT/'github/openduck-mini-'),'status','--porcelain'],text=True).strip()
assert json.loads((ROOT/'tmp/getup_total_target_vmpo_r202b_numeric_exit_20261011.json').read_text())['exit_code']==0
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','OPEN_DUCK_R202_TEST_CAPTURE':str(OUT/'executed_sources')}
OUT.mkdir()
begin=time.monotonic()
with (OUT/'regression.log').open('x') as log:
    result=subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','diagnostics.test_getup_total_target_vmpo_r202'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in sources)
assert unchanged
receipt=dict(natural_exit=True,exit_code=result.returncode,elapsed_seconds=time.monotonic()-begin,new_dynamic_attempts=0,source_hashes_unchanged=unchanged,source_hashes=hashes,full_interface_regression_passed=result.returncode==0,full_smoke_run=False,formal_run=False)
with EXIT.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
