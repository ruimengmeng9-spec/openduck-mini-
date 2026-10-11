"""One actual full-interface preflight, zero skip, no dynamic integration."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
OUT=ROOT/'outputs/getup_total_target_naf_r204_preflight_20261011'
EXIT=ROOT/'tmp/getup_total_target_naf_r204_preflight_exit_20261011.json'
EXPECTED='3b2ae6dc6d745cf02f2ba57c7552251305c1c179'
assert not OUT.exists() and not EXIT.exists()
names=['train_getup_total_target_naf_r204.py','total_target_naf_kernel_r204.py','test_getup_total_target_naf_r204.py','launch_getup_total_target_naf_r204.py']
sources=[PROJECT/'diagnostics'/n for n in names]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
for p in sources:ast.parse(p.read_text())
assert hashes[str(sources[1])]=='6a718980ca2960a5b901d99331ffe6ec12867bc753677ab545300179b84023b4'
assert subprocess.check_output(['git','-C',str(ROOT/'github/openduck-mini-'),'rev-parse','HEAD'],text=True).strip()==EXPECTED
assert not subprocess.check_output(['git','-C',str(ROOT/'github/openduck-mini-'),'status','--porcelain'],text=True).strip()
numeric=json.loads((ROOT/'tmp/getup_total_target_naf_r204_numeric_exit_20261011.json').read_text())
assert numeric['natural_exit'] and numeric['exit_code']==0 and numeric['source_hashes_unchanged']
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','OPEN_DUCK_R204_TEST_CAPTURE':str(OUT/'executed_sources')}
# Import the new wrapper for idle/storage checks, never run an environment or numeric validator.
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','JAX_PLATFORMS'):
    os.environ[name]=env[name]
import sys
sys.path.insert(0,str(PROJECT))
from diagnostics import launch_getup_total_target_naf_r204 as launch
launch.idle();budget=launch.run.storage_gate(start=True)
OUT.mkdir()
begin=time.monotonic()
with (OUT/'regression.log').open('x') as log:
    child=subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','diagnostics.test_getup_total_target_naf_r204'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in sources)
result=json.loads((OUT/'interface_result.json').read_text()) if (OUT/'interface_result.json').exists() else None
passed=bool(child.returncode==0 and result and result['tests_run']==16 and result['skipped']==result['errors']==result['failures']==0 and result['source_hashes_unchanged'] and unchanged)
receipt=dict(natural_exit=True,exit_code=child.returncode,elapsed_seconds=time.monotonic()-begin,new_dynamic_attempts=0,source_hashes_unchanged=unchanged,source_hashes=hashes,full_interface_regression_passed=passed,tests=result,full_smoke_run=False,formal_run=False,resources_before=budget)
with EXIT.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(0 if passed else 1)
