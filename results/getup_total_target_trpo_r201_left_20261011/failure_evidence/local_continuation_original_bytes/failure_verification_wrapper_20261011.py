"""Single new read-only failure audit; records actual subprocess natural return."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
SOURCE=PROJECT/'diagnostics/verify_getup_total_target_trpo_failure_r201.py'
DEST=ROOT/'outputs/getup_total_target_trpo_r201_failure_verification_20261011'
EXIT=ROOT/'tmp/getup_total_target_trpo_r201_failure_verification_exit_20261011.json'
PROOF=ROOT/'tmp/getup_total_target_trpo_r201_failure_verification_20261011.json'
assert not DEST.exists() and not EXIT.exists() and not PROOF.exists()
ast.parse(SOURCE.read_text());sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic()
# Log lives outside the audit's initially absent destination; never truncates old logs.
LOG=ROOT/'tmp/getup_total_target_trpo_r201_failure_verification_20261011.log'
with LOG.open('x') as log:
    process=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.verify_getup_total_target_trpo_failure_r201'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
result=dict(natural_exit=True,exit_code=process.returncode,elapsed_seconds=time.monotonic()-start,new_dynamic_attempts=0,source_sha256=sha,source_hash_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sha,proof_saved=PROOF.exists(),log=str(LOG))
with EXIT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
raise SystemExit(process.returncode)
