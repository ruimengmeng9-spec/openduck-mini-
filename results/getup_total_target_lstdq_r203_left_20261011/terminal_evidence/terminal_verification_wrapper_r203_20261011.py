"""One new readonly R203 full-verification subprocess and real exit receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
SOURCE=PROJECT/'diagnostics/verify_getup_total_target_lstdq_terminal_r203.py'
SHA='9d3bffaf392b185fd610aa3742f65e36d2c196695f933e590eee7604be4938ab'
LOG=ROOT/'tmp/getup_total_target_lstdq_r203_terminal_verification_20261011.log'
EXIT=ROOT/'tmp/getup_total_target_lstdq_r203_terminal_verification_exit_20261011.json'
PROOF=ROOT/'tmp/getup_total_target_lstdq_r203_terminal_verification_20261011.json'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(SOURCE)==SHA
assert not LOG.exists() and not EXIT.exists() and not PROOF.exists()
started=time.monotonic()
with LOG.open('x') as stream:
    child=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.verify_getup_total_target_lstdq_terminal_r203'],
        cwd=PROJECT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'},stdout=stream,stderr=subprocess.STDOUT)
result=dict(natural_exit=True,actual_subprocess_return_observed=True,exit_code=child.returncode,
    elapsed_seconds=time.monotonic()-started,source_unchanged=digest(SOURCE)==SHA,source_sha256=SHA,
    new_dynamic_attempts=0,proof_saved=PROOF.exists(),proof_sha256=digest(PROOF) if PROOF.exists() else None,
    log=str(LOG),log_sha256=digest(LOG),protected_live_server_not_operated=True)
with EXIT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
assert result['source_unchanged']
raise SystemExit(child.returncode)

