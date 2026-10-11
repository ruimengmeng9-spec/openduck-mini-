"""One background launch request; request exit is not smoke/formal terminal exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
RECEIPT=ROOT/'tmp/getup_total_target_vmpo_r202_launch_request_20261011.json'
assert not RECEIPT.exists()
sources=[PROJECT/'diagnostics'/n for n in ('train_getup_total_target_vmpo_r202.py','total_target_vmpo_kernel_r202.py','test_getup_total_target_vmpo_r202.py','launch_getup_total_target_vmpo_r202.py')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
begin=time.monotonic()
result=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.launch_getup_total_target_vmpo_r202','--launch'],cwd=PROJECT,env=env,capture_output=True,text=True)
unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in sources)
receipt=dict(stage='unique_launch_request_only',request_natural_exit_code=result.returncode,elapsed_seconds=time.monotonic()-begin,stdout=result.stdout,stderr=result.stderr,source_hashes=hashes,source_hashes_unchanged=unchanged,smoke_natural_exit_not_yet_observed=True,formal_not_yet_observed=True,full_task_completed=False,hardware_readiness=False)
with RECEIPT.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
