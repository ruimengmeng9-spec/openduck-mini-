"""One append-only failure archive invocation; records actual natural return."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
EXIT=ROOT/'tmp/getup_total_target_trpo_r201_failure_archive_exit_20261011.json'
LOG=ROOT/'tmp/getup_total_target_trpo_r201_failure_archive_20261011.log'
SOURCE=PROJECT/'diagnostics/archive_getup_total_target_trpo_failure_r201.py'
assert not EXIT.exists() and not LOG.exists()
sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest();start=time.monotonic()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
with LOG.open('x') as log:process=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.archive_getup_total_target_trpo_failure_r201'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
result=dict(natural_exit=True,exit_code=process.returncode,elapsed_seconds=time.monotonic()-start,new_dynamic_attempts=0,source_sha256=sha,source_hashes_unchanged=sha==hashlib.sha256(SOURCE.read_bytes()).hexdigest(),archive_result_saved=(ROOT/'tmp/getup_total_target_trpo_r201_failure_archive_result_20261011.json').exists(),log=str(LOG))
with EXIT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True);raise SystemExit(process.returncode)
