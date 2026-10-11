"""Unique real natural-exit receipt for independent interrupted-data verification."""
import pathlib,subprocess,time,json,os,hashlib
root=pathlib.Path('/data/shijinsheng/open_duck');project=root/'projects/Open_Duck_Playground'
out=root/'tmp/getup_total_target_vmpo_r202_interruption_verification_exit_20261011.json'
log=root/'tmp/getup_total_target_vmpo_r202_interruption_verification_20261011.log'
source=project/'diagnostics/verify_getup_total_target_vmpo_interruption_r202.py'
assert not out.exists() and not log.exists()
sha=hashlib.sha256(source.read_bytes()).hexdigest()
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
start=time.monotonic()
with log.open('x') as stream:
    completed=subprocess.run([str(project/'.venv/bin/python'),'-u','-m','diagnostics.verify_getup_total_target_vmpo_interruption_r202'],cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
unchanged=sha==hashlib.sha256(source.read_bytes()).hexdigest()
record=dict(natural_exit=True,exit_code=completed.returncode,elapsed_s=time.monotonic()-start,
    new_dynamic_attempts=0,source_sha256=sha,source_unchanged=unchanged,
    original_formal_natural_exit_claim=False,missing_original_learning_trace_not_filled=True)
with out.open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
assert unchanged
raise SystemExit(completed.returncode)

