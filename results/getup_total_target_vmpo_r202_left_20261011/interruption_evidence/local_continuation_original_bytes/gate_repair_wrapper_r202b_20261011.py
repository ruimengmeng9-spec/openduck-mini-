"""Unique real subprocess exit receipt for corrected offline gate only."""
import pathlib,subprocess,time,json,os
root=pathlib.Path('/data/shijinsheng/open_duck')
project=root/'projects/Open_Duck_Playground'
out=root/'tmp/getup_total_target_vmpo_r202b_gate_repair_exit_20261011.json'
log=root/'tmp/getup_total_target_vmpo_r202b_gate_repair_20261011.log'
assert not out.exists() and not log.exists()
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:6])
start=time.monotonic()
with log.open('x') as stream:
    completed=subprocess.run([str(project/'.venv/bin/python'),'-u','-m','diagnostics.launch_getup_total_target_vmpo_r202b','--gate-only'],cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
record=dict(natural_exit=True,exit_code=completed.returncode,elapsed_s=time.monotonic()-start,new_dynamic_attempts=0,formal_restarted=False,smoke_replayed=False)
with out.open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
raise SystemExit(completed.returncode)

