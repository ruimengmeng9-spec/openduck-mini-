from pathlib import Path
import hashlib, subprocess, json, time, os
root=Path('/data/shijinsheng/open_duck')
directory=root/'tmp/getup_total_target_trpo_numeric_r201_20261011'
expected={"evidence_preparation_numeric_20261011.json":"a6f00dfec7f799dba523e497448946a44fc462c9f406d89b2399f0d951d178ca","GETUP_TOTAL_TARGET_TRPO_PLAN_R201_20261011.json":"4469483fe79dd0f3f83913626b93726aa34dc22587bb1f51d2fdb659e2b9b127","test_total_target_trpo_numeric_r201.py":"bbb4801f28b973ad286b131059c7bb425ac475aca864cb526e4a20228f92164e","total_target_trpo_kernel_r201.py":"38c8f6961b5a53f349d5637ce7789f7c69862594c6e8cfb049ee6ae8f6b0849c","validate_total_target_trpo_numeric_r201.py":"8190036dcf11a6daf0958a7afb5f2a071482b181302b74957e0c31f3633852b3"}
for name,sha in expected.items():
 assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
command=[str(root/'projects/Open_Duck_Playground/.venv/bin/python'),'-u',str(directory/'validate_total_target_trpo_numeric_r201.py')]
begin=time.monotonic()
completed=subprocess.run(command,cwd=directory,env=env,text=True,capture_output=True)
receipt=dict(natural_exit=True,exit_code=completed.returncode,elapsed_s=time.monotonic()-begin,
 command=command,stdout=completed.stdout,stderr=completed.stderr,new_dynamic_attempts=0,
 source_hashes_unchanged=all(hashlib.sha256((directory/n).read_bytes()).hexdigest()==s for n,s in expected.items()),
 synthetic_numeric_only=True,interface_preflight_run=False,full_smoke_run=False,formal_run=False)
target=root/'tmp/getup_total_target_trpo_numeric_r201_exit_20261011.json'
assert not target.exists()
target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
log=root/'tmp/getup_total_target_trpo_numeric_r201_wrapper_20261011.log'
assert not log.exists()
log.write_text(completed.stdout+completed.stderr)
print(json.dumps(receipt,ensure_ascii=False),flush=True)
raise SystemExit(completed.returncode)
