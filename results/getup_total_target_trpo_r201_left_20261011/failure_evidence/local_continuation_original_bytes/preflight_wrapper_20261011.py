import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/data/shijinsheng/open_duck');project=root/'projects/Open_Duck_Playground'
expected={"total_target_trpo_kernel_r201.py":"38c8f6961b5a53f349d5637ce7789f7c69862594c6e8cfb049ee6ae8f6b0849c","train_getup_total_target_trpo_r201.py":"fcec5e8c3f50226a773534781e1960186a1ea4add05d614f604dd16f50eccf46","test_getup_total_target_trpo_r201.py":"5e6ef616965b2fa77cf68d40cae27d76863caa33ace4bfd0f003c0cefc380d29","launch_getup_total_target_trpo_r201.py":"49baf399a57ddc629f6b29ea5fac8c3c0e58f1870bbdf5b231716613e3a40a0b","GETUP_TOTAL_TARGET_TRPO_PLAN_R201_20261011.json":"4469483fe79dd0f3f83913626b93726aa34dc22587bb1f51d2fdb659e2b9b127","evidence_preparation_interface_20261011.json":"4302c22ac679d64bdc5b259484814d0020ef42c72c8acd97d7c33ba98add4bfe"}
paths={n:(project/'diagnostics'/n if n.endswith('.py') else root/'tmp'/('getup_total_target_trpo_r201_interface_preparation_20261011.json' if n.startswith('evidence_') else n)) for n in expected}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(digest(paths[n])==sha for n,sha in expected.items())
for n,p in paths.items():
 if n.endswith('.py'):compile(p.read_text(),str(p),'exec')
available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:6])
assert shutil.disk_usage(root).free>=12*1024**3
assert subprocess.check_output(['git','-C',str(root/'github/openduck-mini-'),'rev-parse','HEAD'],text=True).strip()=='9f3e32e022cd48ad4dfe4f4302e26cc3e359d9db'
assert not subprocess.check_output(['git','-C',str(root/'github/openduck-mini-'),'status','--porcelain'],text=True).strip()
from diagnostics import launch_getup_total_target_trpo_r201 as launcher
launcher.idle()
from diagnostics import train_getup_total_target_trpo_r201 as run
assert not any(p.exists() for p in (run.PREFLIGHT,run.LAUNCH,run.SMOKE,run.OUTPUT))
run.PREFLIGHT.mkdir()
env={**launcher.ENV,'OPEN_DUCK_R201_TEST_CAPTURE':str(run.PREFLIGHT/'executed_sources')}
started=time.monotonic()
with (run.PREFLIGHT/'regression.log').open('x') as stream:
 result=subprocess.run([str(project/'.venv/bin/python'),'-u','-m','diagnostics.test_getup_total_target_trpo_r201','-v'],env=env,stdout=stream,stderr=subprocess.STDOUT)
unchanged=all(digest(paths[n])==sha for n,sha in expected.items())
receipt=dict(natural_exit=True,exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,new_dynamic_attempts=0,source_hashes=expected,source_hashes_unchanged=unchanged,numeric_validator_not_rerun=True,cpu_affinity=sorted(os.sched_getaffinity(0)),simulation_only=True,full_task_completed=False,hardware_readiness=False)
target=root/'tmp/getup_total_target_trpo_r201_preflight_exit_20261011.json'
with target.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print((run.PREFLIGHT/'regression.log').read_text(),flush=True)
print(json.dumps(receipt),flush=True)
assert unchanged
raise SystemExit(result.returncode)
