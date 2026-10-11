import json,os,shutil,subprocess
from pathlib import Path
root=Path('/data/shijinsheng/open_duck');project=root/'projects/Open_Duck_Playground'
from diagnostics import launch_getup_total_target_trpo_r201 as l
l.idle()
assert shutil.disk_usage(root).free>=12*1024**3
assert not any(p.exists() for p in (l.OUTPUT,l.run.SMOKE,l.run.OUTPUT))
assert json.loads((root/'tmp/getup_total_target_trpo_r201_preflight_exit_20261011.json').read_text())['exit_code']==0
assert subprocess.check_output(['git','-C',str(root/'github/openduck-mini-'),'rev-parse','HEAD'],text=True).strip()==l.run.EXPECTED_HEAD
assert not subprocess.check_output(['git','-C',str(root/'github/openduck-mini-'),'status','--porcelain'],text=True).strip()
l.run.storage_gate()
result=subprocess.run([str(project/'.venv/bin/python'),'-u','-m','diagnostics.launch_getup_total_target_trpo_r201','--launch'],env=l.ENV,capture_output=True,text=True)
print(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,launch_request_exit_not_launcher_natural_exit=True,new_dynamic_budget_cap=162)))
raise SystemExit(result.returncode)
