"""Unique R205 finite synthetic CACLA validator: no robot, physics or old data."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import unittest
import numpy as np
import total_target_cacla_kernel_r205 as kernel
import test_total_target_cacla_numeric_r205 as tests
ROOT=Path('/data/shijinsheng/open_duck')
HERE=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs/getup_total_target_cacla_numeric_r205_20261011'
WRAPPER=ROOT/'tmp/numeric_wrapper_r205_20261011.py'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def capture(existing=None):
    manifest={} if existing is None else dict(existing)
    paths={Path(__file__).resolve(),Path(kernel.__file__).resolve(),Path(tests.__file__).resolve(),WRAPPER}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).suffix=='.py' and Path(name).is_file():paths.add(Path(name).resolve())
    dest=OUTPUT/'executed_sources'
    if existing is None:dest.mkdir()
    for p in sorted(paths):
        if str(p) in manifest:
            assert digest(p)==manifest[str(p)]['sha256']==digest(dest/manifest[str(p)]['copy'])
            continue
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        sha=digest(p);assert not (dest/name).exists();shutil.copy2(p,dest/name);assert digest(dest/name)==sha
        manifest[str(p)]=dict(sha256=sha,copy=name)
    return manifest
def resource():
    files={p for p in HERE.rglob('*') if p.is_file()}
    for d in (ROOT/'outputs',ROOT/'tmp',ROOT/'projects/Open_Duck_Playground/diagnostics'):
        for p in d.iterdir():
            if 'r205' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(q for q in p.rglob('*') if q.is_file())
    live=sum(p.stat().st_size for p in files);future=(640+640+128)*1024**2
    assert live<=640*1024**2 and live+future<=2*1024**3
    free=shutil.disk_usage(ROOT).free;assert free>=10*1024**3
    return dict(live_bytes=live,future_archive_git_extra_bytes=future,total_forecast_bytes=live+future,free_bytes=free,
        all_R205_outputs_external_tmp_logs_sources_counted=True,no_cleanup=True)
def main():
    assert not OUTPUT.exists() and shutil.disk_usage(ROOT).free>=12*1024**3
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    resource();OUTPUT.mkdir()
    plan=HERE/'GETUP_TOTAL_TARGET_CACLA_PLAN_R205_20261011.json';psha=digest(plan);shutil.copy2(plan,OUTPUT/plan.name)
    initialized=kernel.Learner()
    assert initialized.actor_count==initialized.value_count==initialized.round_count==0
    initialized.checkpoint(OUTPUT/'synthetic_initial')
    manifest=capture();startup_count=len(manifest);write_json(OUTPUT/'startup_source_manifest.json',manifest)
    tests.OUTPUT=OUTPUT;begin=time.monotonic();result=None
    write_json(OUTPUT/'contract.json',dict(stage='synthetic_numeric_only',fixed_tests=24,new_dynamic_attempts=0,
        old_data_loaded=False,simulator_imported=False,no_production_checkpoint_from_synthetic=True,
        interface_preflight_passed=False,smoke_run=False,formal_run=False,plan_sha256=psha,numpy_version=np.__version__,
        startup_and_additional_runtime_imported_python_captured=True,native_binary_snapshot=False,
        synthetic_initializer_actor_value_count0=True,no_Q_target_Adam=True,full_task_completed=False,hardware_readiness=False))
    try:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.NumericalTests);assert suite.countTestCases()==24
        with (OUTPUT/'regression.log').open('x',encoding='utf-8') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        manifest=capture(manifest);write_json(OUTPUT/'source_manifest.json',manifest)
        unchanged=all(digest(Path(p))==v['sha256']==digest(OUTPUT/'executed_sources'/v['copy']) for p,v in manifest.items())
        assert unchanged and digest(plan)==psha and 'mujoco' not in sys.modules and 'jax' not in sys.modules
        passed=result.wasSuccessful() and result.testsRun==24 and not result.skipped
        terminal=dict(stage='synthetic_numeric_validation_closed',tests_run=result.testsRun,failures=len(result.failures),
            errors=len(result.errors),skipped=len(result.skipped),passed=passed,elapsed_s=time.monotonic()-begin,
            new_dynamic_attempts=0,source_hashes_unchanged=True,plan_hash_unchanged=True,
            startup_imported_python_count=startup_count,complete_imported_python_source_count=len(manifest),
            additional_runtime_python_count=len(manifest)-startup_count,native_binary_snapshot=False,
            interface_preflight_passed=False,smoke_run=False,formal_run=False,production_initialization_not_claimed=True,
            actual_physics_not_checked=True,evidence=tests.EVIDENCE,resource=resource(),
            full_task_completed=False,hardware_readiness=False)
        write_json(OUTPUT/'result.json',terminal);print(json.dumps(terminal),flush=True)
        if not passed:raise SystemExit(1)
    except BaseException:
        write_json(OUTPUT/'failure.json',dict(traceback=traceback.format_exc(),
            tests_run=None if result is None else result.testsRun,new_dynamic_attempts=0))
        raise
if __name__=='__main__':main()

