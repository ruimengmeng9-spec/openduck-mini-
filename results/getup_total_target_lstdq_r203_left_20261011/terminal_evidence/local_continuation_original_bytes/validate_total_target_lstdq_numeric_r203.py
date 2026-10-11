"""One finite synthetic-only R203 validator, no interface or dynamic claim."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest
import numpy as np
import total_target_lstdq_kernel_r203 as kernel
import test_total_target_lstdq_numeric_r203 as tests

ROOT=Path('/data/shijinsheng/open_duck')
HERE=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs/getup_total_target_lstdq_numeric_r203_20261011'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def capture():
    paths={Path(__file__).resolve(),Path(kernel.__file__).resolve(),Path(tests.__file__).resolve()}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).suffix=='.py' and Path(name).is_file():paths.add(Path(name).resolve())
    dest=OUTPUT/'executed_sources';dest.mkdir();manifest={}
    for p in sorted(paths):
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        sha=digest(p);shutil.copy2(p,dest/name);assert digest(dest/name)==sha
        manifest[str(p)]=dict(sha256=sha,copy=name)
    write_json(OUTPUT/'source_manifest.json',manifest);return manifest
def resource():
    files={p for p in HERE.rglob('*') if p.is_file()}
    for d in (ROOT/'outputs',ROOT/'tmp',ROOT/'projects/Open_Duck_Playground/diagnostics'):
        for p in d.iterdir():
            if 'r203' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(q for q in p.rglob('*') if q.is_file())
    live=sum(p.stat().st_size for p in files)
    future=(640+640+128)*1024**2
    assert live<=640*1024**2 and live+future<=2*1024**3
    free=shutil.disk_usage(ROOT).free;assert free>=10*1024**3
    return dict(live_bytes=live,future_archive_git_extra_bytes=future,
        complete_storage_with_forecast_bytes=live+future,free_bytes=free,
        all_R203_outputs_external_tmp_logs_sources_counted=True,no_cleanup=True)
def main():
    assert not OUTPUT.exists();assert shutil.disk_usage(ROOT).free>=12*1024**3
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    resource();OUTPUT.mkdir()
    plan=HERE/'GETUP_TOTAL_TARGET_LSTDQ_PLAN_R203_20261011.json'
    shutil.copy2(plan,OUTPUT/plan.name);psha=digest(plan)
    manifest=capture();tests.OUTPUT=OUTPUT;result=None;begin=time.monotonic()
    write_json(OUTPUT/'contract.json',dict(stage='synthetic_numeric_only',fixed_tests=24,
        new_dynamic_attempts=0,old_data_loaded=False,simulator_imported=False,
        no_production_checkpoint_from_synthetic=True,interface_preflight_passed=False,
        smoke_run=False,formal_run=False,plan_sha256=psha,numpy_version=np.__version__,
        imported_python_not_native_binary_snapshot=True,full_task_completed=False,hardware_readiness=False))
    try:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.NumericalTests)
        assert suite.countTestCases()==24
        with (OUTPUT/'regression.log').open('x') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        unchanged=all(digest(Path(p))==v['sha256'] and digest(OUTPUT/'executed_sources'/v['copy'])==v['sha256'] for p,v in manifest.items())
        assert unchanged and digest(plan)==psha
        assert 'mujoco' not in sys.modules
        passed=result.wasSuccessful() and result.testsRun==24 and not result.skipped
        terminal=dict(stage='numeric_validation_closed',tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
            passed=passed,elapsed_s=time.monotonic()-begin,new_dynamic_attempts=0,
            source_hashes_unchanged=True,plan_hash_unchanged=True,
            imported_python_source_count=len(manifest),native_binary_snapshot=False,
            interface_preflight_passed=False,smoke_run=False,formal_run=False,
            actual_physics_not_checked=True,evidence=tests.EVIDENCE,
            resource=resource(),full_task_completed=False,hardware_readiness=False)
        write_json(OUTPUT/'result.json',terminal);print(json.dumps(terminal),flush=True)
        if not passed:raise SystemExit(1)
    except BaseException:
        write_json(OUTPUT/'failure.json',dict(traceback=traceback.format_exc(),
            tests_run=None if result is None else result.testsRun,new_dynamic_attempts=0))
        raise
if __name__=='__main__':main()

