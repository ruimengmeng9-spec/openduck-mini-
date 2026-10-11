"""One new synthetic-only R202 numerical validation, not a dynamic preflight."""
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
import total_target_vmpo_kernel_r202 as kernel
import test_total_target_vmpo_numeric_r202 as tests

ROOT=Path('/data/shijinsheng/open_duck')
HERE=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs/getup_total_target_vmpo_numeric_r202b_20261011'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def capture_sources():
    paths={Path(__file__).resolve(),Path(kernel.__file__).resolve(),Path(tests.__file__).resolve()}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).suffix=='.py' and Path(name).is_file():paths.add(Path(name).resolve())
    dest=OUTPUT/'executed_sources';dest.mkdir()
    manifest={}
    for path in sorted(paths):
        name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name
        sha=digest(path);shutil.copy2(path,dest/name)
        assert digest(dest/name)==sha
        manifest[str(path)]=dict(sha256=sha,copy=name)
    write_json(OUTPUT/'source_manifest.json',manifest)
    return manifest


def main():
    assert not OUTPUT.exists(),'Unique immutable output required'
    assert shutil.disk_usage(ROOT).free>=12*1024**3
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    learner=kernel.Learner(kernel.initial_weights()) # Runtime imported before manifest.
    assert learner.jax.default_backend()=='cpu'
    OUTPUT.mkdir()
    plan=HERE/'GETUP_TOTAL_TARGET_VMPO_PLAN_R202_20261011.json'
    shutil.copy2(plan,OUTPUT/plan.name);plan_sha=digest(plan)
    manifest=capture_sources()
    tests.OUTPUT=OUTPUT
    begin=time.monotonic();result=None
    write_json(OUTPUT/'contract.json',dict(stage='synthetic_numeric_validation_only',
        new_dynamic_attempts=0,simulator_imported=False,old_trajectory_data_loaded=False,
        synthetic_test_only_no_production_checkpoint=True,interface_preflight_passed=False,
        full_smoke_run=False,formal_run=False,source_manifest_not_native_binary_snapshot=True,
        plan_sha256=plan_sha,fixed_numeric_tests=24,full_task_completed=False,hardware_readiness=False))
    try:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.NumericalTests)
        with (OUTPUT/'regression.log').open('w') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        source_unchanged=all(digest(Path(p))==v['sha256'] and
            digest(OUTPUT/'executed_sources'/v['copy'])==v['sha256'] for p,v in manifest.items())
        assert source_unchanged and digest(plan)==plan_sha
        assert 'mujoco' not in sys.modules,'No simulator in numerical-only checks'
        output_bytes=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
        external_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
        external_bytes+=sum(p.stat().st_size for p in (ROOT/'tmp').iterdir()
                            if p.is_file() and 'r202' in p.name.lower())
        live=output_bytes+external_bytes
        # Fixed prospective archive/Git/extra allocation, not only the live output.
        future=640*1024**2+640*1024**2+128*1024**2
        assert live<=640*1024**2 and live+future<=2*1024**3
        assert shutil.disk_usage(ROOT).free>=10*1024**3
        terminal=dict(stage='numeric_validation_closed',tests_run=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
            elapsed_s=time.monotonic()-begin,new_dynamic_attempts=0,
            full_interface_regression_passed=False,full_smoke_run=False,formal_run=False,
            source_hashes_unchanged=source_unchanged,plan_hash_unchanged=True,
            imported_python_source_count=len(manifest),native_binary_snapshot=False,
            compiled_physics_not_checked_in_numeric_stage=True,
            bytes_before_result=live,output_bytes=output_bytes,external_bytes=external_bytes,
            future_archive_git_extra_forecast_bytes=future,
            complete_storage_with_forecast_bytes=live+future,
            free_bytes=shutil.disk_usage(ROOT).free,
            evidence=tests.EVIDENCE,full_task_completed=False,hardware_readiness=False)
        write_json(OUTPUT/'result.json',terminal)
        print(json.dumps(terminal,ensure_ascii=False),flush=True)
        if not result.wasSuccessful():raise SystemExit(1)
    except BaseException:
        write_json(OUTPUT/'failure.json',dict(traceback=traceback.format_exc(),
            tests_run=None if result is None else result.testsRun,new_dynamic_attempts=0,
            original_source_and_failure_preserved=True))
        raise


if __name__=='__main__':main()
