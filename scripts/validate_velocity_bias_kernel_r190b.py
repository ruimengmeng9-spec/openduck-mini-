"""New offline validation directory, immutable R190 kernel and failed evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import mujoco
from diagnostics import test_velocity_bias_kernel_r190b as tests
from diagnostics import velocity_bias_kernel_r190 as kernel

old=tests.old; ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_velocity_bias_kernel_r190b_20261010'
BASE='5626d17edc5a3d45a7057f051575cfc78edd9558'


def main():
    if len(sys.argv)!=1: raise ValueError('No budget or command overrides')
    commands=subprocess.check_output(['ps','-ww','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any((' -m diagnostics.' in c and any(s in c for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_')))
                   or ('/tmp/publish_getup_' in c and '--push' in c) or 'git push' in c for c in commands)
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==BASE
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    assert not OUTPUT.exists() and shutil.disk_usage(ROOT).free>10.25*1024**3
    OUTPUT.mkdir()
    source=tests.capture_all_sources(OUTPUT/'executed_sources')
    assert str(Path(__file__).resolve()) in source
    files=old.model_files(old.local.prior.program.SCENE)
    nominal=old.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz'
    inputs={str(nominal):old.prior.digest(nominal),**files}
    for p in (old.selector.OUTPUT/'training/model.npz',old.local.prior.OUTPUT/'training/snapshot.npz',old.local.prior.program.REFERENCE,old.local.prior.program.STAND):
        inputs[str(p)]=old.prior.digest(p)
    failure=ROOT/'outputs/getup_velocity_bias_kernel_r190_20261010'
    failed_inputs={str(p):old.prior.digest(p) for p in failure.rglob('*') if p.is_file()}
    model=mujoco.MjModel.from_xml_path(str(old.local.prior.program.SCENE)); model.opt.timestep=.002
    compiled=old.model_digest(model)
    contract=dict(offline_only=True,new_dynamic_attempts=0,new_controller_run=False,learned_parameters=0,search_trials=0,
        storage_budget_gb=.25,reserve_gb=10,prospective_independent_full_smoke=6,prospective_formal_attempts=56,
        compiled_model_sha256=compiled,source_manifest=source,input_hashes=inputs,mujoco_version=mujoco.__version__,
        hypothesis='Velocity-dependent model bias torque difference / original affine position-servo kp, ten legs only.',
        head_branch_limitation='Head motion affects head/root reaction, not leg bias at prescribed root motion and zero qacc.',
        actual_inputs='Current/causalinitial native34 gyro,q,v; fixed samephase nominal; no actual root/up/contact/labels.',
        full_task_completed=False,hardware_readiness=False,qualification_run=False)
    old.local.write_json(OUTPUT/'contract.json',contract)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'',
        'JAX_PLATFORMS':'cpu','OPEN_DUCK_R190_PROOF':str(OUTPUT/'algebra_proof.json'),
        'OPEN_DUCK_R190_EXECUTED_SOURCES':str(OUTPUT/'test_executed_sources')}
    code=None; exception=None
    try:
        with (OUTPUT/'regression.log').open('x') as log:
            result=subprocess.run([sys.executable,'-u','-m','diagnostics.test_velocity_bias_kernel_r190b'],env=env,stdout=log,stderr=subprocess.STDOUT)
        code=result.returncode
    except Exception: exception=traceback.format_exc()
    after={p:old.prior.digest(Path(p)) for p in inputs}
    source_after={p:old.prior.digest(Path(p)) for p in source}
    assert inputs==after and all(v['sha256']==source_after[p] for p,v in source.items())
    assert failed_inputs=={p:old.prior.digest(Path(p)) for p in failed_inputs}
    test_manifest_path=OUTPUT/'test_source_manifest.json'
    if test_manifest_path.exists():
        manifest=json.loads(test_manifest_path.read_text())
        assert all(old.prior.digest(Path(p))==v['sha256']==old.prior.digest(OUTPUT/'test_executed_sources'/v['copy']) for p,v in manifest.items())
    assert old.model_digest(model)==compiled
    used=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
    assert used<.25*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    row=dict(terminal_result_saved=True,tests_passed=code==0,regression_exit_code=code,regression_tests=20,
        new_dynamic_attempts=0,new_controller_run=False,full_task_completed=False,hardware_readiness=False,
        qualification_run=False,expanded_development_run=False,source_input_unchanged=True,
        input_hashes_before=inputs,input_hashes_after=after,source_hashes_after=source_after,
        first_failed_evidence_hashes_before=failed_inputs,first_failed_evidence_unchanged=True,
        compiled_model_sha256=compiled,compiled_model_unchanged=True,stored_bytes_before_results=used,
        kernel_sha256=old.prior.digest(Path(kernel.__file__)),main_sha256=old.prior.digest(Path(__file__)),failure=exception)
    old.local.write_json(OUTPUT/'results.json',row)
    print(json.dumps(dict(output=str(OUTPUT),exit_code=code,new_dynamic_attempts=0)),flush=True)
    raise SystemExit(code if code is not None else 1)


if __name__=='__main__': main()
