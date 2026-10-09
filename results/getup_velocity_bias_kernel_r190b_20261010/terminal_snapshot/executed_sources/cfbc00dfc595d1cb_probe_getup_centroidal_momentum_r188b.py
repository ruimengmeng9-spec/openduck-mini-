"""Fixed instantaneous momentum allocation; original full simulation criteria."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from diagnostics import train_getup_full_episode_pg_r181 as old
from diagnostics import centroidal_momentum_kernel_r187b as kernel

prior=old.prior;local=old.local;selector=old.selector;planner=old.planner;ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_centroidal_momentum_r188b_left_20261010'
SMOKE=ROOT/'outputs/getup_centroidal_momentum_r188b_smoke_20261010'
CASES=old.CASES;SEED=288;MODEL=None
KERNEL_SHA='fe458c4d12fa48756197d9b500dda688bd9bc4ce441db0e15f6bea7476646ac6'


def model_digest(model):
    buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def model_files(scene):
    """Resolve the actual scene includes and file-backed assets, never credentials."""
    files=set();roots=[]
    def visit(path):
        path=Path(path).resolve()
        if path in files:return
        assert path.is_file();files.add(path)
        tree=ET.parse(path).getroot();roots.append((path,tree))
        for node in tree.iter('include'):visit(path.parent/node.attrib['file'])
    visit(scene)
    meshdir=assetdir=texturedir=None
    for path,tree in roots:
        for compiler in tree.iter('compiler'):
            for name in ('meshdir','assetdir','texturedir'):
                if name in compiler.attrib:
                    value=(path.parent/compiler.attrib[name]).resolve()
                    if name=='meshdir':meshdir=value
                    elif name=='assetdir':assetdir=value
                    else:texturedir=value
    for path,tree in roots:
        for node in tree.iter():
            if node.tag=='include' or 'file' not in node.attrib:continue
            folder=(meshdir or assetdir or path.parent) if node.tag=='mesh' else (texturedir or assetdir or path.parent)
            asset=(folder/node.attrib['file']).resolve();assert asset.is_file();files.add(asset)
    return {str(path):prior.digest(path) for path in sorted(files)}


def capture_sources(destination):
    destination.mkdir()
    paths={Path(__file__).resolve(),Path(kernel.__file__).resolve()}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).suffix=='.py' and Path(name).is_file():paths.add(Path(name).resolve())
    for name in ('test_getup_centroidal_momentum_r188b.py','launch_getup_centroidal_momentum_r188b.py'):
        paths.add(Path(__file__).with_name(name).resolve())
    manifest={}
    for path in sorted(paths):
        sha=prior.digest(path);name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name
        shutil.copy2(path,destination/name);assert prior.digest(destination/name)==sha
        manifest[str(path)]=dict(sha256=sha,copy=name)
    assert prior.digest(Path(kernel.__file__))==KERNEL_SHA
    return manifest


def record_preparation_partial(sim,prepare,observe,captured):
    """Same R121 calls/order, with available frames retained before reset returns."""
    original=sim.step_target;frames=[];times=[]
    captured['frames']=frames;captured['times']=times
    def recorded(target):
        value=original(target)
        frame=np.asarray(observe(sim),dtype=np.float32)
        if frame.shape!=(50,) or not np.isfinite(frame).all():raise ValueError('Original finite native50 required')
        frames.append(frame.copy());times.append(float(sim.data.time));return value
    sim.step_target=recorded
    try:result=prepare()
    finally:sim.step_target=original
    if len(frames)!=40:raise ValueError('Original 40 home controls required')
    captured.setdefault('completed_batches',[]).append((np.stack(frames),np.asarray(times)))
    return result,np.stack(frames),np.asarray(times)


def init_worker(frozen):
    global MODEL
    path=Path(frozen)
    with np.load(path/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    assert local.WEIGHTS is not None and 'feature_mode' in local.WEIGHTS
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    assert local.NOMINAL.shape==(2279,55) and prior.digest(Path(kernel.__file__))==KERNEL_SHA
    if path.parent.name.startswith('getup_centroidal_momentum_r188b_'):
        manifest=capture_sources(path.parent/f'worker_executed_sources_{os.getpid()}')
        local.write_json(path.parent/f'worker_source_manifest_{os.getpid()}.json',manifest)


def merge_momentum(fixed,reference,request,geometry,lower,upper):
    np.testing.assert_array_equal(request[geometry.head],np.zeros(len(geometry.head)))
    return local.merge_target(fixed,reference,request[geometry.legs],geometry.legs,lower,upper)


def evaluate(job):
    enabled,case,directory,parity=job
    assert case in CASES
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    captured=dict(frames=[],times=[]);records=[];signals={};last_inputs={};env=sim=original=None
    prepare_module=local.prior.audit;old_prepare=prepare_module.record_preparation
    def recorder(s,p,observe=prepare_module.native_observation):return record_preparation_partial(s,p,observe,captured)
    prepare_module.record_preparation=recorder
    try:
        env=prepare_module.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,SEED,True)
        env.reset(case);sim=env.sim;fingerprint=prior.physics_hash(sim);full_fingerprint=model_digest(sim.model)
        geometry=kernel.MomentumGeometry(sim.model)
        ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]);sensors=np.array([9,10,11])
        gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
        profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
        original=sim.step_target;initial=env.observe().copy();maxima=[]
        keys=('momentum_request_rad','momentum_error_kg_m2_per_s','momentum_allocation_kg_m2','momentum_unconstrained_residual_kg_m2_per_s','actual_momentum_kg_m2_per_s','nominal_momentum_kg_m2_per_s','initial_momentum_kg_m2_per_s','nominal_initial_momentum_kg_m2_per_s','local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad')
        signals={k:[] for k in keys}
        def controlled(target):
            k=env.controls;observed=env.observe().copy()
            last_inputs.update(control=np.asarray(k),current_native55=observed.copy(),initial_native55=initial.copy(),original_requested_target=target.copy(),previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy())
            np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
            frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
            fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
            raw,error,allocation,residual=kernel.momentum_request(observed[:34],local.NOMINAL[k,:34],initial[:34],local.NOMINAL[0,:34],geometry,k,enabled)
            hs=[geometry.measure(x)[0] for x in (observed[:34],local.NOMINAL[k,:34],initial[:34],local.NOMINAL[0,:34])] if enabled and k<529 else [np.zeros(3) for _ in range(4)]
            adjusted=merge_momentum(fixed,env.targets[k],raw,geometry,sim.lower,sim.upper)
            np.testing.assert_array_equal(adjusted[geometry.head],fixed[geometry.head])
            base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
            wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
            assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
            if k==0:np.testing.assert_array_equal(raw,np.zeros(14));assert adjusted is fixed
            for key,value in zip(keys,(raw,error,allocation,residual,*hs,frozen,wanted,base,adjusted-fixed,fixed,adjusted)):signals[key].append(np.asarray(value).copy())
            maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
            applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
            return applied
        sim.step_target=controlled
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
        row=step[5]
        arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=env.preparation_sensors,construction_preparation_sensors=captured['completed_batches'][0][0],construction_preparation_times=captured['completed_batches'][0][1],**signals)
        previous_path=selector.OUTPUT/'candidate'/f'case_{case}';previous=json.loads((previous_path/'result.json').read_text());assert row['initial_hash']==previous['initial_hash']
        with np.load(previous_path/'trajectory.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
            if parity or case is None:
                equal={key:bool(np.array_equal(np.asarray(arrays[key]),z[key])) for key in z.files};assert all(equal.values()),equal
                assert row['peaks']==previous['peaks'];row['complete_R157_parity']=equal
        if case is None:
            for key in ('momentum_request_rad','momentum_error_kg_m2_per_s','momentum_unconstrained_residual_kg_m2_per_s'):np.testing.assert_array_equal(signals[key],np.zeros_like(signals[key]))
        row.update(enabled=enabled,base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,maximum_combined_14_joint_correction_rad=max(maxima),maximum_raw_momentum_request_rad=float(np.abs(signals['momentum_request_rad']).max()),maximum_same_state_post_slew_direct_difference_rad=float(np.abs(np.asarray(signals['planned_before_integration_rad'])-np.asarray(signals['same_state_unperturbed_planned_rad'])).max()),physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),compiled_model_sha256=full_fingerprint,compiled_model_unchanged=full_fingerprint==model_digest(sim.model),no_root_truth_or_case_metadata_in_controller=True,root_edits_during_recovery=0)
        assert row['physics_unchanged'] and row['compiled_model_unchanged']
        np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
        return row
    except Exception:
        partial=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=np.asarray(captured['frames']),preparation_times=np.asarray(captured['times']),completed_preparation_sensors=np.asarray([b[0] for b in captured.get('completed_batches',[])]),completed_preparation_times=np.asarray([b[1] for b in captured.get('completed_batches',[])]),**signals)
        np.savez_compressed(dest/'partial_trajectory.npz',**partial);np.savez_compressed(dest/'failed_causal_inputs.npz',**last_inputs)
        local.write_json(dest/'failure.json',dict(execution_error=True,controls_completed=len(records),prepare_frames_saved=len(captured['frames']),terminal_result_saved=False,traceback=traceback.format_exc(),source_sha256=prior.digest(Path(__file__))))
        raise
    finally:
        prepare_module.record_preparation=old_prepare
        if sim is not None and original is not None:sim.step_target=original


def group(pool,enabled,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(enabled,case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('momentum_smoke',[None,769002,773004])]:
        for case in cases:
            fresh=output/name/f'case_{case}';previous=SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(previous/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            x=json.loads((fresh/'result.json').read_text());y=json.loads((previous/'result.json').read_text());assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    output.mkdir();sources=capture_sources(output/'executed_sources');assets=model_files(local.prior.program.SCENE)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    hashes={str(f):prior.digest(f) for f in [*frozen.iterdir(),local.prior.program.STAND,local.prior.program.REFERENCE]}
    local.write_json(output/'contract.json',dict(seed=SEED,simulation_only=True,smoke=args.smoke,controller='Fixed R187b instantaneous centroidal angular momentum virtual velocity allocation, not a dynamics or safety model.',kernel_sha256=KERNEL_SHA,workers=6,formal_dynamic_attempt_budget=56,independent_smoke_attempts=6,storage_budget_gb=2,reserve_gb=10,learned_parameters=0,search_trials=0,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,sources=sources,model_files=assets,hashes=hashes,mujoco_version=mujoco.__version__))
    try:
        with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
            parity=group(pool,False,[None,769000,773004],output/'zero_parity',True)
            momentum=group(pool,True,[None,769002,773004],output/'momentum_smoke')
            if not args.smoke:compare_smoke(output)
            local.write_json(output/'startup_closed.json',dict(parity=parity,momentum=momentum,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
            if args.smoke:
                local.write_json(output/'results.json',dict(smoke=True,parity=parity,momentum=momentum,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
            assert not momentum['physical_failures'] and momentum['nominal_success']
            assert shutil.disk_usage(ROOT).free>10*1024**3
            baseline=group(pool,False,CASES,output/'development_baseline',True)
            candidate=group(pool,True,CASES,output/'development_candidate')
            assert [r['initial_hash'] for r in baseline['rows']]==[r['initial_hash'] for r in candidate['rows']]
            local.write_json(output/'training_closed.json',dict(fixed_controller=True,learned_parameters=0,search_trials=0,formal_dynamic_attempts=56,rng=np.random.default_rng(SEED).bit_generator.state))
            gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
            local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,original_development_gate=gate,terminal_result_saved=True,formal_dynamic_attempts=56,full_task_completed=False,hardware_readiness=False,expanded_development_run=False,independent_qualification_run=False))
            print('R188_MOMENTUM_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)
    finally:
        assert assets==model_files(local.prior.program.SCENE)
        assert all(prior.digest(Path(p))==info['sha256'] for p,info in sources.items())
        assert all(prior.digest(Path(p))==sha for p,sha in hashes.items())
        local.write_json(output/'input_hashes_after.json',dict(source_and_model_files_unchanged=True,hashes=hashes,model_files=assets))


if __name__=='__main__':main()


