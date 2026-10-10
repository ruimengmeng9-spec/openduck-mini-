"""Finite shared global-horizon teacher search across all original full fallen starts."""
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
from diagnostics import shared_horizon_teacher_kernel_r193 as kernel

prior=old.prior;local=old.local;selector=old.selector;planner=old.planner;ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_shared_horizon_teacher_r193_left_20261010'
SMOKE=ROOT/'outputs/getup_shared_horizon_teacher_r193_smoke_20261010'
CASES=old.CASES;SEED=293;MODEL=None
KERNEL_SHA="3edcc26f9fb4d36b4d645c7d5834289e77f9f66e8cdf681f28f5c37990f2afde"


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
    for name in ('test_getup_shared_horizon_teacher_r193.py','launch_getup_shared_horizon_teacher_r193.py'):
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
    if path.parent.name.startswith('getup_shared_horizon_teacher_r193_'):
        manifest=capture_sources(path.parent/f'worker_executed_sources_{os.getpid()}')
        local.write_json(path.parent/f'worker_source_manifest_{os.getpid()}.json',manifest)


def merge_teacher(fixed,reference,request,geometry,lower,upper):
    np.testing.assert_array_equal(request[geometry.head],np.zeros(len(geometry.head)))
    return local.merge_target(fixed,reference,request[geometry.legs],geometry.legs,lower,upper)


def evaluate(job):
    coefficients,case,directory,parity=job
    coefficients=kernel.validate_coefficients(coefficients)
    enabled=bool(np.any(coefficients))
    assert case in CASES
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    captured=dict(frames=[],times=[]);records=[];signals={};last_inputs={};env=sim=original=None
    prepare_module=local.prior.audit;old_prepare=prepare_module.record_preparation
    def recorder(s,p,observe=prepare_module.native_observation):return record_preparation_partial(s,p,observe,captured)
    prepare_module.record_preparation=recorder
    try:
        env=prepare_module.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,SEED,True)
        env.reset(case);sim=env.sim;fingerprint=prior.physics_hash(sim);full_fingerprint=model_digest(sim.model)
        geometry=kernel.Mapping(sim.model)
        ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]);sensors=np.array([9,10,11])
        gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
        profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
        original=sim.step_target;initial=env.observe().copy();maxima=[]
        keys=('teacher_request_rad','teacher_causal_delta','teacher_waveform_rad','teacher_activation','local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad')
        signals={k:[] for k in keys}
        def controlled(target):
            k=env.controls;observed=env.observe().copy()
            last_inputs.update(control=np.asarray(k),current_native55=observed.copy(),initial_native55=initial.copy(),original_requested_target=target.copy(),previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy())
            np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
            frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
            fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
            raw,error,wave,activation=kernel.teacher_request(observed[:34],local.NOMINAL[k,:34],initial[:34],local.NOMINAL[0,:34],coefficients,geometry,k)
            adjusted=merge_teacher(fixed,env.targets[k],raw,geometry,sim.lower,sim.upper)
            np.testing.assert_array_equal(adjusted[geometry.head],fixed[geometry.head])
            base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
            wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
            assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
            if k==0:np.testing.assert_array_equal(raw,np.zeros(14));assert adjusted is fixed
            for key,value in zip(keys,(raw,error,wave,activation,frozen,wanted,base,adjusted-fixed,fixed,adjusted)):signals[key].append(np.asarray(value).copy())
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
            for key in ('teacher_request_rad','teacher_causal_delta'):np.testing.assert_array_equal(signals[key],np.zeros_like(signals[key]))
        row.update(enabled=enabled,teacher_coefficients_rad=coefficients.tolist(),base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,maximum_combined_14_joint_correction_rad=max(maxima),maximum_raw_teacher_request_rad=float(np.abs(signals['teacher_request_rad']).max()),maximum_same_state_post_slew_direct_difference_rad=float(np.abs(np.asarray(signals['planned_before_integration_rad'])-np.asarray(signals['same_state_unperturbed_planned_rad'])).max()),physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),compiled_model_sha256=full_fingerprint,compiled_model_unchanged=full_fingerprint==model_digest(sim.model),no_root_truth_or_case_metadata_in_controller=True,root_edits_during_recovery=0)
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


def group(pool,coefficients,cases,path,parity=False):
    assert shutil.disk_usage(ROOT).free>10*1024**3
    # Reserve two additional copies for terminal snapshots and Git increment.
    paths=(OUTPUT,SMOKE,ROOT/'outputs/getup_shared_horizon_teacher_launcher_r193_20261010')
    live_bytes=sum(p.stat().st_size for d in paths if d.exists() for p in d.rglob('*') if p.is_file())
    assert 3*live_bytes<2*1024**3,'New experiment storage budget exhausted; no cleanup or extra attempts'
    report=local.aggregate(list(pool.map(evaluate,[(np.asarray(coefficients).tolist(),case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('teacher_smoke',[None,769002,773004])]:
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
    local.write_json(output/'contract.json',dict(seed=SEED,simulation_only=True,smoke=args.smoke,controller='One shared four-global-harmonic ten-leg teacher, causal RMS activation; complete-case retention-constrained offline selection, no safety guarantee.',kernel_sha256=KERNEL_SHA,workers=6,formal_dynamic_attempt_budget=181,independent_smoke_attempts=6,storage_budget_gb=2,reserve_gb=10,teacher_coefficients=40,nonzero_search_trials=4,training_full_attempts=125,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,sources=sources,model_files=assets,hashes=hashes,mujoco_version=mujoco.__version__))
    try:
        with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
            proposals,rng=kernel.make_proposals(SEED)
            local.write_json(output/'proposal_rng.json',dict(seed=SEED,rng=rng,coefficients=[p.tolist() for p in proposals]))
            parity=group(pool,proposals[0],[None,769000,773004],output/'zero_parity',True)
            teacher=group(pool,proposals[1],[None,769002,773004],output/'teacher_smoke')
            if not args.smoke:compare_smoke(output)
            local.write_json(output/'startup_closed.json',dict(parity=parity,teacher=teacher,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
            if args.smoke:
                local.write_json(output/'results.json',dict(smoke=True,parity=parity,teacher=teacher,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
            assert not teacher['physical_failures'] and teacher['nominal_success']
            assert shutil.disk_usage(ROOT).free>10*1024**3
            reports=[]
            for i,c in enumerate(proposals):
                report=group(pool,c,CASES,output/'teacher_training'/f'program_{i:02d}',i==0)
                if i:assert [r['initial_hash'] for r in report['rows']]==[r['initial_hash'] for r in reports[0]['rows']]
                reports.append(report)
                selected,evidence=kernel.choose_teacher(reports)
                local.write_json(output/f'closed_program_{i:02d}.json',dict(program=i,coefficients=c.tolist(),reports=reports,selected=selected,evidence=evidence,rng=rng,full_training_attempts=25*(i+1)))
                print('R193_CLOSED_TEACHER',i,report['successes'],report['physical_failures'],selected,flush=True)
            selected,evidence=kernel.choose_teacher(reports)
            np.savez_compressed(output/'selected_teacher.npz',coefficients=proposals[selected],basis=kernel.BASIS)
            baseline=group(pool,proposals[0],CASES,output/'development_baseline',True)
            candidate=group(pool,proposals[selected],CASES,output/'development_candidate',selected==0)
            assert [r['initial_hash'] for r in baseline['rows']]==[r['initial_hash'] for r in candidate['rows']]
            if selected==0:
                for case in CASES:
                    with np.load(output/'development_baseline'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as a,np.load(output/'development_candidate'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as b:
                        assert a.files==b.files
                        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            local.write_json(output/'training_closed.json',dict(shared_teacher=True,nonzero_search_trials=4,selected_program=selected,reports=reports,retention=evidence,formal_dynamic_attempts=181,rng=rng))
            final_retention=kernel.retention(candidate,baseline)
            gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'] and final_retention['eligible'])
            local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,selected_program=selected,retention=evidence,final_retention=final_retention,original_development_gate=gate,terminal_result_saved=True,formal_dynamic_attempts=181,full_task_completed=False,hardware_readiness=False,expanded_development_run=False,independent_qualification_run=False))
            print('R193_SHARED_TEACHER_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)

    finally:
        assert assets==model_files(local.prior.program.SCENE)
        assert all(prior.digest(Path(p))==info['sha256'] for p,info in sources.items())
        assert all(prior.digest(Path(p))==sha for p,sha in hashes.items())
        local.write_json(output/'input_hashes_after.json',dict(source_and_model_files_unchanged=True,hashes=hashes,model_files=assets))


if __name__=='__main__':main()

