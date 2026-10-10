"""Finite online sensor identification with gated predictive control."""
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
from diagnostics import adaptive_sensor_mpc_kernel_r195 as kernel

prior=old.prior;local=old.local;selector=old.selector;planner=old.planner;ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_adaptive_sensor_mpc_r195_left_20261010'
SMOKE=ROOT/'outputs/getup_adaptive_sensor_mpc_r195_smoke_20261010'
CASES=old.CASES;SEED=295;MODEL=None
KERNEL_SHA="549eddb9f3dada37aae7b1223d9231aad795b06179b40bc2201343540eb79685"


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
    for name in ('test_getup_adaptive_sensor_mpc_r195.py','launch_getup_adaptive_sensor_mpc_r195.py'):
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
    if path.parent.name.startswith('getup_adaptive_sensor_mpc_r195_'):
        manifest=capture_sources(path.parent/f'worker_executed_sources_{os.getpid()}')
        local.write_json(path.parent/f'worker_source_manifest_{os.getpid()}.json',manifest)


def merge_adaptive(fixed,reference,request,geometry,lower,upper):
    np.testing.assert_array_equal(request[geometry.head],np.zeros(len(geometry.head)))
    return local.merge_target(fixed,reference,request[geometry.legs],geometry.legs,lower,upper)


def evaluate(job):
    mode,case,directory,parity=job
    controller=kernel.Controller(mode)
    enabled=mode!='zero'
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
        keys=('adaptive_request_rad','adaptive_causal_delta','adaptive_previous_change','adaptive_update_error','adaptive_transition_input','adaptive_one_step_prediction','local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad')
        signals={k:[] for k in keys}
        def controlled(target):
            k=env.controls;observed=env.observe().copy()
            last_inputs.update(control=np.asarray(k),current_native55=observed.copy(),initial_native55=initial.copy(),original_requested_target=target.copy(),previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy())
            np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
            frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
            fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
            leg_request=controller.request(observed[:34],local.NOMINAL[k,:34],initial[:34],local.NOMINAL[0,:34],k)
            raw=np.zeros(14);raw[geometry.legs]=leg_request
            error=controller.x.copy();change=controller.change.copy()
            if k==kernel.CALIBRATION:np.savez_compressed(dest/'probe_model.npz',theta=controller.model.theta,p=controller.model.p,count=controller.model.count)
            adjusted=merge_adaptive(fixed,env.targets[k],raw,geometry,sim.lower,sim.upper)
            np.testing.assert_array_equal(adjusted[geometry.head],fixed[geometry.head])
            base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
            wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
            assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
            if k==0:np.testing.assert_array_equal(raw,np.zeros(14));assert adjusted is fixed
            maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
            applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
            controller.commit((applied-base)[geometry.legs])
            features=controller.model.pending.copy() if controller.active else np.zeros(78)
            prediction=error+controller.model.theta@features if controller.active else np.zeros(34)
            for key,value in zip(keys,(raw,error,change,controller.model.error.copy() if controller.active else np.zeros(34),features,prediction,frozen,wanted,base,adjusted-fixed,fixed,adjusted)):signals[key].append(np.asarray(value).copy())
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
            for key in ('adaptive_request_rad','adaptive_causal_delta'):np.testing.assert_array_equal(signals[key],np.zeros_like(signals[key]))
        row.update(enabled=enabled,adaptive_mode=mode,adaptive_updates=controller.model.count,adaptive_fallback_controls=controller.fallbacks,base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,maximum_combined_14_joint_correction_rad=max(maxima),maximum_raw_adaptive_request_rad=float(np.abs(signals['adaptive_request_rad']).max()),maximum_same_state_post_slew_direct_difference_rad=float(np.abs(np.asarray(signals['planned_before_integration_rad'])-np.asarray(signals['same_state_unperturbed_planned_rad'])).max()),physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),compiled_model_sha256=full_fingerprint,compiled_model_unchanged=full_fingerprint==model_digest(sim.model),no_root_truth_or_case_metadata_in_controller=True,root_edits_during_recovery=0)
        assert row['physics_unchanged'] and row['compiled_model_unchanged']
        np.savez_compressed(dest/'trajectory.npz',**arrays)
        np.savez_compressed(dest/'online_terminal.npz',theta=controller.model.theta,p=controller.model.p,previous=np.zeros(34) if controller.model.previous is None else controller.model.previous,previous_change=controller.model.previous_change,pending=np.zeros(78) if controller.model.pending is None else controller.model.pending)
        local.write_json(dest/'online_rng.json',controller.rng);local.write_json(dest/'result.json',row)
        return row
    except Exception:
        partial=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=np.asarray(captured['frames']),preparation_times=np.asarray(captured['times']),completed_preparation_sensors=np.asarray([b[0] for b in captured.get('completed_batches',[])]),completed_preparation_times=np.asarray([b[1] for b in captured.get('completed_batches',[])]),**signals)
        np.savez_compressed(dest/'partial_trajectory.npz',**partial);np.savez_compressed(dest/'failed_causal_inputs.npz',**last_inputs)
        local.write_json(dest/'failure.json',dict(execution_error=True,controls_completed=len(records),prepare_frames_saved=len(captured['frames']),terminal_result_saved=False,traceback=traceback.format_exc(),source_sha256=prior.digest(Path(__file__))))
        raise
    finally:
        prepare_module.record_preparation=old_prepare
        if sim is not None and original is not None:sim.step_target=original


def group(pool,mode,cases,path,parity=False):
    assert shutil.disk_usage(ROOT).free>10*1024**3
    # Reserve two additional copies for terminal snapshots and Git increment.
    paths=(OUTPUT,SMOKE,ROOT/'outputs/getup_adaptive_sensor_mpc_launcher_r195_20261010')
    live_bytes=sum(p.stat().st_size for d in paths if d.exists() for p in d.rglob('*') if p.is_file())
    assert 3*live_bytes<2*1024**3,'New experiment storage budget exhausted; no cleanup or extra attempts'
    report=local.aggregate(list(pool.map(evaluate,[(mode,case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


SMOKE_GROUPS=[('zero_parity','zero',[None,769000,773004],True),
              ('calibration_standard','positive',[None],False),
              ('positive_smoke','positive',[769002,773004],False),
              ('negative_smoke','negative',[769002,773004],False)]

def compare_smoke(output,include_zero=False):
    for name,mode,cases,parity in SMOKE_GROUPS:
        if name=='zero_parity' and not include_zero:continue
        for case in cases:
            fresh=output/('development_baseline' if name=='zero_parity' else name)/f'case_{case}';previous=SMOKE/name/f'case_{case}'
            for filename in ('trajectory.npz','online_terminal.npz'):
                with np.load(fresh/filename,allow_pickle=False) as a,np.load(previous/filename,allow_pickle=False) as b:
                    assert a.files==b.files
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            assert json.loads((fresh/'result.json').read_text())==json.loads((previous/'result.json').read_text())

def independent_probe_model(observations,applied,base,nominal,legs):
    # Saved features and weights are comparison-only, not predictors.
    theta=np.zeros((34,78));p=100*np.eye(78);previous=None;pending=None
    initial=observations[0,:34].astype(float)-nominal[0,:34].astype(float)
    for k in range(65):
        x=((observations[k,:34].astype(float)-nominal[k,:34].astype(float))-initial)/kernel.SCALE
        if k==0:x[:]=0
        change=np.zeros(34) if previous is None else x-previous
        if pending is not None:
            pf=p@pending;den=kernel.FORGET+pending@pf
            residual=change-theta@pending
            theta+=np.outer(residual,pf/den)
            p=(p-np.outer(pf,pf)/den)/kernel.FORGET;p=(p+p.T)*.5
        if k==64:return theta,p
        pending=np.r_[x,change,(applied[k]-base[k])[legs]/kernel.BOUND]
        previous=x.copy()

def causal_gate(positive,negative,cases,nominal,legs):
    rows=[];actual=[];predicted=[];valid=True
    for case in cases:
        pa=positive/f'case_{case}';pb=negative/f'case_{case}'
        ra=json.loads((pa/'result.json').read_text());rb=json.loads((pb/'result.json').read_text())
        complete=bool(ra['valid'] and rb['valid'] and ra['controls']==2279 and rb['controls']==2279 and not ra['adaptive_fallback_controls'] and not rb['adaptive_fallback_controls'])
        valid=valid and complete
        if ra['controls']<66 or rb['controls']<66:
            rows.append(dict(case=case,complete_valid=False,probe_reached=False));continue
        with np.load(pa/'trajectory.npz',allow_pickle=False) as a,np.load(pb/'trajectory.npz',allow_pickle=False) as b:
            for key in ('observations','applied','normalized_residual','time','strict'):
                end=65 if key=='observations' else 64
                np.testing.assert_array_equal(a[key][:end],b[key][:end],err_msg='Common prefix '+key)
            # Root truth only isolated equality, not RLS or prediction inputs.
            for key in ('qpos','qvel'):np.testing.assert_array_equal(a[key][:64],b[key][:64])
            np.testing.assert_array_equal(a['preparation_sensors'],b['preparation_sensors'])
            np.testing.assert_array_equal(a['same_state_unperturbed_planned_rad'][64],b['same_state_unperturbed_planned_rad'][64])
            ta,pp=independent_probe_model(a['observations'],a['applied'],a['same_state_unperturbed_planned_rad'],nominal,legs)
            for path in (pa,pb):
                with np.load(path/'probe_model.npz',allow_pickle=False) as model:
                    np.testing.assert_array_equal(ta,model['theta']);np.testing.assert_array_equal(pp,model['p'])
            difference=(a['applied'][64]-b['applied'][64])[legs]
            truth=(a['observations'][65,:34].astype(float)-b['observations'][65,:34].astype(float))/kernel.SCALE
            estimate=ta[:,68:]@(difference/kernel.BOUND)
            nonzero=bool(np.any(difference));valid=valid and nonzero
            actual.append(truth);predicted.append(estimate)
            rows.append(dict(case=case,complete_valid=complete,probe_reached=True,nonzero_executed_probe=nonzero,initial_hash=ra['initial_hash'],positive_original_peaks=ra['peaks'],negative_original_peaks=rb['peaks']))
            assert ra['initial_hash']==rb['initial_hash']
            np.savez_compressed(pa/'held_probe_effect.npz',actual_scaled=truth,predicted_scaled=estimate,executed_difference_rad=difference)
    metrics={}
    if actual:
        truth=np.asarray(actual);estimate=np.asarray(predicted)
        for name,sl in [('gyro',slice(0,3)),('framezaxis',slice(3,6)),('q',slice(6,20)),('native_velocity',slice(20,34))]:
            zero=float(np.mean(truth[:,sl]**2));error=float(np.mean((estimate[:,sl]-truth[:,sl])**2))
            metrics[name]=dict(zero_effect_mse=zero,model_effect_mse=error,passes=bool(zero>0 and error<.8*zero))
    passed=bool(valid and len(actual)==len(cases) and len(metrics)==4 and all(v['passes'] for v in metrics.values()))
    return dict(passed=passed,rows=rows,metrics=metrics,all_complete_valid=valid,pairs=len(actual),no_future_observation_in_prediction=True,no_safety_or_general_identifiability_claim=True)

def smoke_report(pool,output,smoke=True):
    reports={}
    for name,mode,cases,parity in (SMOKE_GROUPS if smoke else SMOKE_GROUPS[1:]):reports[name]=group(pool,mode,cases,output/name,parity)
    model=mujoco.MjModel.from_xml_path(str(local.prior.program.SCENE));mapping=kernel.Mapping(model)
    with np.load(output/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations']
    effect=causal_gate(output/'positive_smoke',output/'negative_smoke',[769002,773004],nominal,mapping.legs)
    local.write_json(output/'causal_smoke_gate.json',effect)
    return reports,effect

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    output.mkdir();sources=capture_sources(output/'executed_sources');assets=model_files(local.prior.program.SCENE)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    hashes={str(f):prior.digest(f) for f in [*frozen.iterdir(),local.prior.program.STAND,local.prior.program.REFERENCE]}
    local.write_json(output/'contract.json',dict(seed=SEED,simulation_only=True,smoke=args.smoke,kernel_sha256=KERNEL_SHA,workers=6,identification_formal_attempt_cap=80,independent_identification_smoke_attempt_cap=8,remaining_candidate_attempt_cap=28,total_cap=116,storage_budget_gib=2,reserve_gib=10,prediction_steps=3,model_feature_dimensions=78,old_R172_R173_model_loaded=False,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,sources=sources,model_files=assets,hashes=hashes,mujoco_version=mujoco.__version__))
    try:
        with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
            reports,effect=smoke_report(pool,output,args.smoke)
            if not args.smoke:compare_smoke(output)
            startup=dict(smoke=args.smoke,reports=reports,causal_effect=effect,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False,actual_dynamic_attempts=8 if args.smoke else 5,prediction_controller_run=False)
            local.write_json(output/'startup_closed.json',startup)
            if args.smoke:local.write_json(output/'results.json',startup);return
            assert effect['passed'],'Independent model gate required before formal'
            assert all(r['nominal_success'] and not r['physical_failures'] for r in reports.values())
            baseline=group(pool,'zero',CASES,output/'development_baseline',True)
            compare_smoke(output,include_zero=True)
            positive=group(pool,'positive',CASES,output/'identification_positive')
            negative=group(pool,'negative',CASES,output/'identification_negative')
            assert [r['initial_hash'] for r in positive['rows']]==[r['initial_hash'] for r in negative['rows']]==[r['initial_hash'] for r in baseline['rows']]
            model=mujoco.MjModel.from_xml_path(str(local.prior.program.SCENE));mapping=kernel.Mapping(model)
            with np.load(frozen/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations']
            full_effect=causal_gate(output/'identification_positive',output/'identification_negative',[c for c in CASES if c is not None],nominal,mapping.legs)
            local.write_json(output/'formal_causal_gate.json',full_effect)
            gate=bool(full_effect['passed'] and all(r['nominal_success'] and not r['physical_failures'] for r in (positive,negative)) and not any(row['adaptive_fallback_controls'] for report in (positive,negative) for row in report['rows']))
            if not gate:
                local.write_json(output/'results.json',dict(terminal_result_saved=True,actual_dynamic_attempts=80,baseline=baseline,positive=positive,negative=negative,causal_effect=full_effect,prediction_controller_run=False,original_development_gate=False,full_task_completed=False,hardware_readiness=False));return
            # Candidate is a separate process stage with its own natural smoke exit.
            # Remaining fixed allocation: three independent candidate paths plus 25 formal.
            local.write_json(output/'results.json',dict(terminal_result_saved=True,actual_dynamic_attempts=80,baseline=baseline,positive=positive,negative=negative,causal_effect=full_effect,identification_gate_passed=True,awaiting_separate_candidate_smoke=True,remaining_candidate_attempt_cap=28,prediction_controller_run=False,original_development_gate=False,full_task_completed=False,hardware_readiness=False,expanded_development_run=False,independent_qualification_run=False))
    finally:
        assert assets==model_files(local.prior.program.SCENE)
        assert all(prior.digest(Path(p))==info['sha256'] for p,info in sources.items())
        assert all(prior.digest(Path(p))==sha for p,sha in hashes.items())
        local.write_json(output/'input_hashes_after.json',dict(source_and_model_files_unchanged=True,hashes=hashes,model_files=assets))

if __name__=='__main__':main()
