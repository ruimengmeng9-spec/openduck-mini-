"""Causal sensor-to-planned-target joint-path FK diagnostic, no dynamics."""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import audit_getup_control_contact_r144 as contact
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics.getup_independent_native import digest

ROOT=contact.ROOT
SOURCE=contact.OUTPUT
OUTPUT=ROOT/'outputs/getup_planned_contact_r145_20261006'
FRACTIONS=(0.,.25,.5,.75,1.)


def sensor_target_path(fk,sensor,target):
    sensor=np.asarray(sensor);target=np.asarray(target)
    if sensor.shape!=(55,) or target.shape!=(14,) or not np.isfinite(sensor).all() or not np.isfinite(target).all():
        raise ValueError('Finite actual sensor55 and already-planned target14 only')
    start=fk.pose(sensor)
    result=[]
    for fraction in FRACTIONS:
        q=start.copy();q[fk.qadr]=start[fk.qadr]+fraction*(target-start[fk.qadr])
        peak,rows=contact.contacts(fk,q)
        result.append(dict(fraction=fraction,contact_peaks_m=peak.tolist(),contact_set=rows))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    fk=contact.geometry.SensorGeometry(mujoco.MjModel.from_xml_path(str(contact.geometry.SCENE)))
    x=np.zeros(55,np.float32);y=x.copy();y[:6]=3.;y[20:]=7.
    target=fk.home.copy()
    assert sensor_target_path(fk,x,target)==sensor_target_path(fk,y,target)
    assert sensor_target_path(fk,x,target)==sensor_target_path(fk,x,target)
    try:sensor_target_path(fk,np.zeros(56),target)
    except ValueError:pass
    else:raise AssertionError('Case metadata accepted')
    # Verify recorded applied targets are planned before all ten integrations.
    source_text=Path(local.__file__).with_name('getup_independent_native.py').read_text()
    # The executing step_target lives in RecoverySim, not in this offline audit.
    import inspect
    from diagnostics.validate_getup_fullpath_r27 import StrictSim
    method=inspect.getsource(StrictSim.step_target)
    assert method.index('self.prev')<method.index('mj_step')
    args.output.mkdir(exist_ok=False)
    for source in (Path(__file__),Path(contact.__file__)):
        shutil.copy2(source,args.output/source.name)
    (args.output/'executed_step_target.txt').write_text(method)
    manifest=json.loads((SOURCE/'results.json').read_text());rows=manifest['rows']
    if args.smoke:rows=[rows[0],rows[1],next(r for r in rows if r['label']=='standard')]
    reports=[];source_hashes={}
    for row in rows:
        name=row['label']+'_p'+str(row['program'])+'_'+str(row['case_seed'])
        saved=SOURCE/name/'result.json';before=digest(saved);data=json.loads(saved.read_text())
        frames=[]
        with np.load(contact.SOURCE/name/'trajectory.npz',allow_pickle=False) as z:
            obs=z['observations'].copy();applied=z['applied'].copy()
        for frame in data['frames']:
            k=frame['control'];np.testing.assert_array_equal(frame['executed_target_rad'],applied[k])
            # A saved target must equal the previous target plus the original
            # fixed slew clip. Truth future states never enter this feature.
            previous_target=fk.home+obs[k,34:48].astype(float)
            assert np.max(np.abs(applied[k]-previous_target))<=5.24*.02+1e-7
            path=sensor_target_path(fk,obs[k],applied[k])
            frames.append(dict(control=k,path=path,envelope_contact_peaks_m=np.max([r['contact_peaks_m'] for r in path],axis=0).tolist(),
                original_substep_peak_m=frame['original_substep_peak_m']))
        assert digest(saved)==before;source_hashes[str(saved)]=before
        terminal=None if not data['first_crossing'] else data['first_crossing']['original']['control_index']
        original_pair=None if not data['first_crossing'] else data['first_crossing']['original']['geom_ids']
        # Original 4 mm is used only to compare diagnostics with original
        # physics labels; no new acceptance or policy threshold is selected.
        warning=[f['control'] for f in frames if f['envelope_contact_peaks_m'][2]>=.004]
        result=dict(label=row['label'],program=row['program'],case_seed=row['case_seed'],initial_hash=row['initial_hash'],
            original_valid=row['original_valid'],original_success=row['original_success'],
            terminal_control=terminal,original_pair=original_pair,
            sampled_warning_controls=warning,warning_at_or_before_terminal=bool(terminal is not None and any(k<=terminal for k in warning)),
            frames=frames)
        dest=args.output/name;dest.mkdir(exist_ok=False);(dest/'result.json').write_text(json.dumps(result,indent=2))
        reports.append({k:v for k,v in result.items() if k!='frames'})
        print('R145_PLANNED_CONTACT',name,warning,flush=True)
    summary=dict(smoke=args.smoke,rows=reports,fractions=FRACTIONS,regression_checks_passed=True,
        invalid_with_sampled_warning=sum(not r['original_valid'] and r['warning_at_or_before_terminal'] for r in reports),
        valid_with_sampled_warning=sum(r['original_valid'] and bool(r['sampled_warning_controls']) for r in reports),
        original_4mm_comparison_not_a_new_threshold=True,windows_not_whole_path_coverage=True,
        inference_input_only_actual_joint_sensor_and_preintegration_planned_target=True,
        no_future_true_state_inputs=True,no_dynamic_integration=True,no_contact_force_inference=True,
        no_training_no_policy_change_no_acceptance_relabel=True,qualification_never_loaded=True,
        full_task_completed=False,hardware_readiness=False,
        hashes={**source_hashes,str(Path(__file__)):digest(Path(__file__)),str(contact.geometry.SCENE):digest(contact.geometry.SCENE)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2))
    print('R145_TERMINAL',summary['invalid_with_sampled_warning'],summary['valid_with_sampled_warning'],flush=True)


if __name__=='__main__':main()
