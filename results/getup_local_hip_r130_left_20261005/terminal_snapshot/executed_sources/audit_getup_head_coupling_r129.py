"""Read-only paired sensor/command audit of R128, no integration or relabeling."""
from pathlib import Path
import json
import shutil
import mujoco
import numpy as np
from diagnostics import probe_getup_head_feedback_r128 as previous
from diagnostics.getup_independent_native import digest

ROOT=previous.ROOT
OUTPUT=ROOT/'outputs/getup_head_coupling_r129_20261005'


def first(values,threshold):
    indices=np.flatnonzero(np.asarray(values)>threshold)
    return None if not len(indices) else float(indices[0]*.02)


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    physics=mujoco.MjModel.from_xml_path(str(previous.previous.program.SCENE))
    names=[physics.actuator(i).name for i in range(physics.nu)]
    head=np.array([names.index(n) for n in previous.HEAD]);legs=np.array([i for i in range(14) if i not in head])
    source=previous.OUTPUT;terminal=json.loads((source/'results.json').read_text());rows=[]
    for case in previous.CASES:
        olddir=source/f'gain_0.0/case_{case}'
        with np.load(olddir/'trajectory.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
        oldrow=json.loads((olddir/'result.json').read_text())
        for gain in (1.,4.):
            folder=source/f'gain_{gain}/case_{case}'
            with np.load(folder/'trajectory.npz',allow_pickle=False) as z:now={k:z[k].copy() for k in z.files}
            row=json.loads((folder/'result.json').read_text());assert oldrow['initial_hash']==row['initial_hash']
            count=min(len(old['time']),len(now['time']))
            command_delta=now['applied'][:count]-old['applied'][:count]
            sensor_delta=now['observations'][:count]-old['observations'][:count]
            leg_command=np.abs(command_delta[:,legs]).max(1);head_command=np.abs(command_delta[:,head]).max(1)
            leg_sensor=np.abs(sensor_delta[:,6+legs]).max(1)
            up_delta=np.linalg.norm(sensor_delta[:,3:6],axis=1)
            extra=np.abs(now['head_correction_rad']).max(1)
            key=f'gain_{gain}_case_{case}'
            np.savez_compressed(OUTPUT/(key+'.npz'),paired_command_delta=command_delta,
                paired_actual_sensor_delta=sensor_delta,original_head_correction_rad=now['head_correction_rad'],
                paired_control_time_s=np.arange(count)*.02)
            record=dict(case_seed=case,gain=gain,paired_initial_hash_equal=True,
                original_success=oldrow['success'],candidate_success=row['success'],
                original_valid=oldrow['valid'],candidate_valid=row['valid'],paired_controls=count,
                first_nonzero_head_correction_s=first(extra,0.),first_head_saturation_s=first(extra,.18-1e-12),
                first_leg_target_difference_s=first(leg_command,1e-8),
                first_leg_sensor_difference_s=first(leg_sensor,1e-4),
                first_up_difference_s=first(up_delta,.01),
                early_0_to_0p9s_max_leg_command_delta_rad=float(leg_command[:45].max()),
                early_0_to_0p9s_max_leg_sensor_delta_rad=float(leg_sensor[:45].max()),
                max_head_command_delta_rad=float(head_command.max()),
                original_final=oldrow['final'],candidate_final=row['final'],
                candidate_controls=row['controls'],candidate_entry_s=row['entry_time_s'],candidate_strict_tail_s=row['strict_tail_s'],
                candidate_peaks=row['peaks'],
                trace_hashes={str(p):digest(p) for p in [olddir/'trajectory.npz',folder/'trajectory.npz']},
                thresholds_diagnostic_only=True,no_saved_label_changes=True)
            rows.append(record)
    report=dict(rows=rows,head_actuator_indices=head.tolist(),leg_actuator_names=[names[i] for i in legs],
        no_dynamic_integration=True,read_only=True,no_force_inference=True,no_success_relabeling=True,
        no_unique_root_cause_claim=True,reserved_qualification_never_loaded=True,
        hashes={str(p):digest(p) for p in [Path(__file__),previous.previous.program.SCENE,source/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R129_TERMINAL',json.dumps([{k:r[k] for k in ('case_seed','gain','first_nonzero_head_correction_s','first_head_saturation_s',
        'first_leg_target_difference_s','first_leg_sensor_difference_s','first_up_difference_s','candidate_valid')} for r in rows]),flush=True)


if __name__=='__main__':main()
