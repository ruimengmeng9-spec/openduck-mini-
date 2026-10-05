"""Read-only early-state comparison of R134 failures and R133 alternatives."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_success_selector_r134 as selector
from diagnostics.getup_independent_native import digest

ROOT=selector.ROOT
OUTPUT=ROOT/'outputs/getup_dynamic_feedback_audit_r146_20261006'


def first_over(values,threshold):
    ids=np.flatnonzero(values>threshold)
    return None if not len(ids) else int(ids[0])


def main():
    assert not OUTPUT.exists()
    result=json.loads((selector.OUTPUT/'results.json').read_text())
    rows=result['reports']['candidate']['rows']
    assert sum(r['success'] and r['case_seed'] is not None for r in rows)==17
    with np.load(selector.source.OUTPUT/'frozen_programs.npz',allow_pickle=False) as z:gains=z['gains'].copy()
    OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    reports=[];hashes={}
    for row in rows:
        case=row['case_seed'];chosen=row['global_program_choice']
        a=selector.OUTPUT/'candidate'/f'case_{case}'
        with np.load(a/'trajectory.npz',allow_pickle=False) as z:observed={k:z[k].copy() for k in z.files}
        for program in range(4):
            folder=selector.source.OUTPUT/f'program_{program:02d}'/f'case_{case}'
            alternative=json.loads((folder/'result.json').read_text())
            assert alternative['initial_hash']==row['initial_hash']
            with np.load(folder/'trajectory.npz',allow_pickle=False) as z:other={k:z[k].copy() for k in z.files}
            if program==chosen:
                assert all(np.array_equal(observed[k],other[k]) for k in observed)
            if case is None:
                assert all(np.array_equal(observed[k],other[k]) for k in ('observations','qpos','qvel','applied','strict'))
            if row['success'] or not alternative['success']:continue
            count=min(529,len(observed['observations']),len(other['observations']))
            obs=observed['observations'][:count];alt=other['observations'][:count]
            np.testing.assert_array_equal(obs[0],alt[0])
            targetdiff=np.abs(observed['applied'][:count]-other['applied'][:count]).max(1)
            sensordiff=np.abs(obs[:,:34].astype(float)-alt[:,:34].astype(float)).max(1)
            # Current sensor differences are available causally; comparison
            # to another successful run is offline evidence, not policy input.
            report=dict(case_seed=case,chosen_program=chosen,successful_alternative=program,
                initial_hash=row['initial_hash'],first_target_difference_control=first_over(targetdiff,1e-8),
                first_sensor_difference_control=first_over(sensordiff,1e-8),
                first_sensor_difference_over_001_control=first_over(sensordiff,.01),
                chosen_valid=row['valid'],alternative_valid=alternative['valid'],
                diagnostic_thresholds_not_acceptance=True)
            dest=OUTPUT/f'case_{case}_alternative_{program}';dest.mkdir()
            np.savez_compressed(dest/'paired_causal_sensor_differences.npz',target_difference_rad=targetdiff,
                native_sensor_difference=sensordiff,chosen_native34=obs[:,:34],alternative_native34=alt[:,:34])
            (dest/'result.json').write_text(json.dumps(report,indent=2));reports.append(report)
            for p in (a/'trajectory.npz',folder/'trajectory.npz'):hashes[str(p)]=digest(p)
    failed=[r['case_seed'] for r in rows if not r['success']]
    assert len(failed)==7 and {r['case_seed'] for r in reports}==set(failed)
    summary=dict(rows=reports,failed_cases=failed,selected_program_execution_exact=True,
        all_programs_standard_motion_exact=True,all_paired_initial_sensors_exact=True,
        no_dynamics_no_label_changes=True,no_case_or_alternative_success_input_authorized=True,
        proposed_hypothesis='Keep the R134 initial action and test subsequent bounded gain adaptation from actual state change; this audit does not show switching or adaptation will succeed',
        qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={**hashes,str(Path(__file__)):digest(Path(__file__))})
    (OUTPUT/'results.json').write_text(json.dumps(summary,indent=2))
    print('R146_TERMINAL',json.dumps(dict(rows=reports,failed_cases=failed)),flush=True)


if __name__=='__main__':main()
