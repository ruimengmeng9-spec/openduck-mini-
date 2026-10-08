"""Read-only scalar and terminal-field audit of closed R168, no simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_expert_mixture_r168 as run

OUTPUT=run.ROOT/'outputs/getup_expert_mixture_terminal_audit_r169_20261008'
SMOKE=run.ROOT/'outputs/getup_expert_mixture_terminal_audit_r169_smoke_20261008'
FIELDS=['observations','applied','local_hip_extra_rad','selected_base_feedback_rad','expert_feedback_rad',
        'mixture_gates','mixture_weights','mixture_delta_features','feedback_difference_from_base_rad']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def label(a,b):return 'rescued' if a['success'] and not b['success'] else 'regressed' if b['success'] and not a['success'] else 'retained' if a['success'] else 'both_failed'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert not args.output.exists() and args.output.parent==run.ROOT/'outputs'
    source=run.OUTPUT;terminal=read(source/'results.json');closed=read(source/'training_closed.json')
    assert terminal['terminal_result_saved'] and not terminal['smoke'] and len(closed['history'])==4
    assert not terminal['expanded_development_run'] and not terminal['independent_qualification_run'] and not np.any(terminal['parameters'])
    paths=sorted({s for h in closed['history'] for s in h['closed_trial_directories']})
    reports={s:read(Path(s)/'results.json') for s in paths}
    zero=[s for s in paths if not np.any(reports[s]['rows'][0]['parameters'])];assert len(zero)==1
    base={r['case_seed']:r for r in reports[zero[0]]['rows']}
    nonzero=[s for s in paths if s!=zero[0]]
    best_any=max(nonzero,key=lambda s:(reports[s]['successes'],-reports[s]['physical_failures'],reports[s]['return_sum']))
    valid=[s for s in nonzero if not reports[s]['physical_failures']]
    best_valid=max(valid,key=lambda s:(reports[s]['successes'],reports[s]['return_sum'])) if valid else None
    source_hashes={}
    def remember(f):
        f=Path(f);source_hashes[str(f)]=digest(f);return f
    for f in [source/'results.json',source/'training_closed.json',source/'contract.json',Path(__file__),source/'executed_sources'/Path(run.__file__).name]:remember(f)
    assert digest(run.__file__)==digest(source/'executed_sources'/Path(run.__file__).name)
    contract=read(source/'contract.json');assert all(digest(f)==h for f,h in contract['hashes'].items())
    with np.load(remember(source/'frozen/nominal_sensor_trajectory.npz'),allow_pickle=False) as z:nominal=z['observations'].copy()
    with np.load(remember(source/'frozen/frozen_programs.npz'),allow_pickle=False) as z:experts=z['gains'].copy()
    args.output.mkdir();shutil.copy2(__file__,args.output/Path(__file__).name)
    programs=[];selected_paths=[zero[0],best_valid or best_any] if args.smoke else paths
    for s in selected_paths:
        report=reports[s];remember(Path(s)/'results.json');rows=[]
        for row in report['rows']:
            case=row['case_seed']
            if args.smoke and case not in [None,769002,773004]:continue
            b=base[case];assert row['initial_hash']==b['initial_hash']
            f=remember(Path(s)/f'case_{case}/trajectory.npz');bf=remember(Path(zero[0])/f'case_{case}/trajectory.npz')
            remember(Path(s)/f'case_{case}/result.json')
            with np.load(f,allow_pickle=False) as z:a={k:z[k].copy() for k in FIELDS}
            with np.load(bf,allow_pickle=False) as z:baseline={k:z[k].copy() for k in ['observations','applied']}
            count=len(a['observations']);initial=a['observations'][0];expected=[[] for _ in range(6)]
            for k,obs in enumerate(a['observations']):
                if k<529:values=run.mixture_feedback(row['parameters'],obs,nominal[k],initial,nominal[0],row['base_gains'],experts)
                else:values=(np.zeros(3),np.zeros(3),np.zeros((4,3)),np.zeros(4),np.r_[1.,np.zeros(4)],np.zeros(12))
                for v,store in zip(values,expected):store.append(v)
            for k,field in enumerate(['local_hip_extra_rad','selected_base_feedback_rad','expert_feedback_rad','mixture_gates','mixture_weights','mixture_delta_features']):
                np.testing.assert_array_equal(expected[k],a[field])
            np.testing.assert_array_equal(a['local_hip_extra_rad']-a['selected_base_feedback_rad'],a['feedback_difference_from_base_rad'])
            np.testing.assert_array_equal(a['mixture_weights'][0],[1,0,0,0,0]);np.testing.assert_array_equal(a['feedback_difference_from_base_rad'][0],np.zeros(3))
            np.testing.assert_array_equal(a['applied'][0],baseline['applied'][0])
            if case is None:
                for name in ['mixture_gates','mixture_delta_features','expert_feedback_rad','local_hip_extra_rad']:np.testing.assert_array_equal(a[name],np.zeros_like(a[name]))
            n=min(count,len(baseline['observations']));target_diff=a['applied'][:n]-baseline['applied'][:n]
            sensor_diff=a['observations'][:n]-baseline['observations'][:n]
            first=lambda v:next((int(i) for i in np.flatnonzero(np.max(np.abs(v),axis=1)>1e-8)),None)
            r=dict(case_seed=case,initial_hash=row['initial_hash'],label=label(row,b),success=row['success'],valid=row['valid'],controls=row['controls'],peaks=row['peaks'],
                first_target_difference_control=first(target_diff),first_sensor_difference_control=first(sensor_diff),
                max_gate=float(a['mixture_gates'].max()),max_expert_total_weight=float(a['mixture_weights'][:,1:].sum(1).max()),
                max_same_state_raw_feedback_difference_rad=float(np.abs(a['feedback_difference_from_base_rad']).max()),
                early_raw_feedback_difference_rad=float(np.abs(a['feedback_difference_from_base_rad'][:min(n,50)]).max()),
                early_executed_target_difference_rad=float(np.abs(target_diff[:min(n,50)]).max()),
                early_gyro_up_difference=float(np.abs(sensor_diff[:min(n,50),:6]).max()),
                maximum_combined_14_joint_correction_rad=row['maximum_combined_14_joint_correction_rad'],scalar_feedback_exact=True,
                raw_three_hip_difference_not_post_limit_executed_fourteen_joint_difference=True)
            rows.append(r)
            if s in {best_any,best_valid}:
                dest=args.output/'representatives'/Path(s).relative_to(source/'training')/f'case_{case}';dest.mkdir(parents=True)
                np.savez_compressed(dest/'paired_signals.npz',**a,executed_target_difference_rad=target_diff,
                    sensor_difference=sensor_diff,baseline_observations=baseline['observations'][:n],baseline_applied=baseline['applied'][:n])
        programs.append(dict(directory=s,parameters=report['rows'][0]['parameters'],successes=report['successes'],physical_failures=report['physical_failures'],
            rescued=[r['case_seed'] for r in rows if r['label']=='rescued'],regressed=[r['case_seed'] for r in rows if r['label']=='regressed'],rows=rows))
    parity=[]
    for a,b in zip(terminal['candidate']['rows'],terminal['baseline']['rows']):
        case=a['case_seed'];assert case==b['case_seed'] and a['initial_hash']==b['initial_hash'] and a['peaks']==b['peaks']
        files=[source/'development_candidate'/f'case_{case}/trajectory.npz',source/'development_baseline'/f'case_{case}/trajectory.npz',run.prior.previous.prior.OUTPUT/'candidate'/f'case_{case}/trajectory.npz']
        priorrow=read(remember(files[2].with_name('result.json')));assert b['initial_hash']==priorrow['initial_hash'] and b['peaks']==priorrow['peaks']
        for f in files:remember(f)
        with np.load(files[0],allow_pickle=False) as x,np.load(files[1],allow_pickle=False) as y,np.load(files[2],allow_pickle=False) as z:
            equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files};prior={k:bool(np.array_equal(y[k],z[k])) for k in z.files}
        assert all(equal.values()) and all(prior.values());parity.append(dict(case_seed=case,all_candidate_baseline_fields_exact=equal,baseline_R157_fields_exact=prior))
    if not args.smoke:
        smoke=read(SMOKE/'results.json');assert smoke['smoke']
        for sp in smoke['programs']:
            fp=next(p for p in programs if p['directory']==sp['directory'])
            assert sp['rows']==[r for r in fp['rows'] if r['case_seed'] in [None,769002,773004]]
        for f in (SMOKE/'representatives').rglob('paired_signals.npz'):
            other=args.output/f.relative_to(SMOKE)
            with np.load(f,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
                assert a.files==b.files
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    assert all(digest(f)==h for f,h in source_hashes.items()),'Source changed during read-only audit'
    summary=dict(read_only=True,no_MjData_environment_forward_or_integration=True,no_outcome_or_validity_relabeling=True,
        smoke=args.smoke,distinct_actual_programs=len(paths),nonzero_programs=len(nonzero),actual_training_attempts=25*len(paths),
        scalar_trajectories_audited=sum(len(p['rows']) for p in programs),generations=4,selected_parameters=terminal['parameters'],
        candidate_successes=terminal['candidate']['successes'],baseline_successes=terminal['baseline']['successes'],candidate_physical_failures=terminal['candidate']['physical_failures'],
        development_gate=terminal['original_development_gate'],best_nonzero=best_any,best_all_valid_nonzero=best_valid,all_valid_nonzero_programs=len(valid),
        programs=programs,terminal_parity=parity,raw_feedback_and_executed_differences_separate_not_dynamic_counterfactual=True,
        qpos_qvel_only_read_for_separate_original_field_parity_not_feedback_inputs=True,source_hashes_unchanged=True,
        independent_smoke_bitwise_equal=not args.smoke,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False)
    write(args.output/'results.json',summary);write(args.output/'source_hashes.json',source_hashes)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['programs','terminal_parity','selected_parameters']}),flush=True)
    for p in programs:print('R169_PROGRAM',Path(p['directory']).relative_to(source),p['successes'],p['physical_failures'],p['rescued'],p['regressed'],flush=True)

if __name__=='__main__':main()
