"""Read-only stage decomposition, with explicit native-float32 error bounds.

No MjData, environment constructor, forward/step, force or outcome relabeling.
Saved current sensors and previous executed targets suffice for this algebra.
Stage telescoping is not a counterfactual dynamic recovery experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import probe_getup_fixed_bilateral_r162 as source
from diagnostics.getup_independent_native import digest, DT, SLEW
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS as ORIGINAL_JOINTS
from diagnostics.search_getup_reference_feedback_r64 import residual

run=source.run
local=run.prior.old.local
OUTPUT=run.ROOT/'outputs/getup_response_stages_r164b_20261006'


def gain_matrix(g):
    g=np.asarray(g,dtype=float);m=np.zeros((10,4))
    m[1,[0,2]]=[g[0],g[3]];m[5]=-m[1]
    m[2,[0,2]]=m[6,[0,2]]=[g[1],g[4]]
    m[3,[0,2]]=m[7,[0,2]]=[g[2],g[5]]
    m[0,[1,3]]=[g[6],g[7]];m[4]=-m[0]
    return m


def bounded_apply(target,previous,lower,upper):
    return np.clip(np.clip(target,lower,upper),previous-SLEW*DT,previous+SLEW*DT)


def projection_fraction(values,tangent):
    # Fixed same-phase analytic projection, not nearest-phase matching.
    den=np.sum(tangent*tangent,axis=1)
    projected=np.zeros_like(values)
    moving=den>1e-20
    projected[moving]=tangent[moving]*(np.sum(values[moving]*tangent[moving],axis=1)/den[moving])[:,None]
    energy=float(np.sum(values*values))
    return float(np.sum(projected*projected)/energy) if energy>0 else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert args.output.parent.resolve()==(run.ROOT/'outputs').resolve() and not args.output.exists()
    local.init_worker()
    model=mujoco.MjModel.from_xml_path(str(local.prior.program.SCENE))
    # Model metadata only. Never construct a data object or reset an episode.
    lower,upper=model.jnt_range[model.actuator_trnid[:,0]].T.copy()
    original_ids=np.array([model.actuator(n).id for n in ORIGINAL_JOINTS])
    hip_ids=np.array([model.actuator(n).id for n in run.JOINTS])
    sensor_ids=np.array([0,1,2,9,10,11])
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    np.testing.assert_array_equal([int(np.flatnonzero(qadr==model.joint(n).qposadr)[0]) for n in run.JOINTS],sensor_ids)
    hashes={}
    def tracked(path):
        hashes[str(path)]=digest(path);return path
    def load(folder):
        row=json.loads(tracked(folder/'result.json').read_text())
        with np.load(tracked(folder/'trajectory.npz'),allow_pickle=False) as z:data={k:z[k].copy() for k in ('observations','normalized_residual','applied','local_hip_extra_rad','preparation_sensors','bilateral_extra_rad','bilateral_activation') if k in z.files}
        return data,row
    with np.load(tracked(local.prior.program.REFERENCE),allow_pickle=False) as z:
        targets=z['targets'].copy();phases=z['phases'].copy();gains=z['gains'].copy()
    matrices=np.stack([gain_matrix(g) for g in gains])
    tangent=np.gradient(targets[:529],DT,axis=0)
    for f in [local.prior.program.SCENE,local.prior.OUTPUT/'training/snapshot.npz',Path(__file__)]:tracked(f)
    def stages(folder):
        data,row=load(folder);case=row['case_seed'];n=min(529,len(data['observations']))
        with np.load(tracked(local.prior.audit.OUTPUT/'cases'/f'case_{case}'/'audit_initial_state.npz'),allow_pickle=False) as z:
            previous_initial=z['observed_prev'].copy()  # Only previous target, not root state.
        profile,knots,_=local.prior.scalar_program(local.WEIGHTS,data['preparation_sensors'])
        base_gains=np.array(row.get('base_gains',row.get('global_gains',row.get('gains'))))
        imus=[];actions=[];frozen=[];extra=[];bounds=[];pre=[];original=[];right=[];combined=[];applied=[];without=[]
        for k in range(n):
            obs=data['observations'][k]
            action=local.prior.program.execute_program(local.WEIGHTS,obs,k,profile,knots)
            np.testing.assert_array_equal(action,data['normalized_residual'][k])
            rf=local.local_feedback(obs,local.NOMINAL[k],sensor_ids[3:],base_gains)
            np.testing.assert_array_equal(rf,data['local_hip_extra_rad'][k])
            new=data['bilateral_extra_rad'][k] if 'bilateral_extra_rad' in data else np.zeros(6)
            if 'parameters' in row:
                check,activation=run.bilateral_feedback(row['parameters'],obs,local.NOMINAL[k],data['observations'][0],local.NOMINAL[0],sensor_ids)
                np.testing.assert_array_equal(check,new);np.testing.assert_array_equal(activation,data['bilateral_activation'][k])
            # Original env uses double IMU error; saved obs contains rounded
            # error. Half one float32 spacing bounds round-to-nearest loss.
            err=obs[-5:-1];matrix=matrices[int(phases[k])]
            # Match original expression ordering (matrix only gives the bound).
            imu=residual(err.astype(float),gains[int(phases[k])])
            eps=np.abs(np.spacing(err)).astype(float)/2
            b=np.zeros(model.nu);b[original_ids]=np.abs(matrix)@eps+2e-14
            correction=np.clip(imu+.18*action,-.18,.18)
            target=targets[k].copy();target[original_ids]=np.clip(target[original_ids]+correction,lower[original_ids],upper[original_ids])
            fixed=local.merge_target(target,targets[k],rf,hip_ids[3:],lower,upper)
            final=local.merge_target(fixed,targets[k],new,hip_ids,lower,upper)
            previous=previous_initial if k==0 else data['applied'][k-1]
            calculated=bounded_apply(final,previous,lower,upper)
            actual=data['applied'][k]
            assert np.all(np.abs(calculated-actual)<=b+2e-14),(folder,k,float(np.abs(calculated-actual).max()),float(b.max()))
            imus.append(imu);actions.append(action);frozen.append(rf);extra.append(new);bounds.append(b);pre.append(previous)
            original.append(target);right.append(fixed);combined.append(final);applied.append(calculated);without.append(bounded_apply(fixed,previous,lower,upper))
        return dict(data=data,row=row,n=n,imu=np.array(imus),action=np.array(actions),frozen=np.array(frozen),extra=np.array(extra),
            bound=np.array(bounds),previous=np.array(pre),original=np.array(original),right=np.array(right),combined=np.array(combined),
            calculated=np.array(applied),without_new=np.array(without))
    terminal=json.loads(tracked(source.OUTPUT/'results.json').read_text());pairs=[]
    cases=[None,769002,773002,773011] if args.smoke else [None,*local.prior.program.TRAIN]
    for case in cases:pairs.append(('R162_full',case,source.OUTPUT/'candidate'/f'case_{case}',source.OUTPUT/'baseline'/f'case_{case}'))
    if not args.smoke:
        alternatives=json.loads(tracked(run.ROOT/'outputs/getup_bilateral_coupling_r159_20261006/results.json').read_text())
        for r in alternatives['rows']:
            for alt in r['alternatives']:
                case=r['case_seed'];i=alt['program']
                pairs.append((f'R133_alternative_{i}',case,run.prior.old.source.OUTPUT/f'program_{i:02d}'/f'case_{case}',run.prior.OUTPUT/'candidate'/f'case_{case}'))
        audit=json.loads(tracked(run.ROOT/'outputs/getup_bilateral_terminal_audit_r161_20261006/results.json').read_text())
        best=Path(audit['best_all_valid_nonzero_short']['path'])
        for case in [None,*local.prior.program.TRAIN]:pairs.append(('R160_best_valid_SHORT',case,best/f'case_{case}',run.OUTPUT/'training/generation_0001/candidate_00'/f'case_{case}'))
    args.output.mkdir();shutil.copy2(__file__,args.output/Path(__file__).name);rows=[]
    for label,case,path,reference in pairs:
        x,y=stages(path),stages(reference);assert x['row']['initial_hash']==y['row']['initial_hash']
        np.testing.assert_array_equal(x['data']['observations'][0],y['data']['observations'][0])
        n=min(x['n'],y['n']);actual=x['data']['applied'][:n]-y['data']['applied'][:n]
        direct=x['calculated'][:n]-x['without_new'][:n]
        response=x['without_new'][:n]-y['calculated'][:n]
        np.testing.assert_allclose(direct+response,x['calculated'][:n]-y['calculated'][:n],atol=2e-16,rtol=0)
        bound=x['bound'][:n]+y['bound'][:n]
        assert np.all(np.abs(direct+response-actual)<=bound+4e-14)
        dest=args.output/f'{label}_case_{case}';dest.mkdir()
        np.savez_compressed(dest/'stages.npz',original_target_x=x['original'][:n],original_target_y=y['original'][:n],
            frozen_right_target_x=x['right'][:n],frozen_right_target_y=y['right'][:n],new_target_x=x['combined'][:n],
            actual_applied_difference=actual,same_state_post_slew_direct_new=direct,subsequent_state_and_history_response=response,
            float32_reconstruction_error_bound=bound,original_imu_correction_difference=x['imu'][:n]-y['imu'][:n],
            original_program_action_difference=x['action'][:n]-y['action'][:n],frozen_right_feedback_difference=x['frozen'][:n]-y['frozen'][:n],
            new_extra_rad=x['extra'][:n],native55_difference=x['data']['observations'][:n]-y['data']['observations'][:n])
        windows=[]
        for hi in [min(n,50),n]:
            d=float(np.abs(direct[:hi]).max());r=float(np.abs(response[:hi]).max())
            windows.append(dict(end_control_exclusive=hi,direct_new_max_rad=d,response_max_rad=r,response_to_direct_max_ratio=r/d if d else None,
                imu_delta_max_rad=float(np.abs(x['imu'][:hi]-y['imu'][:hi]).max()),
                program_delta_max_rad=float(.18*np.abs(x['action'][:hi]-y['action'][:hi]).max()),
                frozen_right_delta_max_rad=float(np.abs(x['frozen'][:hi]-y['frozen'][:hi]).max()),
                applied_delta_max_rad=float(np.abs(actual[:hi]).max()),
                reference_tangent_energy_fraction=projection_fraction(actual[:hi],tangent[:hi])))
        row=dict(group=label,case_seed=case,initial_hash=x['row']['initial_hash'],candidate_success=x['row']['success'],baseline_success=y['row']['success'],
            candidate_valid=x['row']['valid'],baseline_valid=y['row']['valid'],original_candidate_peaks=x['row']['peaks'],original_baseline_peaks=y['row']['peaks'],
            windows=windows,all_original_scalar_program_and_local_feedback_exact=True,
            original_IMU_stage_rounded_not_bitwise=True,maximum_reconstruction_error_rad=float(np.abs(x['calculated']-x['data']['applied'][:x['n']]).max()),
            maximum_rounding_bound_rad=float(x['bound'].max()),short_labels_not_full_acceptance=label.endswith('SHORT'))
        local.write_json(dest/'result.json',row);rows.append(row)
        print('R164B_PAIR',label,case,windows[0]['response_to_direct_max_ratio'],windows[0]['reference_tangent_energy_fraction'],flush=True)
    assert all(digest(Path(f))==h for f,h in hashes.items())
    local.write_json(args.output/'source_hashes.json',hashes)
    local.write_json(args.output/'results.json',dict(smoke=args.smoke,read_only=True,no_environment_or_MjData_constructed=True,no_forward_or_dynamic_integration=True,
        no_root_qpos_qvel_read=True,original_labels_peaks_unchanged=True,pairs=len(rows),rows=rows,
        float32_original_IMU_rounding_explicitly_bounded=True,telescoping_stage_algebra_not_dynamic_counterfactual=True,
        fixed_same_phase_tangent_projection_not_clock_search=True,qualification_never_loaded=True,new_controller_trained=False,
        full_task_completed=False,hardware_readiness=False))
    print('R164B_CLOSED',len(rows),max(r['maximum_reconstruction_error_rad'] for r in rows),flush=True)

if __name__=='__main__':main()


