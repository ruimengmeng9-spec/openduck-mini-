"""Same-visited-state expert directions, offline labels never enter algebra.

No environment, MjData, forward/integration or outcome relabeling. Four
legal fixed-program outputs on an off-program state are not recovery proof.
"""
import argparse
import inspect
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import audit_getup_velocity_terminal_r166 as audit
from diagnostics import audit_getup_response_stages_r164b as stages
from diagnostics import probe_getup_complementary_feedback_r133 as experts
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.search_getup_reference_feedback_r64 import residual

run=audit.run
OUTPUT=run.ROOT/'outputs/getup_expert_directions_r167_20261008'
SMOKE=run.ROOT/'outputs/getup_expert_directions_r167_smoke_20261008'

def family(current,nominal,base_gains,global_gains):
    """Every expert, without case, success labels, phase or future readings."""
    ids=np.array([9,10,11])
    base=run.local.local_feedback(current,nominal,ids,base_gains)
    all_fixed=np.stack([run.local.local_feedback(current,nominal,ids,g) for g in global_gains])
    return base,all_fixed,all_fixed-base

def cosine(x,y):
    den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.sum(x*y)/den) if den>1e-20 else None

def regress(gains):
    assert list(inspect.signature(family).parameters)==['current','nominal','base_gains','global_gains']
    n=np.zeros(55,dtype=np.float32)
    np.testing.assert_array_equal(family(n,n,gains[0],gains)[2],np.zeros((4,3)))
    x=n.copy();x[15:18]=[.01,-.02,.03];x[29:32]=[.003,-.002,.001]
    b,f,d=family(x,n,gains[0],gains)
    np.testing.assert_array_equal(family(x,n,gains[0],gains[::-1])[1],f[::-1])
    x2=x.copy();x2[:6]=1;x2[34:]=2
    np.testing.assert_array_equal(family(x2,n,gains[0],gains)[1],f)
    assert np.isfinite(f).all() and np.abs(f).max()<=.18
    np.testing.assert_array_equal(d,f-b)
    return 6

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert not args.output.exists() and args.output.parent==run.ROOT/'outputs'
    hashes={}
    def tracked(f):
        f=Path(f);hashes[str(f)]=audit.digest(f);return f
    ar=audit.read(tracked(audit.OUTPUT/'results.json'))
    best=Path(ar['best_all_valid_nonzero']);assert any(r['directory']==str(best) and r['successes']==8 and not r['physical_failures'] for r in ar['programs'])
    with np.load(tracked(experts.OUTPUT/'frozen_programs.npz'),allow_pickle=False) as z:gains=z['gains'].copy()
    tests=regress(gains)
    with np.load(tracked(run.OUTPUT/'frozen/nominal_sensor_trajectory.npz'),allow_pickle=False) as z:nominal=z['observations'].copy()
    with np.load(tracked(run.local.prior.program.REFERENCE),allow_pickle=False) as z:targets=z['targets'].copy();phases=z['phases'].copy();imu_gains=z['gains'].copy()
    model=mujoco.MjModel.from_xml_path(str(tracked(run.local.prior.program.SCENE)))
    lower,upper=model.jnt_range[model.actuator_trnid[:,0]].T.copy()
    original_ids=np.array([model.actuator(n).id for n in JOINTS]);hip=np.array([model.actuator(n).id for n in run.local.JOINTS])
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    np.testing.assert_array_equal([int(np.flatnonzero(qadr==model.joint(n).qposadr)[0]) for n in run.local.JOINTS],[9,10,11])
    matrices=np.stack([stages.gain_matrix(g) for g in imu_gains])
    for f in [Path(__file__),Path(audit.__file__),Path(stages.__file__),Path(run.local.__file__)]:tracked(f)
    args.output.mkdir();shutil.copy2(__file__,args.output/Path(__file__).name);rows=[]
    cases=[None,769004,773014] if args.smoke else [None,*run.local.prior.program.TRAIN]
    for case in cases:
        folder=best/f'case_{case}';row=audit.read(tracked(folder/'result.json'))
        basefolder=run.previous.prior.OUTPUT/'candidate'/f'case_{case}';baseline=audit.read(tracked(basefolder/'result.json'))
        with np.load(tracked(folder/'trajectory.npz'),allow_pickle=False) as z:
            a={k:z[k].copy() for k in ['observations','normalized_residual','applied','local_hip_extra_rad','damping_extra_rad','actual_velocity_error_rad_s','same_state_post_slew_direct_new_rad']}
        with np.load(tracked(basefolder/'trajectory.npz'),allow_pickle=False) as z:baseobs=z['observations'].copy();baseapplied=z['applied'].copy()
        assert row['initial_hash']==baseline['initial_hash'];np.testing.assert_array_equal(a['observations'][0],baseobs[0])
        with np.load(tracked(run.local.prior.audit.OUTPUT/'cases'/f'case_{case}/audit_initial_state.npz'),allow_pickle=False) as z:prev0=z['observed_prev'].copy()
        labels=[];otherobs=[];otherapplied=[]
        for i in range(4):
            f=experts.OUTPUT/f'program_{i:02d}'/f'case_{case}';r=audit.read(tracked(f/'result.json'))
            assert r['initial_hash']==row['initial_hash']
            with np.load(tracked(f/'trajectory.npz'),allow_pickle=False) as z:ob=z['observations'].copy();ap=z['applied'].copy()
            np.testing.assert_array_equal(ob[0],a['observations'][0]);np.testing.assert_array_equal(r['gains'],gains[i])
            labels.append(dict(program=i,success=r['success'],valid=r['valid'],peaks=r['peaks']))
            otherobs.append(ob);otherapplied.append(ap)
        n=min(529,len(a['observations']));fixed=[];deltas=[];post=[];bounds=[];actual=[]
        for k,obs in enumerate(a['observations'][:n]):
            b,f,d=family(obs,nominal[k],row['base_gains'],gains)
            np.testing.assert_array_equal(b,a['local_hip_extra_rad'][k])
            damp,err=run.damping_feedback(row['parameters'],obs,nominal[k],row['groups'])
            np.testing.assert_array_equal(damp,a['damping_extra_rad'][k]);np.testing.assert_array_equal(err,a['actual_velocity_error_rad_s'][k])
            imu=residual(obs[-5:-1].astype(float),imu_gains[int(phases[k])])
            bound=np.zeros(14);bound[original_ids]=np.abs(matrices[int(phases[k])])@(np.abs(np.spacing(obs[-5:-1])).astype(float)/2)+2e-14
            original=targets[k].copy();original[original_ids]=np.clip(original[original_ids]+np.clip(imu+.18*a['normalized_residual'][k],-.18,.18),lower[original_ids],upper[original_ids])
            prev=prev0 if k==0 else a['applied'][k-1]
            freeze=run.local.merge_target(original,targets[k],b,hip,lower,upper)
            without=run.apply_limits(freeze,prev,lower,upper)
            adjusted=run.merge_damping(freeze,targets[k],damp,err,lower,upper)
            observed=run.apply_limits(adjusted,prev,lower,upper)
            assert np.all(np.abs(observed-a['applied'][k])<=bound+2e-14)
            assert np.all(np.abs(observed-without-a['same_state_post_slew_direct_new_rad'][k])<=2*bound+4e-14)
            counter=np.stack([run.apply_limits(run.local.merge_target(original,targets[k],v,hip,lower,upper),prev,lower,upper)-without for v in f])
            if case is None:np.testing.assert_array_equal(counter,np.zeros((4,14)))
            fixed.append(f);deltas.append(d);post.append(counter);bounds.append(bound);actual.append(observed)
        fixed=np.array(fixed);deltas=np.array(deltas);post=np.array(post);bounds=np.array(bounds)
        dest=args.output/f'case_{case}';dest.mkdir()
        arrays=dict(current_native55=a['observations'][:n],nominal_native55=nominal[:n],fixed_expert_feedback=fixed,raw_expert_difference_from_selected=deltas,
            same_state_post_slew_expert_difference=post,actual_damping_direct=a['same_state_post_slew_direct_new_rad'][:n],
            float32_imu_reconstruction_bound=bounds,actual_damping_applied=a['applied'][:n],baseline_applied=baseapplied[:n],baseline_native55=baseobs[:n])
        for i in range(4):
            arrays[f'expert_{i}_own_rollout_native55']=otherobs[i][:n];arrays[f'expert_{i}_own_rollout_applied']=otherapplied[i][:n]
        np.savez_compressed(dest/'directions.npz',**arrays)
        comparisons=[]
        for i in range(4):
            windows=[]
            for hi in [1,min(50,n),n]:
                same=post[:hi,i];direct=a['same_state_post_slew_direct_new_rad'][:hi];raw=deltas[:hi,i];new=a['damping_extra_rad'][:hi,hip]
                windows.append(dict(end_control_exclusive=hi,post_slew_cosine=cosine(same,direct),raw_right_cosine=cosine(raw,new),
                    expert_displacement_max_rad=float(np.abs(same).max()),damping_direct_max_rad=float(np.abs(direct).max()),
                    fraction_frames_positive_dot=float(np.mean(np.sum(same*direct,axis=1)>1e-20)),
                    own_rollout_vs_visited_sensor_max=float(np.abs(otherobs[i][:hi]-a['observations'][:hi]).max())))
            comparisons.append(dict(**labels[i],windows=windows))
        item=dict(case_seed=case,initial_hash=row['initial_hash'],label=audit.label(row,baseline),candidate_success=row['success'],candidate_valid=row['valid'],baseline_success=baseline['success'],
            original_candidate_peaks=row['peaks'],original_baseline_peaks=baseline['peaks'],comparisons=comparisons,
            max_applied_reconstruction_error_rad=float(np.abs(np.array(actual)-a['applied'][:n]).max()),max_rounding_bound_rad=float(bounds.max()),original_scalar_feedback_exact=True)
        audit.write(dest/'result.json',item);rows.append(item)
        if not args.smoke and (SMOKE/f'case_{case}/directions.npz').exists():
            with np.load(SMOKE/f'case_{case}/directions.npz',allow_pickle=False) as z:
                for key,val in arrays.items():np.testing.assert_array_equal(z[key],val)
        print('R167_CASE',case,item['label'],[(c['program'],c['windows'][1]['post_slew_cosine']) for c in comparisons if c['success']],flush=True)
    assert all(audit.digest(f)==h for f,h in hashes.items())
    audit.write(args.output/'source_hashes.json',hashes)
    audit.write(args.output/'results.json',dict(smoke=args.smoke,read_only=True,regression_assertions=tests,rows=rows,cases=len(rows),same_state_expert_outputs_per_case=4,
        no_environment_MjData_forward_or_dynamic_integration=True,no_root_qpos_qvel_read=True,initial_audit_only_previous_target=True,
        no_success_label_in_family_function=True,labels_offline_not_case_oracle=True,no_outcome_or_validity_relabeling=True,source_hashes_unchanged=True,
        rounded_IMU_reconstruction_not_bitwise=True,off_program_state_expert_output_not_recovery_proof=True,qualification_never_loaded=True,new_controller_trained=False,full_task_completed=False,hardware_readiness=False))
    print('R167_CLOSED',len(rows),flush=True)

if __name__=='__main__':main()
