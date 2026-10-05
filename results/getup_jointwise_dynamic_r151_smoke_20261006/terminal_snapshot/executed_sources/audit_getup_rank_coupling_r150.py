"""Read-only R149 coupling audit; no simulation steps or outcome relabeling."""
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import probe_getup_fixed_dynamic_r149 as source
from diagnostics import train_getup_dynamic_gain_r147 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_rank_coupling_r150_20261006'
WINDOWS=((0,50),(50,150),(150,300),(300,529))

def first(values,threshold):
    ids=np.flatnonzero(values>threshold)
    return int(ids[0]) if len(ids) else None

def decompose(obs,nominal,ids,parameters):
    initial=run.state_features(obs[0],nominal[0],ids)
    delta=np.stack([run.state_features(x,n,ids)-initial for x,n in zip(obs,nominal)])
    contributions=np.tanh(delta)*parameters[6:]
    # Deliberately reconstruct actual scalar execution, not batch matmul.
    activation=np.array([float(np.tanh(parameters[6:]@np.tanh(d))) if np.any(d) else 0. for d in delta])
    return delta,contributions,activation

def main():
    assert not OUTPUT.exists()
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
    ids=np.array([int(np.flatnonzero(qadr==model.joint(n).qposadr)[0]) for n in run.local.JOINTS])
    aid=np.array([model.actuator(n).id for n in run.local.JOINTS])
    run.local.init_worker()
    terminal=json.loads((source.OUTPUT/'results.json').read_text())
    assert terminal['candidate']['successes']==4 and terminal['baseline']['successes']==17
    OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    hashes={};rows=[];early_by_label={};profiles=[]
    def load(path):
        hashes[str(path/'trajectory.npz')]=digest(path/'trajectory.npz')
        hashes[str(path/'result.json')]=digest(path/'result.json')
        with np.load(path/'trajectory.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        return arrays,json.loads((path/'result.json').read_text())
    for a,b in zip(terminal['candidate']['rows'],terminal['baseline']['rows']):
        case=a['case_seed'];assert case==b['case_seed'] and a['initial_hash']==b['initial_hash']
        x,_=load(source.OUTPUT/'candidate'/f'case_{case}')
        y,_=load(source.OUTPUT/'baseline'/f'case_{case}')
        old,_=load(run.OUTPUT/'development_baseline'/f'case_{case}')
        r134,_=load(run.selector.OUTPUT/'candidate'/f'case_{case}')
        assert all(np.array_equal(y[k],old[k]) for k in y)
        assert all(np.array_equal(y[k],r134[k]) for k in r134)
        n=min(529,len(x['observations']));obs=x['observations'][:n];nom=run.local.NOMINAL[:n]
        parameters=np.array(a['parameters']);base=np.array(a['base_gains'])
        delta,contributions,activation=decompose(obs,nom,ids,parameters)
        np.testing.assert_array_equal(activation,x['dynamic_activation'][:n])
        gains=np.stack([run.dynamic_gains(base,parameters,o,nm,obs[0],nom[0],ids)[0] for o,nm in zip(obs,nom)])
        np.testing.assert_array_equal(gains,x['dynamic_gains'][:n])
        extra=np.stack([run.local.local_feedback(o,nm,ids,g) for o,nm,g in zip(obs,nom,gains)])
        np.testing.assert_array_equal(extra,x['local_hip_extra_rad'][:n])
        frozen_same_state=np.stack([run.local.local_feedback(o,nm,ids,base) for o,nm in zip(obs,nom)])
        extra_delta=extra-frozen_same_state
        gain_delta=gains-base
        np.testing.assert_allclose(gain_delta,activation[:,None]*parameters[:6],atol=2e-16,rtol=0.)
        label='standard' if case is None else ('rescued' if a['success'] and not b['success'] else ('regressed' if b['success'] and not a['success'] else ('retained' if a['success'] else 'failed_both')))
        if case is None:assert not np.any(delta) and not np.any(activation) and not np.any(extra)
        early_by_label.setdefault(label,[]).append(float(activation[:50].mean()))
        profiles.append(activation[:50])
        count=min(len(x['observations']),len(y['observations']))
        target_difference=np.abs(x['applied'][:count]-y['applied'][:count]).max(1)
        sensor_difference=np.abs(x['observations'][:count,:34].astype(float)-y['observations'][:count,:34].astype(float)).max(1)
        windows=[]
        for lo,hi in WINDOWS:
            hi=min(hi,n)
            if lo>=hi:continue
            c=contributions[lo:hi].reshape(hi-lo,4,3).sum(2)
            windows.append(dict(control_start=lo,control_end_exclusive=hi,
                activation_mean=float(activation[lo:hi].mean()),activation_min=float(activation[lo:hi].min()),activation_max=float(activation[lo:hi].max()),
                contribution_mean_gyro_up_position_velocity=c.mean(0).tolist(),
                contribution_mean_absolute_gyro_up_position_velocity=np.abs(c).mean(0).tolist(),
                gain_change_mean=gain_delta[lo:hi].mean(0).tolist(),
                same_state_feedback_change_max_rad=np.abs(extra_delta[lo:hi]).max(0).tolist(),
                observed_extra_near_cap_controls=int(np.sum(np.any(np.abs(extra[lo:hi])>=.18-1e-8,axis=1)))))
        alternatives=[]
        for p in range(4):
            path=run.selector.source.OUTPUT/f'program_{p:02d}'/f'case_{case}'
            z,zrow=load(path);assert zrow['initial_hash']==a['initial_hash']
            alternatives.append(dict(program=p,success=zrow['success'],valid=zrow['valid'],
                first_target_difference_vs_candidate_control=first(np.abs(x['applied'][:min(len(z['applied']),len(x['applied']))]-z['applied'][:min(len(z['applied']),len(x['applied']))]).max(1),1e-8)))
        row=dict(case_seed=case,label=label,initial_hash=a['initial_hash'],candidate_success=a['success'],baseline_success=b['success'],candidate_valid=a['valid'],
            first_executed_target_difference_control=first(target_difference,1e-8),first_sensor_difference_control=first(sensor_difference,1e-8),
            first_sensor_difference_over_001_control=first(sensor_difference,.01),diagnostic_thresholds_not_acceptance=True,
            scalar_actual_execution_exact=True,maximum_gain_change=float(np.abs(gain_delta).max()),
            six_gain_updates_rank_one=True,windows=windows,original_peaks=a['peaks'],controls=a['controls'],alternatives=alternatives)
        dest=OUTPUT/f'case_{case}';dest.mkdir()
        np.savez_compressed(dest/'actual_coupling.npz',state_change_delta=delta,feature_contributions=contributions,
            activation=activation,gain_change=gain_delta,same_state_extra_rad=extra,same_state_frozen_extra_rad=frozen_same_state,
            feedback_change_on_identical_state_rad=extra_delta,target_difference_rad=target_difference,sensor_difference=sensor_difference)
        run.local.write_json(dest/'result.json',row);rows.append(row)
    summary={k:dict(count=len(v),mean=float(np.mean(v)),minimum=float(min(v)),maximum=float(max(v))) for k,v in early_by_label.items()}
    nonstandard=np.stack(profiles[1:]);cosines=[]
    for i in range(len(nonstandard)):
        for j in range(i+1,len(nonstandard)):
            denominator=np.linalg.norm(nonstandard[i])*np.linalg.norm(nonstandard[j])
            if denominator:cosines.append(float(nonstandard[i]@nonstandard[j]/denominator))
    assert all(digest(p)==h for p,h in hashes.items())
    run.local.write_json(OUTPUT/'results.json',dict(read_only=True,no_dynamics_no_label_changes=True,rows=rows,
        sensor_joint_indices=ids.tolist(),actuator_indices=aid.tolist(),early_activation_by_outcome=summary,
        first_50_activation_pair_cosine_min=float(min(cosines)),first_50_activation_pair_cosine_median=float(np.median(cosines)),
        rank_one_is_algebra_not_unique_cause=True,full_task_completed=False,hardware_readiness=False,
        qualification_never_loaded=True,hashes={**hashes,str(Path(__file__)):digest(__file__),str(run.local.prior.program.SCENE):digest(run.local.prior.program.SCENE)}))
    print('R150_TERMINAL',json.dumps(dict(early_activation_by_outcome=summary,cosine_median=float(np.median(cosines)),
        rescued=[r for r in rows if r['label']=='rescued'],invalid=[r for r in rows if not r['candidate_valid']])),flush=True)

if __name__=='__main__':main()
