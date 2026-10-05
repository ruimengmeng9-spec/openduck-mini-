"""Read-only causal error-product audit of complete R153 paired trajectories."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import probe_getup_fixed_dynamic_r153 as source
from diagnostics.getup_independent_native import digest

run=source.run
OUTPUT=run.ROOT/'outputs/getup_error_products_r154_20261006'

def main():
    assert not OUTPUT.exists();OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    run.prior.local.init_worker();ids=np.array([9,10,11]);result=json.loads((source.OUTPUT/'results.json').read_text())
    rows=[];hashes={}
    for a,b in zip(result['candidate']['rows'],result['baseline']['rows']):
        case=a['case_seed'];assert a['initial_hash']==b['initial_hash'] and case==b['case_seed']
        paths=[source.OUTPUT/name/f'case_{case}' for name in ['candidate','baseline']];data=[]
        for path in paths:
            for name in ['trajectory.npz','result.json']:hashes[str(path/name)]=digest(path/name)
            with np.load(path/'trajectory.npz',allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
        x,y=data;n=min(529,len(x['observations']));obs=x['observations'][:n];nom=run.prior.local.NOMINAL[:n];base=np.array(a['base_gains'])
        errors=np.stack([run.prior.state_features(o,m,ids)[6:] for o,m in zip(obs,nom)])
        activation=[];extra=[];frozen=[];gains=[]
        for o,m in zip(obs,nom):
            g,act=run.jointwise_gains(base,a['parameters'],o,m,obs[0],nom[0],ids)
            gains.append(g);activation.append(act);extra.append(run.prior.local.local_feedback(o,m,ids,g))
            frozen.append(run.prior.local.local_feedback(o,m,ids,base))
        activation=np.array(activation);gains=np.array(gains);extra=np.array(extra);frozen=np.array(frozen)
        np.testing.assert_array_equal(activation,x['jointwise_activation'][:n]);np.testing.assert_array_equal(gains,x['jointwise_gains'][:n])
        np.testing.assert_array_equal(extra,x['local_hip_extra_rad'][:n])
        products=activation*errors
        weighted=(gains-base)*errors
        input_delta=weighted[:,:3]+weighted[:,3:]
        reconstructed=-.18*np.tanh(base[:3]*errors[:,:3]+base[3:]*errors[:,3:]+input_delta)
        # Different associative grouping can round at machine precision. The
        # execution scalar above remains the exact record; algebra is numerical.
        np.testing.assert_allclose(reconstructed,extra,atol=2e-16,rtol=0.)
        label='standard' if case is None else ('rescued' if a['success'] and not b['success'] else ('regressed' if b['success'] and not a['success'] else ('retained' if a['success'] else 'failed_both')))
        count=min(len(x['observations']),len(y['observations']))
        td=np.abs(x['applied'][:count]-y['applied'][:count]).max(1)
        sd=np.abs(x['observations'][:count,:34].astype(float)-y['observations'][:count,:34].astype(float)).max(1)
        stats=[]
        for hi in [min(50,n),n]:
            active=np.abs(products[:hi])>0
            stats.append(dict(end_control_exclusive=hi,product_mean_by_channel=products[:hi].mean(0).tolist(),
                product_nonnegative_fraction_where_nonzero=[float(np.mean(products[:hi,j][active[:,j]]>=0)) if np.any(active[:,j]) else None for j in range(6)],
                maximum_input_delta_by_joint=np.abs(input_delta[:hi]).max(0).tolist(),
                same_state_feedback_delta_max_rad=np.abs(extra[:hi]-frozen[:hi]).max(0).tolist()))
        row=dict(case_seed=case,label=label,initial_hash=a['initial_hash'],candidate_success=a['success'],baseline_success=b['success'],candidate_valid=a['valid'],
            scalar_activation_gain_feedback_exact=True,algebra_max_error_rad=float(np.abs(reconstructed-extra).max()),windows=stats,
            first_target_difference_control=int(np.flatnonzero(td!=0)[0]) if np.any(td!=0) else None,
            first_sensor_difference_control=int(np.flatnonzero(sd!=0)[0]) if np.any(sd!=0) else None,
            original_peaks=a['peaks'],controls=a['controls'],diagnostic_not_first_physics_substep=True)
        dest=OUTPUT/f'case_{case}';dest.mkdir();np.savez_compressed(dest/'error_products.npz',current_errors=errors,causal_error_change=errors-errors[0],
            activation=activation,error_products=products,weighted_increment=weighted,input_delta=input_delta,
            actual_extra_rad=extra,same_state_frozen_extra_rad=frozen,target_difference_rad=td,sensor_difference=sd)
        run.prior.local.write_json(dest/'result.json',row);rows.append(row)
    assert all(digest(p)==h for p,h in hashes.items())
    run.prior.local.write_json(OUTPUT/'results.json',dict(read_only=True,no_dynamic_integration=True,no_outcome_relabeling=True,rows=rows,
        algebra='gain increment c*tanh(e-e0) multiplies current e inside local feedback; for e0=0 this product is even in e, not an odd restorative residual',
        not_a_unique_failure_cause=True,hypothesis='Test a direct odd residual from causal joint error changes, without multiplying by the current error, on the unchanged frozen R134 base. This remains unverified.',
        qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False))
    run.prior.local.write_json(OUTPUT/'source_hashes.json',hashes)
    print('R154_READ_ONLY_EXACT',len(rows),max(r['algebra_max_error_rad'] for r in rows),flush=True)

if __name__=='__main__':main()
