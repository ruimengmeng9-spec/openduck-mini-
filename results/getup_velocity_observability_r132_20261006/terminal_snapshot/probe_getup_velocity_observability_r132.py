"""Sensor-only feature regression to offline privileged velocity targets.

Never a controller or dynamic rollout. No unseen qualification access.
"""
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import probe_getup_sensor_history_r121 as history
from diagnostics.train_getup_history_program_r122 import sensor_features
from diagnostics.getup_independent_native import digest

ROOT=history.ROOT
OUTPUT=ROOT/'outputs/getup_velocity_observability_r132_20261006'
RIDGES=(1.,10.,100.)


def temporal(history):
    h=np.asarray(history,dtype=float)
    if h.shape!=(40,50) or not np.isfinite(h).all():raise ValueError('Original causal sensors only')
    return np.concatenate((h[-1],h[:,:34].reshape(8,5,34).mean(1).reshape(-1)))


def fit(x,y,ridge):
    mean=x.mean(0);std=np.maximum(x.std(0),1e-4);z=(x-mean)/std
    ymean=y.mean(0);weight=z.T@np.linalg.solve(z@z.T+ridge*np.eye(len(z)),y-ymean)
    return dict(mean=mean,std=std,weight=weight,ymean=ymean)


def predict(w,x):return (x-w['mean'])/w['std']@w['weight']+w['ymean']


def metrics(y,p,constant):
    mse=np.mean((y-p)**2,axis=0);base=np.mean((y-constant)**2,axis=0)
    return dict(rmse_mps=np.sqrt(mse).tolist(),mean_mse=float(mse.mean()),
        reduction_vs_training_mean=(1-mse/np.maximum(base,1e-20)).tolist(),mean_target_speed_mps=float(np.linalg.norm(y,axis=1).mean()))


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    source=history.OUTPUT;manifest=json.loads((source/'manifest.json').read_text())
    cases=[r['case_seed'] for r in manifest];assert cases==[None,*history.program.TRAIN,*range(3160000,3160040),*range(3180000,3180040)]
    with np.load(source/'causal_sensor_dataset.npz',allow_pickle=False) as z:h=z['histories'].copy()
    assert h.shape==(105,40,50)
    target=[];initials=[]
    for case in cases:
        file=source/'cases'/f'case_{case}'/'audit_initial_state.npz'
        with np.load(file,allow_pickle=False) as z:
            q=z['observed_qpos'];v=z['observed_qvel'];matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,q[3:7])
            target.append(matrix.reshape(3,3).T@v[:3])
        initials.append(dict(case_seed=case,target_source_hash=digest(file)))
    y=np.stack(target);features=dict(snapshot=np.stack([sensor_features(v,'snapshot') for v in h]),
        means=np.stack([sensor_features(v,'history') for v in h]),temporal=np.stack([temporal(v) for v in h]))
    reports={}
    for name,x in features.items():
        scores=[];predictions=[]
        for ridge in RIDGES:
            loo=np.empty((64,3));constants=np.empty_like(loo)
            for index in range(1,65):
                keep=np.arange(65)!=index;w=fit(x[:65][keep],y[:65][keep],ridge)
                loo[index-1]=predict(w,x[index]);constants[index-1]=w['ymean']
            score=metrics(y[1:65],loo,constants);scores.append(dict(ridge=ridge,**score));predictions.append(loo)
        chosen=min(range(3),key=lambda i:(scores[i]['mean_mse'],-RIDGES[i]));w=fit(x[:65],y[:65],RIDGES[chosen])
        held=predict(w,x[65:]);reports[name]=dict(feature_dimension=x.shape[1],grid=scores,chosen_ridge=RIDGES[chosen],
            known_65_leave_one_out=scores[chosen],held_318_development=metrics(y[65:],held,w['ymean']))
        np.savez_compressed(OUTPUT/(name+'_diagnostic_model.npz'),**w,chosen_ridge=RIDGES[chosen])
        # Targets saved only as offline diagnostics, not a policy model input.
        np.savez_compressed(OUTPUT/(name+'_diagnostic_predictions.npz'),loo_prediction=predictions[chosen],held_prediction=held,
            offline_supervision_targets=y,train_sensor_features=x[:65],held_sensor_features=x[65:])
    result=dict(reports=reports,source_initials=initials,training_count=65,held_development_count=40,
        no_dynamic_integration=True,not_controller=True,no_getup_success_claim=True,privileged_targets_offline_only=True,
        no_qualification_state_access=True,source_hashes={str(p):digest(p) for p in (Path(__file__),source/'causal_sensor_dataset.npz',source/'manifest.json')})
    (OUTPUT/'results.json').write_text(json.dumps(result,indent=2));print('R132_TERMINAL',json.dumps(reports),flush=True)


if __name__=='__main__':main()
