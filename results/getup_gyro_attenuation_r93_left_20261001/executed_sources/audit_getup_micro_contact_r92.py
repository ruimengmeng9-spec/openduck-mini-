"""Read actual substep contacts in exact R91 paired replays; no control change."""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics import train_getup_head_distillation_r89 as student
from diagnostics.probe_getup_head_tolerance_r91 import constant_network
from diagnostics.train_getup_head_feedback_r87 import path
from diagnostics.getup_independent_native import DT,digest
from diagnostics.probe_getup_wait_library_r77 import save_trace
from diagnostics.audit_getup_wait_failures_r79 import first_run,load,up_z


class Observer:
    columns=('elapsed_s','left_foot_force_n','right_foot_force_n','trunk_force_n',
             'head_force_n','other_floor_force_n','left_contacts','right_contacts',
             'trunk_contacts','head_contacts','self_penetration_m','floor_penetration_m')
    def __init__(self,original,model,start):
        self.original=original;self.start=start;self.rows=[]
        self.floor=model.geom('floor').id
        self.bodies=[model.body(n).id for n in ('foot_assembly','foot_assembly_2','trunk_assembly','head_assembly')]
    def __call__(self,model,data,*args,**kwargs):
        value=self.original(model,data,*args,**kwargs)  # Exactly one original integration.
        force=np.zeros(5);counts=np.zeros(4);self_peak=floor_peak=0.
        for i,c in enumerate(data.contact):
            if self.floor not in c.geom:
                self_peak=max(self_peak,-float(c.dist));continue
            floor_peak=max(floor_peak,-float(c.dist))
            if c.dist>.001:continue
            other=int(c.geom[1] if c.geom[0]==self.floor else c.geom[0])
            body=int(model.geom_bodyid[other]);slot=self.bodies.index(body) if body in self.bodies else 4
            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
            force[slot]+=max(float(wrench[0]),0.)
            if slot<4:counts[slot]+=1
        self.rows.append(np.r_[float(data.time)-self.start,force,counts,self_peak,floor_peak])
        return value


def onset(values,threshold,samples,dt):
    i=first_run(np.asarray(values)>threshold,samples)
    return None if i is None else float((i+1)*dt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Need fresh output')
    root=Path('/data/shijinsheng/open_duck');prior=root/'outputs/getup_head_tolerance_r91_left_20261001'
    result=json.loads((prior/'results.json').read_text())
    scene=root/'training/getup_decomposed_r4/model/scene.xml';stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if result[key]!=digest(file):raise RuntimeError('Frozen R91 inputs changed')
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    seeds=(None,769003,773014);base.init_worker(str(scene),str(stand),ck,[s for s in seeds if s is not None])
    sim,cases,ck,ids=base._CTX;targets,phases,ref,gains=path(2.)
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    records=[];hashes={};observed={}
    for seed in seeds:
        for arm in (0,1,3):
            bias=result['rows'][arm]['bias'];source,peaks,initial,initial_hash=cases[seed]
            observer=Observer(mujoco.mj_step,sim.model,source['time'])
            mujoco.mj_step=observer
            try:
                value,trace,_,_=student.run(sim,source,peaks,targets,phases,ids,ref,gains,
                    network=constant_network(bias),strength=0. if arm==0 else 1.,record=True)
            finally:mujoco.mj_step=observer.original
            dest=args.output/f'case_{seed}_arm{arm}';dest.mkdir()
            save_trace(dest/'control.npz',trace)
            samples=np.asarray(observer.rows);np.savez_compressed(dest/'substeps.npz',values=samples,columns=Observer.columns)
            expected=result['rows'][arm]['nominal'] if seed is None else next(v for v in result['rows'][arm]['cases'] if v['seed']==seed)
            for key in ('score','valid','strict_tail_s','completed_steps'):
                if value[key]!=expected[key]:raise RuntimeError('Observer changed recorded outcome')
            original=prior/f'grid_{arm:02d}/case_{seed}/trajectory.npz'
            with np.load(original,allow_pickle=False) as z:
                if not (np.array_equal(z['qpos'],np.array([r[1] for r in trace])) and
                        np.array_equal(z['qvel'],np.array([r[2] for r in trace]))):
                    raise RuntimeError('Observer changed trajectory')
            hashes[str(original.relative_to(prior))]=digest(original)
            observed[(seed,arm)]=(samples,load(dest/'control.npz'))
            row={'seed':seed,'arm':arm,'bias':bias,'initial_state_sha256':initial_hash,
                 'integration_calls':len(samples),'unchanged_outcome_and_trace':True,'result':value}
            records.append(row);(dest/'summary.json').write_text(json.dumps(row,indent=2))
            print('OBSERVED',seed,arm,'UNCHANGED',len(samples),'substeps',flush=True)
    comparisons=[]
    for seed in seeds:
        baseline,old=observed[(seed,0)]
        for arm in (1,3):
            samples,control=observed[(seed,arm)];n=min(len(samples),len(baseline));k=min(len(control['time']),len(old['time']))
            force=abs(samples[:n,1:6]-baseline[:n,1:6]).max(axis=1)
            topology=np.any(samples[:n,6:10]!=baseline[:n,6:10],axis=1)
            gyro=np.linalg.norm(control['imu_error'][:k,2:]-old['imu_error'][:k,2:],axis=1)
            leg=abs(control['residual_rad'][:k,:8]-old['residual_rad'][:k,:8]).max(axis=1)
            row={'seed':seed,'arm':arm,'first_floor_force_difference_over_3N_for_20ms':onset(force,3.,10,.002),
                 'first_floor_contact_topology_difference_for_10ms':onset(topology,.5,5,.002),
                 'first_scaled_gyro_error_difference_over_0_03_for_100ms':onset(gyro,.03,5,DT),
                 'first_leg_feedback_difference_over_0_01rad_for_100ms':onset(leg,.01,5,DT),
                 'first_root_up_z_difference_over_0_05_for_100ms':onset(abs(control['root_up_z'][:k]-old['root_up_z'][:k]),.05,5,DT)}
            comparisons.append(row);print('ONSET',json.dumps(row),flush=True)
    report={'simulation_only':True,'diagnosis_only':True,'hardware_readiness':False,'full_task_completed':False,
            'no_physics_command_audit_change':True,'no_midpath_state_reset':True,'no_fresh_test_seeds':True,
            'diagnostic_thresholds_not_acceptance_changes':True,'substep_contacts_observed_not_reconstructed':True,
            'source_sha256':digest(__file__),'scene_sha256':digest(scene),'stand_sha256':digest(stand),
            'checkpoint_sha256':digest(args.checkpoint),'r91_result_sha256':digest(prior/'results.json'),
            'input_trace_hashes':hashes,'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')},
            'records':records,'comparisons':comparisons,
            'causal_limit':'Temporal onsets alone do not prove feedback causes contact divergence; test controlled gyro scaling separately'}
    (args.output/'results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
