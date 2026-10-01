"""Observe original mj_step calls without changing physics or commands.

Control-rate saved qpos cannot identify every transient contact rejected by
the substep audit. Record contacts directly after each unmodified substep.
This is diagnosis, not a collision-model change or a qualification result.
"""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np

from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import digest,DT
from diagnostics.probe_getup_initial_settling_r84 import prepend, audited_reference
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.search_getup_reference_feedback_r64 import rollout
from diagnostics.probe_getup_wait_library_r77 import save_trace


class ContactObserver:
    def __init__(self,step):
        self.original=step; self.calls=0; self.events=[]

    def __call__(self,model,data,*args,**kwargs):
        result=self.original(model,data,*args,**kwargs)
        self.calls+=1
        floor=model.geom('floor').id
        for index,c in enumerate(data.contact):
            if floor in c.geom or c.dist>=-.003:continue
            g1,g2=map(int,c.geom); b1,b2=map(int,model.geom_bodyid[[g1,g2]])
            force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
            self.events.append({'substep':self.calls,'time':float(data.time),
                                'distance_m':float(c.dist),'geom_ids':[g1,g2],
                                'geom_names':[model.geom(g1).name,model.geom(g2).name],
                                'body_names':[model.body(b1).name,model.body(b2).name],
                                'contact_force':force.tolist(),
                                'qpos':data.qpos.copy().tolist(),
                                'qvel':data.qvel.copy().tolist(),
                                'ctrl':data.ctrl.copy().tolist()})
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Need new output')
    root=Path('/data/shijinsheng/open_duck')
    scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    prior=root/'outputs/getup_initial_settling_r84_left_20261001/results.json'
    r84=json.loads(prior.read_text())
    for key,path in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if r84[key]!=digest(path):raise RuntimeError('Frozen input mismatch')
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    base.init_worker(str(scene),str(stand),ck,[769001,773008])
    sim,cases,ck,ids=base._CTX
    original,phases=timed_targets(sim,ck,(.4,.6,.8),2.)
    targets0,phases0=prepend(original,phases,sim.home,0)
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    gains=np.concatenate([ck['prefix_gains'],ck['terminal_gains'],np.zeros((1,8))])
    ref0,_=audited_reference(sim,cases[None],targets0,args.output)
    if ref0 is None:raise RuntimeError('Identity reference changed')
    rows=[]
    for seed,controls in [(None,0),(None,10),(None,25),(None,50),(769001,0),(773008,0)]:
        dest=args.output/f'case_{seed}_wait_{controls}';dest.mkdir()
        targets,phase=prepend(original,phases,sim.home,controls)
        observer=ContactObserver(mujoco.mj_step)
        # A temporary observer in this isolated process. Exactly one original
        # integration per call; no modification to model, state, controls or audit.
        mujoco.mj_step=observer
        try:
            if controls:
                reference,peaks=audited_reference(sim,cases[None],targets,dest)
                value={'reference_valid':reference is not None,'peaks':peaks}
                trace=[]
            else:
                value,trace=rollout(sim,*cases[seed][:2],True,targets,phase,ids,ref0,gains,True)
                save_trace(dest/'control_trace.npz',trace)
        finally:
            mujoco.mj_step=observer.original
        if seed is None and controls==0:
            expected=r84['training_grid'][0]['nominal']
            for key in ('score','valid','strict_tail_s','completed_steps'):
                if value[key]!=expected[key]:raise RuntimeError('Observer altered canonical result')
        (dest/'contact_events.json').write_text(json.dumps(observer.events,indent=2))
        deepest=min(observer.events,key=lambda e:e['distance_m']) if observer.events else None
        row={'seed':seed,'settling_controls':controls,'initial_state_sha256':cases[seed][3],
             'integration_calls':observer.calls,'events':len(observer.events),
             'result':value,'deepest_event':deepest}
        rows.append(row)
        print('CONTACT',seed,controls,'events',len(observer.events),'deepest',
              None if deepest is None else [deepest['distance_m'],deepest['body_names']],flush=True)
    result={'diagnosis_only':True,'simulation_only':True,'hardware_readiness':False,
            'full_task_completed':False,'physics_commands_and_audit_unchanged':True,
            'observer_identity_contract_passed':True,'fresh_qualification_seeds_used':False,
            'source_sha256':digest(__file__),'scene_sha256':digest(scene),'stand_sha256':digest(stand),
            'checkpoint_sha256':digest(args.checkpoint),'r84_result_sha256':digest(prior),
            'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')},'results':rows}
    (args.output/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
