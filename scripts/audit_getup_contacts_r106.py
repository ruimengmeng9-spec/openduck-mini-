"""Offline collision geometry of stored R102 states, never a dynamic trial.

Scratch qpos/qvel assignment is solely mj_forward diagnostics. There is no
integration, controller call or relabeling of saved physics/standing evidence.
Forces are not inferred from incomplete solver warmstart snapshots.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from diagnostics.getup_independent_native import digest
from diagnostics.probe_getup_anchor_r101 import SCENE,ROOT


def contacts(model,data,floor,feet):
    bodies=set()
    for contact in data.contact:
        if floor in contact.geom and contact.dist<=.001:
            other=int(contact.geom[1] if contact.geom[0]==floor else contact.geom[0])
            bodies.add(int(model.geom_bodyid[other]))
    return np.array([body in bodies for body in feet],dtype=int)


def main():
    run=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    out=ROOT/'outputs/getup_contact_audit_r106_20261005';out.mkdir(exist_ok=False)
    result=json.loads((run/'results.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    floor=model.geom('floor').id;feet=[model.body(n).id for n in ('foot_assembly','foot_assembly_2')]
    rows=[];nominal=None
    for group,label in [('development_baseline','baseline'),('development_candidate','candidate')]:
        for row in result['baseline' if label=='baseline' else 'development']['rows']:
            path=run/group/f"case_{row['case_seed']}.npz"
            with np.load(path) as z:
                flags=[]
                for q,v in zip(z['qpos'][:160],z['qvel'][:160]):
                    data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
                    flags.append(contacts(model,data,floor,feet))
            flags=np.array(flags)
            if row['case_seed'] is None and label=='baseline':nominal=flags.copy()
            mismatch=np.any(flags!=nominal[:len(flags)],axis=1)
            changed=np.flatnonzero(mismatch)
            window=flags[80:130];modes=window[:,0]+2*window[:,1]
            item=dict(group=label,seed=row['case_seed'],success=row['success'],valid=row['valid'],
                first_contact_mismatch_s=None if not len(changed) else float((changed[0]+1)*.02),
                early_mismatch_fraction=float(mismatch[80:130].mean()),
                early_mode_counts=np.bincount(modes,minlength=4).tolist(),
                trace_sha256=digest(path))
            rows.append(item)
            np.savez_compressed(out/f"{label}_{row['case_seed']}.npz",foot_contacts=flags,mismatch=mismatch)
    summary={}
    for group in ('baseline','candidate'):
        summary[group]={}
        for success in (False,True):
            subset=[r for r in rows if r['group']==group and r['success']==success and r['seed'] is not None]
            summary[group]['success' if success else 'failure']=dict(count=len(subset),
                median_early_mismatch_fraction=float(np.median([r['early_mismatch_fraction'] for r in subset])),
                aggregate_mode_counts=np.sum([r['early_mode_counts'] for r in subset],axis=0).tolist())
    (out/'results.json').write_text(json.dumps(dict(rows=rows,summary=summary,
        geometry_only=True,force_reconstruction_not_attempted=True,no_dynamics_integration=True,
        no_saved_label_changes=True,not_causal_proof=True,window_s=[1.6,2.6],
        source_sha256=digest(Path(__file__)),scene_sha256=digest(SCENE),
        simulation_only=True,hardware_readiness=False,full_task_completed=False),indent=2))
    print('R106_SUMMARY',json.dumps(summary),flush=True)
    print('R106_ROWS',json.dumps([{k:v for k,v in r.items() if k!='trace_sha256'} for r in rows]),flush=True)


if __name__=='__main__':main()
