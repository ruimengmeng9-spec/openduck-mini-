"""Read-only geometry at passively recorded pre/post collision states.

No integration, no force inference, no policy use of privileged root state.
Check whether canonical-root FK from joints reproduces observed self contact.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from diagnostics import audit_getup_self_geometry_r125 as geometry
from diagnostics.probe_getup_substep_contacts_r126b import OUTPUT as REPLAY
from diagnostics.getup_independent_native import digest

ROOT=geometry.ROOT
OUTPUT=ROOT/'outputs/getup_contact_prestate_r127_20261005'


def body_joints(m,body):
    result=[]
    while body>0:
        for j in range(int(m.body_jntadr[body]),int(m.body_jntadr[body]+m.body_jntnum[body])):
            if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:
                result.append(dict(name=m.joint(j).name,qpos_address=int(m.jnt_qposadr[j])))
        body=int(m.body_parentid[body])
    return result


def main():
    result=json.loads((REPLAY/'results.json').read_text());assert result['all_full_trace_bitwise_equal']
    OUTPUT.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(geometry.SCENE));d=mujoco.MjData(m);floor=m.geom('floor').id
    records=[]
    for row in result['rows']:
        crossing_file=REPLAY/(row['label']+'_'+str(row['case_seed']))/'threshold_crossings.json'
        crossings=json.loads(crossing_file.read_text())
        if not crossings:continue
        event=crossings[0];pair=tuple(event['geom_ids']);pre=np.array(event['pre_step_qpos']);post=np.array(event['post_step_qpos'])
        before=geometry.geometry(m,d,floor,pre,np.zeros(m.nv));after=geometry.geometry(m,d,floor,post,np.zeros(m.nv))
        joint_only=m.qpos0.copy();joint_only[7:]=pre[7:];joint_only[:3]=[0.,0.,10.];joint_only[3:7]=[1.,0.,0.,0.]
        canonical=geometry.geometry(m,d,floor,joint_only,np.zeros(m.nv))
        assert before.keys()==canonical.keys()
        error=max([abs(before[k]-canonical[k]) for k in before] or [0.]);assert error<1e-8
        distance=before.get(pair)
        # Contact distance is assessed, never assumed reproducible solely from
        # saved end-state. Preserve discrepancies rather than changing labels.
        reproduced=bool(distance is not None and abs(-distance-event['self_penetration_m'])<1e-8)
        geometries=[]
        for g in pair:
            body=int(m.geom_bodyid[g]);joints=body_joints(m,body)
            geometries.append(dict(id=g,name=m.geom(g).name,type=int(m.geom_type[g]),size=m.geom_size[g].tolist(),
                mesh_id=int(m.geom_dataid[g]),body=m.body(body).name,ancestor_joints=joints,
                pre_joint_angles={j['name']:float(pre[j['qpos_address']]) for j in joints},
                post_joint_angles={j['name']:float(post[j['qpos_address']]) for j in joints}))
        records.append(dict(label=row['label'],case_seed=row['case_seed'],first_event={k:v for k,v in event.items()
            if k not in ('pre_step_qpos','post_step_qpos','applied_target')},geometries=geometries,
            forward_pre_distance_m=distance,forward_post_distance_m=after.get(pair),
            observed_first_penetration_reproduced_from_prestate=reproduced,
            joint_only_canonical_root_distance_error_m=error,
            crossing_sha256=digest(crossing_file),no_forces_inferred=True))
    report=dict(records=records,no_dynamic_integration=True,no_success_relabeling=True,not_causal_proof=True,
        joint_only_geometry_checked_not_new_control_policy=True,
        all_observed_first_distances_reproduced=all(r['observed_first_penetration_reproduced_from_prestate'] for r in records),
        reserved_qualification_never_loaded=True,simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hashes={str(p):digest(p) for p in [Path(__file__),geometry.SCENE,REPLAY/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R127_TERMINAL',json.dumps(dict(count=len(records),prestate_reproduced=report['all_observed_first_distances_reproduced'],
        pairs=[dict(label=r['label'],case=r['case_seed'],bodies=[g['body'] for g in r['geometries']],
            types=[g['type'] for g in r['geometries']],before=r['forward_pre_distance_m'],after=r['forward_post_distance_m']) for r in records])),flush=True)


if __name__=='__main__':main()
