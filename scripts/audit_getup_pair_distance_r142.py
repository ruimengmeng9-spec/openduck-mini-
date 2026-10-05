"""Check pair distance API vs contact penetration on recorded pre-states."""
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics.audit_getup_sensor_geometry_r137 import ROOT,SCENE,SensorGeometry
from diagnostics.probe_getup_substep_contacts_r126b import OUTPUT as REPLAY
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_pair_distance_r142_20261006'


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    fk=SensorGeometry(mujoco.MjModel.from_xml_path(str(SCENE)));rows=[]
    for file in sorted(REPLAY.glob('*/threshold_crossings.json')):
        crossings=json.loads(file.read_text())
        if not crossings:continue
        event=crossings[0];q=np.array(event['pre_step_qpos'])
        sensor=np.zeros(55,dtype=np.float32);sensor[6:20]=(q[fk.qadr]-fk.home).astype(np.float32)
        result=fk.risk(sensor)
        a,b=event['geom_ids'];distance=float(mujoco.mj_geomDistance(fk.model,fk.data,a,b,.1,None))
        contact=min(float(c.dist) for c in fk.data.contact if sorted(map(int,c.geom))==[a,b])
        rows.append(dict(source=str(file),source_hash=digest(file),pair=[a,b],original_prestate_penetration_m=event['self_penetration_m'],
            scalar_pair_distance_api_m=distance,actual_forward_contact_distance_m=contact,
            pair_distance_matches_contact=abs(distance-contact)<1e-6,
            sensor_contact_matches_original=abs(-contact-event['self_penetration_m'])<1e-6,
            sensor_all_contact_max_penetration_m=float(result[2])))
    report=dict(rows=rows,known_collision_count=len(rows),pair_distance_match_count=sum(r['pair_distance_matches_contact'] for r in rows),
        sensor_contact_match_count=sum(r['sensor_contact_matches_original'] for r in rows),
        no_dynamic_integration=True,no_policy_changes=True,privileged_recorded_prestate_only_offline_metric_validation=True,
        no_success_or_physical_relabeling=True,reserved_qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(p):digest(p) for p in [Path(__file__),SCENE,REPLAY/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R142_TERMINAL',json.dumps(dict(count=len(rows),pair_match=report['pair_distance_match_count'],sensor_contact_match=report['sensor_contact_match_count'],examples=rows[:2])),flush=True)


if __name__=='__main__':main()
