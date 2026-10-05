"""Fixed distance-query argument contract audit; no physics flag changes."""
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics.audit_getup_sensor_geometry_r137 import ROOT,SCENE,SensorGeometry
from diagnostics.probe_getup_substep_contacts_r126b import OUTPUT as OLD
from diagnostics.probe_getup_passive_risk_r141b import OUTPUT as NEW
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_distance_contract_r143b_20261006'


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    model=mujoco.MjModel.from_xml_path(str(SCENE));fk=SensorGeometry(model)
    metadata=dict(version=mujoco.__version__,enableflags=int(model.opt.enableflags),disableflags=int(model.opt.disableflags),
        integrator=int(model.opt.integrator),solver=int(model.opt.solver),timestep=model.opt.timestep,
        ccd_tolerance=model.opt.ccd_tolerance,ccd_iterations=int(model.opt.ccd_iterations))
    metadata['enable_bits']={n:bool(int(model.opt.enableflags)&int(getattr(mujoco.mjtEnableBit,n))) for n in dir(mujoco.mjtEnableBit) if n.startswith('mjENBL_')}
    metadata['disable_bits']={n:bool(int(model.opt.disableflags)&int(getattr(mujoco.mjtDisableBit,n))) for n in dir(mujoco.mjtDisableBit) if n.startswith('mjDSBL_')}
    rows=[];limits=(0.,.001,.005,.1)
    for base in (OLD,NEW):
        assert json.loads((base/'results.json').read_text())['all_full_trace_bitwise_equal']
        for file in sorted(base.glob('*/threshold_crossings.json')):
            events=json.loads(file.read_text())
            if not events:continue
            event=events[0];q=np.array(event['pre_step_qpos']);sensor=np.zeros(55,dtype=np.float32)
            sensor[6:20]=(q[fk.qadr]-fk.home).astype(np.float32);fk.risk(sensor)
            pair=event['geom_ids'];contacts=[float(c.dist) for c in fk.data.contact if sorted(map(int,c.geom))==pair]
            assert contacts
            queries=[float(mujoco.mj_geomDistance(model,fk.data,*pair,limit,None)) for limit in limits]
            rows.append(dict(source=str(file),source_hash=digest(file),pair=pair,contact_distances_m=contacts,
                original_self_penetration_m=event['self_penetration_m'],query_distances_m=queries,
                query_matches_deepest_contact=[abs(d-min(contacts))<1e-6 for d in queries]))
    report=dict(engine_metadata=metadata,query_cutoffs_m=limits,rows=rows,
        deepest_contact_match_counts=[sum(r['query_matches_deepest_contact'][i] for r in rows) for i in range(4)],
        query_cutoff_not_controller_threshold=True,no_model_flag_or_collision_changes=True,
        no_dynamic_integration=True,no_force_inference=True,truth_only_offline_metric_validation=True,
        original_success_physics_labels_unchanged=True,no_new_controller=True,qualification_never_loaded=True,
        full_task_completed=False,hardware_readiness=False,
        hashes={str(f):digest(f) for f in [Path(__file__),SCENE,OLD/'results.json',NEW/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R143B_TERMINAL',json.dumps(dict(metadata=metadata,count=len(rows),cutoffs=limits,matches=report['deepest_contact_match_counts'],examples=rows[:1])),flush=True)


if __name__=='__main__':main()
