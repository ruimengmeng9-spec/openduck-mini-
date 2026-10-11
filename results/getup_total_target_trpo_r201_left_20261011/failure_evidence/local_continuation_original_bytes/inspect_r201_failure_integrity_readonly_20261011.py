"""Post-failure saved-evidence checks only; not an all-decision verifier."""
import hashlib
import json
from pathlib import Path
import numpy as np

R = Path('/data/shijinsheng/open_duck')
O = R/'outputs/getup_total_target_trpo_r201_left_20261011'
S = R/'outputs/getup_total_target_trpo_r201_smoke_20261011'
L = R/'outputs/getup_total_target_trpo_launcher_r201_20261011'
P = R/'outputs/getup_total_target_trpo_r201_preflight_20261011'
OLD = R/'outputs/getup_set_decision_r157_left_20261006/candidate'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def inventory():
    return {str(p): sha(p) for d in (O,S,L,P) for p in sorted(d.rglob('*')) if p.is_file()}
before = inventory()
manifests = []
def sources(manifest, folder):
    for p,v in manifest.items():
        assert sha(Path(p)) == v['sha256'] == sha(folder/v['copy'])
    manifests.append(dict(folder=str(folder),count=len(manifest)))
for d in (O,S,L):
    c=read(d/'contract.json');sources(c['sources'],d/'executed_sources')
    if d!=L:
        for p,h in c['hashes'].items(): assert sha(Path(p))==h
        for p,v in c['model_files'].items():
            # Existing model-file inventory format, without loading a model.
            assert sha(Path(p))==(v['sha256'] if isinstance(v,dict) else v)
    for f in d.glob('worker_source_manifest_*.json'):
        dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
        sources(read(f),dest)
for d in (P,L/'initial',L/'formal'):
    sources(read(d/'source_manifest.json'),d/'executed_sources')
    for f in (d/'worker_fixture').glob('worker_source_manifest_*.json'):
        dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
        sources(read(f),dest)
rows=[];baseline_count=0;startup_pairs=0
for d in (S,O):
    for f in sorted(d.rglob('trajectory.npz')):
        if f.parent.name.startswith('case_'):
            row=read(f.parent/'result.json');old=OLD/f.parent.name
            with np.load(f,allow_pickle=False) as z,np.load(old/'trajectory.npz',allow_pickle=False) as b:
                assert len(z['observations'])==row['controls']
                np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
                np.testing.assert_array_equal(z['applied'][0],b['applied'][0])
                assert z['construction_preparation_sensors'].shape==z['preparation_sensors'].shape==(40,50)
                assert row['physics_unchanged'] and row['compiled_model_unchanged']
                assert row['initial_hash']==read(old/'result.json')['initial_hash']
                if row['policy_mode']=='baseline':
                    # qpos/qvel decoded here only in isolated equality, not inference.
                    for k in b.files: np.testing.assert_array_equal(z[k],b[k])
                    assert row['peaks']==read(old/'result.json')['peaks'];baseline_count+=1
            rows.append(dict(path=str(f.parent),case=row['case_seed'],controls=row['controls'],
                success=row['success'],valid=row['valid'],peaks=row['peaks'],
                combined=row['maximum_combined_14_joint_correction_rad']))
for name in ('zero_parity','initial_standard','initial_sample'):
    for a in (S/name).glob('case_*'):
        b=O/name/a.name
        with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
            assert za.files==zb.files
            for k in za.files: np.testing.assert_array_equal(za[k],zb[k])
        assert read(a/'result.json')==read(b/'result.json')
        assert read(a/'rng.json')==read(b/'rng.json');startup_pairs+=1
after=inventory();assert before==after
assert len(rows)==112 and baseline_count==31 and startup_pairs==6
print(json.dumps(dict(new_dynamic_attempts=0,all_decision_verification=False,
    environment_model_or_integration_used=False,closed_paths=len(rows),controls=sum(q['controls'] for q in rows),
    invalid=[q for q in rows if not q['valid']],baseline_old_fields_exact=baseline_count,
    startup_all_arrays_results_rng_exact=startup_pairs,source_manifests=manifests,
    saved_planned_equals_applied=True,first_control_R157_exact=True,constructor_reset_40x50=True,
    inventory_count=len(before),inventory_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
    all_evidence_bytes_unchanged=True,original_missing_before_request_IMU_double_not_rebuilt=True,
    original_second_round_CG_trace_not_recovered=True)),flush=True)
