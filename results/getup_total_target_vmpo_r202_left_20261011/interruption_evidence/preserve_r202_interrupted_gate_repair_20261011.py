"""Append-only interruption inventory/interface preservation, not full-decision proof."""
import pathlib,json,hashlib,time,shutil
import numpy as np
root=pathlib.Path('/data/shijinsheng/open_duck')
smoke=root/'outputs/getup_total_target_vmpo_r202_smoke_20261011'
formal=root/'outputs/getup_total_target_vmpo_r202_left_20261011'
repair=root/'outputs/getup_total_target_vmpo_launcher_r202b_20261011'
receipt=root/'tmp/getup_total_target_vmpo_r202_interrupted_gate_repaired_20261011.json'
assert not receipt.exists()
digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
before=json.loads((repair/'original_dynamic_evidence_inventory_before.json').read_text())
assert all(digest(p)==sha for p,sha in before.items())
rows=[];controls=0
for directory in (smoke,formal):
    for f in sorted(directory.rglob('result.json')):
        row=json.loads(f.read_text())
        if 'controls' not in row:continue
        zpath=f.parent/'trajectory.npz'
        with np.load(zpath,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
            assert z['construction_preparation_sensors'].shape==(40,50)
            assert z['preparation_sensors'].shape==(40,50)
            assert len(z['applied'])==row['controls']
        rows.append(dict(file=str(f),sha256=digest(f),controls=row['controls'],
                         valid=row['valid'],success=row['success'],case_seed=row['case_seed']))
        controls+=row['controls']
assert len(rows)==62 and controls==139323
paired=[]
for folder,case in [('zero_parity',None),('zero_parity',769000),('zero_parity',773004),
                    ('initial_standard',None),('initial_sample',769002),('initial_sample',773004)]:
    a=smoke/folder/f'case_{case}';b=formal/folder/f'case_{case}'
    with np.load(a/'trajectory.npz',allow_pickle=False) as x,np.load(b/'trajectory.npz',allow_pickle=False) as y:
        assert x.files==y.files
        # qpos/qvel are used only in this isolated equality comparison.
        for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    assert json.loads((a/'result.json').read_text())==json.loads((b/'result.json').read_text())
    assert (a/'rng.json').read_bytes()==(b/'rng.json').read_bytes()
    paired.append(dict(folder=folder,case=case,all_arrays_result_RNG_exact=True))
assert (formal/'checkpoint_0000.learner.msgpack').exists()
missing=['checkpoint_0001.npz','checkpoint_0001.learner.msgpack','checkpoint_0002.npz',
         'training_0001/complete_original_MC_batch.npz','training_0001/learner_rng_before.json',
         'training_0001/actor_full_decisions.npz','training_0001/actor_update.json',
         'training_0001/value_full_updates.npz','training_0001/value_update.json',
         'check_0001','training_0002','check_0002','candidate','results.json',
         'training_closed_history.json','input_hashes_after.json']
assert all(not (formal/p).exists() for p in missing)
baseline=json.loads((formal/'development_baseline/results.json').read_text())
training=json.loads((formal/'training_0001/results.json').read_text())
assert baseline['successes']==18 and baseline['nominal_success'] and baseline['physical_failures']==0
assert training['successes']==8 and not training['nominal_success'] and training['physical_failures']==1
invalid=[dict(file=str(formal/'training_0001'/f"case_{v['case_seed']}"/'result.json'),row=v)
         for v in training['rows'] if not v['valid']]
files=set()
for d in (root/'outputs').iterdir():
    if 'r202' in d.name.lower() and d.is_dir():files.update(p for p in d.rglob('*') if p.is_file())
for d in (root/'tmp',root/'outputs',root/'projects/Open_Duck_Playground/diagnostics'):
    files.update(p for p in d.iterdir() if p.is_file() and 'r202' in p.name.lower())
for d in (root/'tmp').iterdir():
    if d.is_dir() and 'r202' in d.name.lower():files.update(p for p in d.rglob('*') if p.is_file())
live=sum(p.stat().st_size for p in files);forecast=(640+640+128)*1024**2
assert live+forecast<=2*1024**3 and shutil.disk_usage(root).free>=10*1024**3
record=dict(stage='user_interrupted_original_formal_gate_repaired_offline_no_restart',
    interrupted_by_user_authorization=True,original_formal_natural_exit_zero=False,
    original_launcher_natural_exit_receipt_exists=False,
    smoke_exit_zero_evidence='Original immutable launcher subprocess.run returncode-zero assertion and formal transition; no standalone smoke exit receipt fabricated',
    independent_smoke_paths=6,formal_saved_paths=56,total_saved_paths=62,controls=controls,
    all_saved_planned_equal_applied=True,constructor_reset_each40x50=True,
    formal_startup_pairs=paired,baseline_successes=18,training_first_round_successes=8,
    training_first_round_standard_success=False,training_first_round_invalid=invalid,
    completed_persisted_actor_updates=0,completed_persisted_value_updates=0,
    missing_original_evidence_not_recreated=missing,in_memory_progress_not_claimed=True,
    original_numeric_path_failure_preserved=True,
    original_launcher_initial_and_formal_regressions_each_16_with2skips=True,
    repaired_complete_gate_16_skipped0=True,repaired_gate_exit_code=0,
    new_dynamic_attempts_after_stop=0,remaining100_formal_paths_not_replayed=True,
    original_dynamic_budget_closed=True,best='R157 18/24',new_candidate_exists=False,
    full_decision_verification_complete=False,archive_complete=False,publication_complete=False,
    new_independent_readonly_full_decision_verifier_next=True,
    protected_live_server_not_signalled=True,original_files_unchanged=True,
    original_dynamic_evidence_inventory=before,rows=rows,
    all_R202_live_external_bytes=live,forecast_archive_git_extra_bytes=forecast,
    actual_plus_forecast_bytes=live+forecast,free_bytes=shutil.disk_usage(root).free,
    all_R202_output_dirs_including_new_repair_included=True,
    full_task_completed=False,hardware_readiness=False,unseen320_not_read=True,
    time_unix=time.time())
assert all(digest(p)==sha for p,sha in before.items())
with receipt.open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(dict(receipt=str(receipt),sha256=digest(receipt),
    paths=62,controls=controls,bytes=receipt.stat().st_size,live=live,total=live+forecast)))

