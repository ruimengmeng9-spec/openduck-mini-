"""Offline forward geometry only: no integration, force inference or relabeling."""
import json
from collections import Counter
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics.probe_getup_anchor_r101 import ROOT, SCENE
from diagnostics.getup_independent_native import digest

OUTPUT = ROOT/'outputs/getup_self_geometry_r125_20261005'
R122 = ROOT/'outputs/getup_history_program_r122_left_20261005'
R124 = ROOT/'outputs/getup_continuous_nodes_r124_left_20261005'


def geometry(model, data, floor, q, v):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_forward(model, data)
    pairs = {}
    for c in data.contact:
        if floor in c.geom:
            continue
        pair = tuple(sorted(map(int, c.geom)))
        pairs[pair] = min(pairs.get(pair, 0.), float(c.dist))
    return pairs


def describe(model, pair):
    return dict(geom_ids=list(pair), geom_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in pair],
                body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in pair])


def analyze(model, folder, trace, label, case, saved_row):
    assert case is None or case < 3200000
    with np.load(trace, allow_pickle=False) as z:
        qpos = z['qpos'].copy(); qvel = z['qvel'].copy()
        applied = z['applied'].copy(); times = z['time'].copy()
    floor = model.geom('floor').id
    data = mujoco.MjData(model); repeat = mujoco.MjData(model)
    extrema = {}; maxima = []; worst_ids = []; flags = []
    root_invariance_max_error = 0.
    for i, (q, v) in enumerate(zip(qpos, qvel)):
        pairs = geometry(model, data, floor, q, v)
        # Independent data objects; no solver warmstart or force reconstruction.
        if i in (0, len(qpos)-1):
            assert pairs == geometry(model, repeat, floor, q, v)
        if pairs:
            pair = min(pairs, key=pairs.get); penetration = max(0., -pairs[pair])
        else:
            pair = (-1, -1); penetration = 0.
        maxima.append(penetration); worst_ids.append(pair); flags.append(penetration >= .004)
        for p, distance in pairs.items():
            if p not in extrema or distance < extrema[p]['distance_m']:
                extrema[p] = dict(distance_m=distance, control_index=i, control_endpoint_s=(i+1)*.02)
        # Diagnostic root-rigid-transform invariance, not dynamic root reset.
        # Only sample saved states; a potential runtime feature would use joint
        # sensors with canonical root in a separate FK-only data object.
        if i in (0, len(qpos)-1) or (i < 350 and i % 25 == 0):
            assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[0] == 0
            canonical = q.copy(); canonical[:3] = [0., 0., 10.]; canonical[3:7] = [1., 0., 0., 0.]
            moved = geometry(model, repeat, floor, canonical, np.zeros_like(v))
            assert pairs.keys() == moved.keys(), 'Self-contact identity changed under rigid transform'
            error = max([abs(pairs[p]-moved[p]) for p in pairs] or [0.])
            root_invariance_max_error = max(root_invariance_max_error, error)
            assert error < 1e-8
    order = sorted(extrema, key=lambda p: extrema[p]['distance_m'])
    marked = np.flatnonzero(flags)
    dest = folder/(label+'_'+str(case)); dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'endpoint_geometry.npz', self_penetration_m=maxima,
        worst_geom_ids=worst_ids, diagnostic_threshold_crossed=flags,
        original_saved_time=times, original_applied_targets=applied)
    result = dict(label=label, case_seed=case, trace_path=str(trace), trace_sha256=digest(trace),
        controls=len(qpos), saved_success=saved_row.get('success'), saved_valid=saved_row.get('valid'),
        saved_initial_hash=saved_row.get('initial_hash'), saved_substep_peaks=saved_row.get('peaks'),
        max_endpoint_self_penetration_m=max(maxima),
        first_endpoint_above_diagnostic_limit_s=None if not len(marked) else (int(marked[0])+1)*.02,
        deepest_pairs=[dict(**describe(model,p), **extrema[p]) for p in order[:8]],
        repeated_independent_forward_bitwise_equal=True, root_rigid_transform_error_m=root_invariance_max_error,
        endpoint_geometry_not_original_substep_peak=True, no_force_inference=True, no_success_relabeling=True)
    (dest/'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    OUTPUT.mkdir(exist_ok=False); shutil.copy2(__file__, OUTPUT/Path(__file__).name)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    old = json.loads((R122/'results.json').read_text())
    new = json.loads((R124/'results.json').read_text())
    jobs = []; invalid_cases = set()
    for mode in ('snapshot','history'):
        for prefix, rows, base in [
            ('r122',old['summaries'][mode]['all']['rows'],R122/mode),
            ('r124',new['summaries'][mode]['groups']['all']['rows'],R124/'continuous'/mode)]:
            for row in rows:
                if not row['valid']:
                    invalid_cases.add(row['case_seed'])
                    jobs.append((prefix+'_'+mode,row['case_seed'],base/f"case_{row['case_seed']}/trajectory.npz",row))
    assert len(invalid_cases) and all(s < 3200000 for s in invalid_cases)
    # Paired controls: both original R122 programmes, fixed zero, and successful
    # R115 teachers where available, using exactly the same saved initial hash.
    for case in sorted(invalid_cases):
        if case < 3000000:
            zero = ROOT/'outputs/getup_distill_r110_left_20261005/baseline'/f'case_{case}'
        elif case < 3180000:
            zero = ROOT/'outputs/getup_program_calibration_r114_left_20261005/qualification_baseline'/f'case_{case}'
        else:
            zero = ROOT/'outputs/getup_expanded_program_r116_left_20261005/qualification_baseline'/f'case_{case}'
        jobs.append(('zero',case,zero/'trajectory.npz',json.loads((zero/'result.json').read_text())))
        teacher = ROOT/'outputs/getup_expanded_teachers_r115_left_20261005/teachers'/f'case_{case}'
        if teacher.exists():
            jobs.append(('teacher',case,teacher/'trajectory.npz',json.loads((teacher/'result.json').read_text())))
        for mode in ('snapshot','history'):
            row = next(r for r in old['summaries'][mode]['all']['rows'] if r['case_seed']==case)
            key = ('r122_'+mode,case)
            if not any((j[0],j[1])==key for j in jobs):
                jobs.append((*key,R122/mode/f'case_{case}/trajectory.npz',row))
    rows = []
    for label, case, trace, row in jobs:
        rows.append(analyze(model,OUTPUT,trace,label,case,row))
        print('R125_TRACE',label,case,rows[-1]['max_endpoint_self_penetration_m'],flush=True)
    paired = {}
    for case in sorted(invalid_cases):
        selected = [r for r in rows if r['case_seed']==case]
        assert len({r['saved_initial_hash'] for r in selected})==1
        paired[str(case)] = dict(initial_hash_equal=True, labels=[r['label'] for r in selected])
    worst = Counter(tuple(r['deepest_pairs'][0]['body_names']) for r in rows
                    if r['label'].startswith(('r122','r124')) and r['saved_valid'] is False and r['deepest_pairs'])
    report = dict(rows=rows, paired_initial_hashes=paired, invalid_case_seeds=sorted(invalid_cases),
        invalid_deepest_body_pair_counts=[dict(bodies=list(k), count=v) for k,v in worst.items()],
        read_only_geometry=True, no_dynamic_integration=True, force_reconstruction_not_attempted=True,
        no_saved_label_changes=True, not_causal_proof=True, endpoint_not_substep_peak=True,
        reserved_qualification_never_loaded=True, simulation_only=True, hardware_readiness=False,
        full_task_completed=False, hashes={str(p):digest(p) for p in [Path(__file__),SCENE,R122/'results.json',R124/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R125_TERMINAL',json.dumps(report['invalid_deepest_body_pair_counts']),flush=True)


if __name__ == '__main__': main()
