"""R7: explore normal-knee recovery branches with foot-orientation diagnostics.

The target restriction is a search curriculum, not a change to physical limits.
No root or velocity edits occur after initial fallen-state preparation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.audit_getup_crouch import SupportSim
from diagnostics.getup_aligned_extension import strict_entry
from diagnostics.getup_beam_reference import primitive
from diagnostics import getup_dynamic_beam as dynamic
from diagnostics.getup_independent_native import digest, feature_targets


def supported_entry(m):
    return strict_entry(m) and min(m['foot_up_alignment_to_home']) > .85


def anatomical_library(sim, previous, rng):
    rows = []
    for hip in (-1.15, -.8, -.4, .1):
        for knee in (.15, .7, 1.2, 1.55):
            for neck in (-.3, 1.1):
                # Coordinate signs follow the existing bilateral sagittal map.
                # Keep foot pitch close to home when the body becomes upright.
                ankle = np.clip(-hip-knee, sim.lower[4], sim.upper[4])
                rows.append([hip, knee, ankle, neck, 0., .05, 0.])
    targets = feature_targets(rows, sim.home, sim.lower, sim.upper)
    local = np.clip(rng.normal(previous, .4, (16, sim.model.nu)), sim.lower, sim.upper)
    home = np.clip(rng.normal(sim.home, .25, (8, sim.model.nu)), sim.lower, sim.upper)
    targets = np.concatenate([targets, local, home, sim.home[None], previous[None]])
    targets[:, [3, 12]] = np.clip(targets[:, [3, 12]], 0., sim.upper[[3, 12]])
    targets[:, [7, 8]] = 0.
    return targets


def posture_score(m):
    upright = float(np.clip((m['up_z']-.4)/.55, 0., 1.))
    lift = float(np.clip((m['height_m']-.04)/.12, 0., 1.2))
    alignment = float(np.mean(np.clip(m['foot_up_alignment_to_home'], -1., 1.)))
    knees = np.array(m['joint_positions_rad'])[[3, 12]]
    wrong_knees = float(np.maximum(-knees, 0.).sum())
    margin = m['com_support_margin_m']
    # A contact flag alone is not evidence of flat or loaded foot support.
    support = 0. if margin is None else float(np.clip(margin/.025, -1., 1.))
    return (2.*m['up_z'] + upright*(4.*lift + 1.3*alignment + .4*support - wrong_knees)
            + .2*sum(m['feet']) - .3*m['torso_contact'])


_sim = None


def init_worker(scene, stand):
    global _sim
    _sim = SupportSim(scene, stand)


def evaluate(job):
    state, target, duration = job
    result = primitive(_sim, state, target, duration)
    tail = result['measurements'][-min(8, len(result['measurements'])):]
    result['score'] = float(np.mean([posture_score(m) for m in tail])
                            + 8.*supported_entry(result['metric']) - 100.*result['maximum_penetration_m'])
    return result


def safe(result):
    ms = result['measurements']
    return (all(m['finite'] for m in ms)
            and result['maximum_joint_overshoot_rad'] < .08
            and result['max_sampled_actuator_force_nm'] <= 3.23+1e-8
            and result['max_command_slew_rad_s'] <= 5.24+1e-8
            and max(m['self_penetration_m'] for m in ms) < .004
            and max(m['floor_penetration_m'] for m in ms) < .01)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--width', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=197)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    qualification = json.loads((scene.parent/'qualification.json').read_text())
    if qualification['standing_passed_runs'] != qualification['standing_runs']:
        raise RuntimeError('collision model standing qualification failed')
    sim = SupportSim(scene, stand)
    initial = sim.prepare('prone')
    beam = [dict(state=initial, path=[], score=0., metric=sim.measure())]
    rng, history, started = np.random.default_rng(args.seed), [], time.time()
    best = beam[0]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand))) as pool:
        for depth in range(args.depth):
            jobs, paths = [], []
            for parent in beam:
                targets = anatomical_library(sim, parent['state']['prev'], rng)
                for target in targets:
                    for duration in (.28, .70):
                        jobs.append((parent['state'], target, duration))
                        paths.append(parent['path']+[(target.copy(), duration)])
            evaluated = list(pool.map(evaluate, jobs, chunksize=4))
            candidates = []
            for i, result in enumerate(evaluated):
                if safe(result):
                    result['path'] = paths[i]
                    candidates.append(result)
            candidates.sort(key=lambda r: -r['score'])
            if not candidates:
                raise RuntimeError('no safe anatomical search nodes')
            beam, bins = [], set()
            for c in candidates:
                m, q = c['metric'], c['state']['qpos'][3:7]
                key = (*np.round(q/.13).astype(int), round(m['height_m']/.015),
                       *np.round(c['state']['prev'][[2, 3, 4, 5]]/.35).astype(int))
                if key not in bins:
                    beam.append(c)
                    bins.add(key)
                if len(beam) == args.width:
                    break
            best = beam[0]
            np.savez_compressed(args.output/'best_reference.npz', targets=[t for t,d in best['path']],
                                durations_s=[d for t,d in best['path']])
            row = dict(depth=depth+1, expanded=len(jobs), safe_nodes=len(candidates),
                       score=best['score'], metric=best['metric'],
                       ready_nodes=sum(supported_entry(c['metric']) for c in candidates),
                       wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            brief = {k:v for k,v in row.items() if k != 'metric'}
            brief['metric'] = {k:best['metric'][k] for k in ('height_m','up_z','feet','foot_up_alignment_to_home','joint_home_error_max_rad')}
            print(json.dumps(brief), flush=True)
            if supported_entry(best['metric']):
                break
    dynamic.ready_to_handoff = supported_entry
    rows = []
    for seed in range(20):
        result, trace = dynamic.replay(sim, 'prone', best['path'], 21000+seed, seed>0, seed==0)
        rows.append(result)
        print('VALIDATING',seed,result['success'],flush=True)
        if trace:
            np.savez_compressed(args.output/'best_trajectory.npz', time=[r[0] for r in trace],qpos=[r[1] for r in trace],
                                qvel=[r[2] for r in trace],ctrl=[r[3] for r in trace],phase=[r[4] for r in trace])
    summary = dict(method='normal-knee target curriculum + foot-orientation shaping', pose='prone',
                   successful_validation_runs=sum(r['success'] for r in rows), validation_runs=len(rows), results=rows,
                   simulation_only=True, hardware_readiness=False, root_pose_edits_after_initialization=0,
                   scene_path=str(scene), seed=args.seed, depth=args.depth, width=args.width,
                   knee_target_curriculum_rad=[0.,float(sim.upper[3])],
                   hashes={str(f):digest(f) for f in (Path(__file__), Path(__file__).with_name('audit_getup_crouch.py'),scene,stand)})
    (args.output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('COMPLETE:',summary['successful_validation_runs'],'/20',flush=True)


if __name__ == '__main__':
    main()
