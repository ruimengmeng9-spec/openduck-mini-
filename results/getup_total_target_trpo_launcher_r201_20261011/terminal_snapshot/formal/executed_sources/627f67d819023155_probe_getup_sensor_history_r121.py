"""Causal preparation sensor histories with unchanged full teacher replay.

Only native 50D sensor observations become context. True simulator state is
saved separately for parity audits, never used as a policy input.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import search_getup_case_teachers_r109 as search
from diagnostics import train_getup_program_r113 as program
from diagnostics.search_getup_expanded_teachers_r115 import OUTPUT as TEACHERS
from diagnostics.getup_reference_env_r100 import ReferenceEpisode, native_observation
from diagnostics.train_getup_joint_anchor_r102 import write_json
from diagnostics.getup_independent_native import digest

ROOT = program.ROOT
OUTPUT = ROOT / 'outputs/getup_sensor_history_r121_left_20261005'
FORMER = ROOT / 'outputs/getup_expanded_program_r116_left_20261005'
KNOWN = {None, *program.TRAIN, *range(3160000,3160040), *range(3180000,3180040)}


def record_preparation(sim, prepare, observe=native_observation):
    original = sim.step_target
    frames, times = [], []
    def recorded(target):
        value = original(target)
        frame = np.asarray(observe(sim), dtype=np.float32)
        if frame.shape != (50,) or not np.isfinite(frame).all():
            raise ValueError('Only original finite native50 sensors allowed')
        frames.append(frame.copy())
        times.append(float(sim.data.time))
        return value
    sim.step_target = recorded
    try:
        result = prepare()
    finally:
        sim.step_target = original
    if len(frames) != 40:
        raise ValueError('Original 40 home controls must remain unchanged')
    return result, np.stack(frames), np.asarray(times)


class HistoryReferenceEpisode(ReferenceEpisode):
    def reset(self, seed_override='sample'):
        result, frames, times = record_preparation(self.sim,
            lambda: super(HistoryReferenceEpisode, self).reset(seed_override))
        self.preparation_sensors, self.preparation_times = frames, times
        return result


def init_worker(anchors, gains):
    search.init_worker(anchors, gains)


def rollout(job):
    case, baseline, teacher, directory = job
    if case not in KNOWN:
        raise ValueError('Unseen qualification states must not be loaded')
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=False)
    args = (str(program.SCENE), str(program.STAND), program.REFERENCE, 221, True)
    plain, observed = ReferenceEpisode(*args), HistoryReferenceEpisode(*args)
    plain.reset(case)
    observed.reset(case)
    a, b = plain.sim.snapshot(), observed.sim.snapshot()
    state_equal = {k: bool(np.array_equal(a[k],b[k])) for k in a}
    state_equal['ctrl'] = bool(np.array_equal(plain.sim.data.ctrl,observed.sim.data.ctrl))
    state_equal['rng'] = plain.rng.bit_generator.state == observed.rng.bit_generator.state
    np.savez_compressed(dest / 'audit_initial_state.npz',
        **{'plain_'+k: v for k,v in a.items()}, **{'observed_'+k:v for k,v in b.items()})
    np.savez_compressed(dest / 'sensor_context.npz', history=observed.preparation_sensors,
        times=observed.preparation_times, initial_observation=observed.observe())
    parity = dict(case_seed=case, initial_hash=observed.initial_hash,
        same_original_initial_hash=plain.initial_hash==observed.initial_hash==baseline['initial_hash'],
        state_equal=state_equal, causal_preparation_samples=40, sensor_dimension=50,
        simulator_truth_separate_audit_only=True, independent_qualification_trial=False)
    write_json(dest / 'initial_parity.json',parity)
    assert parity['same_original_initial_hash'] and all(state_equal.values())
    np.testing.assert_array_equal(observed.preparation_sensors[-1], observed.observe()[:50])
    np.testing.assert_allclose(np.diff(observed.preparation_times),.02,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(plain.observe(), observed.observe())
    if teacher:
        with np.load(TEACHERS / f'teachers/case_{case}/teacher_parameters.npz') as data:
            profile, knots = int(data['profile']), data['knots'].copy()
        records=[]
        while True:
            obs=observed.observe(); k=observed.controls
            base=np.zeros(10) if profile==0 else search.r102.action_for(
                search.r102.WEIGHTS,obs,search.r102.ANCHORS[k],search.GAINS)
            action=np.clip(base+search.knot_action(knots,k),-1.,1.)
            step=observed.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),observed.sim.data.time,
                observed.sim.data.qpos.copy(),observed.sim.data.qvel.copy(),observed.sim.prev.copy(),observed.tail>0))
            if step[2]:break
        arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],
            time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],
            applied=[r[5] for r in records],strict=[r[6] for r in records])
        np.savez_compressed(dest / 'trajectory.npz',**arrays)
        row=step[5]
        row.update(full_path=True,success=bool(row['valid'] and row['controls']==2279
            and row['entry_time_s'] is not None and row['entry_time_s']<=12.
            and row['strict_tail_s']>=30.-1e-8))
        with np.load(TEACHERS / f'teachers/case_{case}/trajectory.npz') as previous:
            equal={key:bool(np.array_equal(np.asarray(value),previous[key])) for key,value in arrays.items()}
        row.update(full_original_teacher_bitwise_parity=all(equal.values()),trajectory_equal=equal)
        write_json(dest / 'teacher_replay.json',row)
        assert all(equal.values()) and row['success']
        parity.update(full_teacher_replayed=True,full_original_teacher_bitwise_parity=True,
            original_teacher_success_not_new_policy_success=True)
    else:
        parity.update(full_teacher_replayed=False,initial_history_only=True)
    write_json(dest / 'result.json',parity)
    return parity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    cases=[None,*program.TRAIN,*range(3160000,3160040)]
    rows=[json.loads((TEACHERS/f'teachers/case_{s}/result.json').read_text()) for s in cases]
    jobs=[(s,r,True,str(args.output/'cases'/f'case_{s}')) for s,r in zip(cases,rows)]
    if args.smoke:
        jobs=[j for j in jobs if j[0] in (None,769000,3160001)]
    else:
        previous=json.loads((FORMER/'results.json').read_text())['independent']['baseline']['rows']
        assert [r['case_seed'] for r in previous]==list(range(3180000,3180040))
        jobs.extend((r['case_seed'],r,False,str(args.output/'cases'/f'case_{r["case_seed"]}')) for r in previous)
    with np.load(TEACHERS/'frozen_profiles.npz') as data:
        anchors,gains=data['anchors'].copy(),data['gains'].copy()
    args.output.mkdir(exist_ok=False)
    shutil.copytree(ROOT/'outputs/getup_common_program_r120_left_20261005/executed_sources',args.output/'executed_sources')
    for name in (Path(__file__).name,'test_getup_sensor_history_r121.py','launch_getup_sensor_history_r121.py'):
        shutil.copy2(Path(__file__).with_name(name),args.output/'executed_sources'/name)
    write_json(args.output/'contract.json',dict(simulation_only=True,no_new_policy_training=True,
        hypothesis='Preparation sensor history may retain information omitted by a single snapshot; instrumentation first requires exact physical and trajectory parity',
        fixed_seed=221,workers=args.workers,smoke=args.smoke,cases=[j[0] for j in jobs],
        actual_fallen_start=True,preparation_controls=40,preparation_hz=50,sensor_dimension=50,
        full_teacher_replays=3 if args.smoke else 65,other_known_history_only=0 if args.smoke else 40,
        policy_inputs_exclude_true_root_position_velocity=True,no_case_or_seed_policy_input=True,
        true_state_only_separate_parity_audit=True,unseen_3200000_to_3200039_never_loaded=True,
        physics_rewards_acceptance_unchanged=True,no_extra_settling=True,no_mid_episode_root_reset=True,
        control_hz=50,physics_hz=500,full_path_controls=2279,entry_deadline_s=12,strict_tail_s=30,
        hardware_readiness=False,full_task_completed=False,
        hashes={str(f):digest(f) for f in [program.SCENE,program.STAND,program.REFERENCE,program.MODEL,
            TEACHERS/'results.json',FORMER/'results.json',*(args.output/'executed_sources').iterdir()]}))
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
        initargs=(anchors,gains)) as pool:
        reports=list(pool.map(rollout,jobs))
        contexts=[];histories=[];times=[]
        for j in jobs:
            with np.load(Path(j[3])/'sensor_context.npz') as data:
                contexts.append(data['initial_observation']);histories.append(data['history']);times.append(data['times'])
        # Only causal sensor arrays; case metadata is in the separate manifest.
        np.savez_compressed(args.output/'causal_sensor_dataset.npz',
            initial_observations=contexts,histories=histories,times=times)
        write_json(args.output/'manifest.json',reports)
        write_json(args.output/'results.json',dict(smoke=args.smoke,contexts_recorded=len(reports),
            initial_state_and_warmstart_bitwise_equal=True,
            full_original_teacher_replays=sum(r['full_teacher_replayed'] for r in reports),
            full_original_teacher_bitwise_equal=True,new_policy_success=False,
            independent_qualification_run=False,hardware_readiness=False,full_task_completed=False))
        print('R121_HISTORY_PARITY_PASS',len(reports),sum(r['full_teacher_replayed'] for r in reports),flush=True)
    print('R121_TERMINAL',flush=True)


if __name__=='__main__':main()
