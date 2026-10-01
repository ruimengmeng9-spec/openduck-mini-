"""Progressively harder physical reset states for get-up PPO.

Only reset may set the root pose. Every subsequent transition uses the same
audited MuJoCo scene, joint limits, actuator limits and 50 Hz motor slew.
Stages 0-2 are training curricula; only stage 3 starts fully fallen and can
contribute evidence for the user's actual get-up objective.
"""
import os
import traceback

import numpy as np

from diagnostics.getup_fullfallen_env_r32 import FullFallEpisode, observation
from diagnostics.getup_independent_native import DT, POSES
from diagnostics.probe_getup_partial_curriculum_r35 import prepare_partial
from diagnostics.train_getup_fullpath_r27 import state_hash


# Fractions of the 90-degree canonical fallen orientation. These fractions
# were chosen from the measured R35 HOME-recovery boundaries, not by relabeling
# them as full falls. Stage 2 includes 20% actual full falls from the start.
FRACTIONS = {
    0: {'prone': (.05, .10, .15), 'supine': (.03, .05, .08),
        'left_side': (.10, .15, .20), 'right_side': (.10, .15, .20)},
    1: {'prone': (.10, .15, .20), 'supine': (.05, .08, .10),
        'left_side': (.15, .20, .25), 'right_side': (.15, .20, .25)},
    2: {'prone': (.15, .20, .35), 'supine': (.08, .10, .20),
        'left_side': (.20, .25, .35), 'right_side': (.20, .25, .35)},
}


class CurriculumEpisode(FullFallEpisode):
    def __init__(self, scene, stand, seed, stage, hold_controls=5):
        if stage not in (0, 1, 2, 3):
            raise ValueError('Curriculum stage must be 0, 1, 2, or 3')
        self.stage = stage
        super().__init__(scene, stand, seed, hold_controls)

    def reset(self, pose_override=None):
        self.pose = pose_override or POSES[int(self.rng.integers(4))]
        self.reset_seed = int(self.rng.integers(1000000))
        if self.stage == 3 or (self.stage == 2 and self.rng.random() < .20):
            self.fraction = 1.
            # A previous episode may have ended with an audited violation.
            # Clear its peaks BEFORE the 40 settling controls, so this reset
            # audits only the new physical trajectory, not the old episode.
            self.sim.clear_audit()
            self.sim.prepare(self.pose, self.reset_seed, True)
            self.initial = self.sim.measure()
            self.initial_valid = self.sim.physical_valid()
        else:
            self.fraction = float(self.rng.choice(FRACTIONS[self.stage][self.pose]))
            self.initial, _, self.initial_valid, _ = prepare_partial(
                self.sim, self.pose, self.fraction, 5, self.reset_seed)
        if not self.initial_valid:
            raise RuntimeError(
                f'Curriculum reset failed physical audit: pose={self.pose}, '
                f'fraction={self.fraction}, seed={self.reset_seed}, '
                f'peaks={self.sim.peaks}, initial={self.initial}')
        if self.fraction == 1. and not (
                self.initial['up_z'] < .5 and self.initial['torso_contact']):
            raise RuntimeError('A full-fallen reset lacks a fallen state')
        self.initial_hash = state_hash(self.sim)
        self.sim.clear_audit()
        self.controls = 0
        self.tail_controls = 0
        self.return_sum = 0.
        self.previous_phi = self.phi(self.initial)
        return observation(self.sim)

    def step(self, normalized):
        fraction = self.fraction
        row = list(super().step(normalized))
        if row[5] is not None:
            row[5]['tilt_fraction'] = fraction
            row[5]['actual_fallen_start'] = bool(fraction == 1.)
            row[5]['curriculum_stage'] = self.stage
        return tuple(row)


def pipe_worker(connection, scene, stand, number, seed, stage):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        envs = [CurriculumEpisode(scene, stand, seed + i, stage)
                for i in range(number)]
        connection.send(('ready', np.stack([observation(e.sim) for e in envs])))
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            if operation != 'step':
                raise ValueError('Unknown worker operation')
            connection.send(('ok', [e.step(a) for e, a in zip(envs, payload)]))
    except Exception as exc:
        connection.send(('error', type(exc).__name__ + ': ' + str(exc) + '\n'
                         + traceback.format_exc()))
    finally:
        connection.close()
