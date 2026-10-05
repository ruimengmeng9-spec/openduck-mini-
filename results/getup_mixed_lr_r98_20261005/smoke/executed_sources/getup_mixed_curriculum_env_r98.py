"""R98 episode-start curriculum, with provenance-safe terminal labels.

Prefix states teach only late-stage recovery, never full-fall acceptance.
Original FullFallEpisode dynamics, targets, rewards and audits are unchanged.
"""
import os
import traceback

import mujoco
import numpy as np

from diagnostics.getup_fullfallen_env_r32 import FullFallEpisode, observation
from diagnostics.train_getup_fullpath_r27 import state_hash


def label_completed_episode(info):
    """Base step resets the environment before returning: use completed info."""
    prefix = info['pose'] == 'prefix_library'
    info['mixed_prefix_start'] = prefix
    info['actual_fallen_start'] = bool(not prefix and info['actual_fallen_start'])
    info['curriculum_stage'] = 'r98_mixed_prefix_fullfall'
    return info


class MixedEpisode(FullFallEpisode):
    def __init__(self, scene, stand, seed, library, prefix_probability=.5,
                 hold_controls=5):
        if not 0. <= prefix_probability <= 1.:
            raise ValueError('Probability outside [0,1]')
        with np.load(library, allow_pickle=False) as data:
            self.library_qpos = data['qpos'].copy()
            self.library_qvel = data['qvel'].copy()
            self.library_offsets = data['offsets'].copy()
        if not len(self.library_qpos) or not (
                len(self.library_qpos) == len(self.library_qvel) == len(self.library_offsets)):
            raise ValueError('Empty or mismatched state library')
        self.prefix_probability = prefix_probability
        super().__init__(scene, stand, seed, hold_controls)

    def reset(self, pose_override=None):
        # Explicit diagnostic pose always means a real pose, not a prefix.
        if pose_override is None and self.rng.random() < self.prefix_probability:
            i = int(self.rng.integers(len(self.library_qpos)))
            self.sim.clear_audit()
            mujoco.mj_resetData(self.sim.model, self.sim.data)
            self.sim.data.qpos[:] = self.library_qpos[i]
            self.sim.data.qvel[:] = self.library_qvel[i]
            self.sim.prev = self.sim.home.copy()
            self.sim.history = np.zeros_like(self.sim.history)
            self.sim.data.ctrl[:] = self.sim.prev
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self.pose = 'prefix_library'
            self.reset_seed = i
            self.fraction = float(self.library_offsets[i])
            self.initial = self.sim.measure()
            if not self.sim.physical_valid():
                raise RuntimeError(f'Invalid training prefix {i}')
            self.initial_hash = state_hash(self.sim)
            self.sim.clear_audit()
            self.controls = self.tail_controls = 0
            self.return_sum = 0.
            self.previous_phi = self.phi(self.initial)
            return observation(self.sim)
        return super().reset(pose_override)

    def step(self, normalized):
        row = list(super().step(normalized))
        if row[5] is not None:
            row[5] = label_completed_episode(row[5])
        return tuple(row)


def pipe_worker(connection, scene, stand, number, seed, library, probability):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        envs = [MixedEpisode(scene, stand, seed+i, library, probability)
                for i in range(number)]
        connection.send(('ready', np.stack([observation(e.sim) for e in envs])))
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            if operation != 'step':
                raise ValueError(operation)
            connection.send(('ok', [e.step(a) for e,a in zip(envs,payload)]))
    except Exception as exc:
        connection.send(('error', type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally:
        connection.close()
