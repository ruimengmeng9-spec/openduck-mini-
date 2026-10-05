"""Physical reset curriculum sampled from audited successful get-up prefixes."""
import os
import traceback

import mujoco
import numpy as np

from diagnostics.getup_fullfallen_env_r32 import FullFallEpisode, observation
from diagnostics.train_getup_fullpath_r27 import state_hash


class PrefixLibraryEpisode(FullFallEpisode):
    def __init__(self, scene, stand, seed, library, hold_controls=5):
        z = np.load(library, allow_pickle=False)
        self.library_qpos = np.asarray(z['qpos'])
        self.library_qvel = np.asarray(z['qvel'])
        self.library_offsets = np.asarray(z['offsets'])
        super().__init__(scene, stand, seed, hold_controls)

    def reset(self, pose_override=None):
        index = int(self.rng.integers(len(self.library_qpos)))
        mujoco.mj_resetData(self.sim.model, self.sim.data)
        self.sim.data.qpos[:] = self.library_qpos[index]
        self.sim.data.qvel[:] = self.library_qvel[index]
        self.sim.prev = self.sim.home.copy()
        self.sim.history = np.zeros_like(self.sim.history)
        self.sim.data.ctrl[:] = self.sim.prev
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.pose = 'prefix_library'
        self.reset_seed = index
        self.fraction = float(self.library_offsets[index])
        self.initial = self.sim.measure()
        self.initial_valid = self.sim.physical_valid()
        if not self.initial_valid or not np.isfinite(self.sim.data.qpos).all():
            raise RuntimeError(f'Invalid prefix state {index}: {self.initial}, {self.sim.peaks}')
        self.initial_hash = state_hash(self.sim)
        self.sim.clear_audit()
        self.controls = 0
        self.tail_controls = 0
        self.return_sum = 0.0
        self.previous_phi = self.phi(self.initial)
        return observation(self.sim)

    def step(self, normalized):
        row = list(super().step(normalized))
        if row[5] is not None:
            row[5]['prefix_offset_steps'] = self.fraction
            row[5]['actual_fallen_start'] = False
            row[5]['curriculum_stage'] = 'prefix_library'
        return tuple(row)


def pipe_worker(connection, scene, stand, number, seed, library):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        envs = [PrefixLibraryEpisode(scene, stand, seed + i, library)
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
