"""Mixed episode-start curriculum: audited prefixes plus actual full falls."""
import os, traceback
import mujoco
import numpy as np
from diagnostics.getup_fullfallen_env_r32 import FullFallEpisode, observation
from diagnostics.train_getup_fullpath_r27 import state_hash

class MixedEpisode(FullFallEpisode):
    def __init__(self, scene, stand, seed, library, prefix_probability=0.5, hold_controls=5):
        self.library_qpos = np.load(library, allow_pickle=False)['qpos']
        self.library_qvel = np.load(library, allow_pickle=False)['qvel']
        self.library_offsets = np.load(library, allow_pickle=False)['offsets']
        self.prefix_probability = prefix_probability
        super().__init__(scene, stand, seed, hold_controls)

    def reset(self, pose_override=None):
        if self.rng.random() < self.prefix_probability:
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
            self.initial_valid = self.sim.physical_valid()
            if not self.initial_valid:
                raise RuntimeError(f'invalid prefix state {i}')
            self.initial_hash = state_hash(self.sim)
            self.sim.clear_audit()
            self.controls = self.tail_controls = 0
            self.return_sum = 0.0
            self.previous_phi = self.phi(self.initial)
            return observation(self.sim)
        return super().reset(pose_override)

    def step(self, normalized):
        row = list(super().step(normalized))
        if row[5] is not None:
            row[5]['mixed_prefix_start'] = self.pose == 'prefix_library'
            row[5]['curriculum_stage'] = 'mixed_prefix_fullfall'
        return tuple(row)

def pipe_worker(connection, scene, stand, number, seed, library, prefix_probability):
    os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        envs=[MixedEpisode(scene,stand,seed+i,library,prefix_probability) for i in range(number)]
        connection.send(('ready',np.stack([observation(e.sim) for e in envs])))
        while True:
            op,payload=connection.recv()
            if op=='close': break
            if op!='step': raise ValueError(op)
            connection.send(('ok',[e.step(a) for e,a in zip(envs,payload)]))
    except Exception as exc:
        connection.send(('error',type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally: connection.close()
