"""Sensor-only velocity-dependent inertial bias, not contact dynamics control."""
import mujoco
import numpy as np


class VelocityBias:
    """Private canonical root; actual root, upvector and contact never enter."""
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = model.actuator_trnid[:, 0]
        if model.nu != 14 or len(np.unique(joints)) != 14:
            raise ValueError('Original 14 distinct servo joints required')
        self.qadr = model.jnt_qposadr[joints].copy()
        self.vadr = model.jnt_dofadr[joints].copy()
        self.home = model.keyframe('home').qpos[self.qadr].copy()
        self.canonical = model.keyframe('home').qpos.copy()
        roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(roots) != 1 or model.nv != 20:
            raise ValueError('One free root and original 14 hinge joints required')
        root = int(roots[0]); qa = int(model.jnt_qposadr[root]); va = int(model.jnt_dofadr[root])
        self.rootq = slice(qa, qa + 7)
        self.rootlinear = np.arange(va, va + 3)
        self.rootangular = np.arange(va + 3, va + 6)
        self.canonical[self.rootq] = [0, 0, 0, 1, 0, 0, 0]
        self.trunk = model.body('trunk_assembly').id
        gyro = model.sensor('gyro').id
        if model.sensor_type[gyro] != mujoco.mjtSensor.mjSENS_GYRO:
            raise ValueError('Original site gyro required')
        self.site = int(model.sensor_objid[gyro])
        self.gyrobody = int(model.site_bodyid[self.site])
        if model.body_weldid[self.gyrobody] != model.body_weldid[self.trunk]:
            raise ValueError('Original gyro rigid weld required')
        self.legs = np.array([model.actuator(side + '_' + part).id for side in ('left', 'right')
                              for part in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')])
        self.head = np.array([i for i in range(14) if i not in self.legs])
        self.kp = model.actuator_gainprm[:, 0].copy()
        if (np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
                or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
                or np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)
                or not np.isfinite(self.kp).all() or np.any(self.kp <= 0)
                or not np.array_equal(model.actuator_biasprm[:, 0], np.zeros(14))
                or not np.array_equal(model.actuator_biasprm[:, 1], -self.kp)
                or not np.array_equal(model.actuator_gear, np.tile([1., 0, 0, 0, 0, 0], (14, 1)))):
            raise ValueError('Original unit-gear affine position servos required')

    def reconstruct(self, sensor34):
        x = np.asarray(sensor34, dtype=np.float32)
        if x.shape != (34,) or not np.isfinite(x).all():
            raise ValueError('Finite causal native34 required')
        d, m = self.data, self.model
        d.qpos[:] = self.canonical
        d.qpos[self.qadr] = self.home + x[6:20].astype(float)
        d.qvel[:] = 0.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, None, jr, self.site)
        d.qvel[self.vadr] = x[20:34].astype(float) / .05
        d.qvel[self.rootangular] = np.linalg.solve(
            jr[:, self.rootangular], d.site_xmat[self.site].reshape(3, 3) @ x[:3].astype(float)
            - jr[:, self.vadr] @ d.qvel[self.vadr])
        return d.qvel.copy()

    def measure(self, sensor34):
        velocity = self.reconstruct(sensor34)
        d, m = self.data, self.model
        moving = np.empty(m.nv); static = np.empty(m.nv)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, moving)
        # Same pose/model/gravity, but zero velocity. No model flags or forces
        # are changed. Subtraction removes gravity, not external contacts.
        d.qvel[:] = 0.
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, static)
        d.qvel[:] = velocity
        mujoco.mj_comVel(m, d)
        return (moving - static)[self.vadr].copy(), moving[self.vadr].copy(), static[self.vadr].copy(), velocity


def velocity_bias_request(current, nominal, initial, initial_nominal, geometry, control, enabled):
    """Fixed feedforward hypothesis; no torque write, fit, pinv or scale search.

    e is N m. Positive e/kp is a position request (rad) that would add e to
    the unclipped affine servo torque at the SAME state. Reference/joint/slew
    and actuator limits can change it; no realized dynamics claim is made.
    """
    if isinstance(control, (bool, np.bool_)) or not isinstance(control, (int, np.integer)) or control < 0:
        raise ValueError('Nonnegative integer original phase required')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('Fixed boolean experiment switch required')
    xs = [np.asarray(v, dtype=np.float32) for v in (current, nominal, initial, initial_nominal)]
    if any(v.shape != (34,) or not np.isfinite(v).all() for v in xs):
        raise ValueError('Finite current/causal initial native34 required')
    if not enabled or control >= 529:
        return np.zeros(14), np.zeros(14), np.zeros((4, 14))
    biases = np.stack([geometry.measure(x)[0] for x in xs])
    error = (biases[0] - biases[1]) - (biases[2] - biases[3])
    request = np.zeros(14)
    request[geometry.legs] = error[geometry.legs] / geometry.kp[geometry.legs]
    return request, error, biases
