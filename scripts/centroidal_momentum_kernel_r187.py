"""Causal model angular-momentum geometry, not a dynamics predictor/controller."""
import mujoco
import numpy as np

DT = .02
PINV_RCOND = 1e-10


class MomentumGeometry:
    """A private canonical root; never accepts an episode or actual root state."""
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = model.actuator_trnid[:, 0]
        self.qadr = model.jnt_qposadr[joints].copy()
        self.vadr = model.jnt_dofadr[joints].copy()
        if len(joints) != 14 or len(np.unique(self.vadr)) != 14:
            raise ValueError('Original 14 distinct actuator joints required')
        self.home = model.keyframe('home').qpos[self.qadr].copy()
        self.canonical = model.keyframe('home').qpos.copy()
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError('Exactly one original free root required')
        j = int(free[0]); qa = int(model.jnt_qposadr[j]); va = int(model.jnt_dofadr[j])
        self.rootq = slice(qa, qa + 7)
        self.canonical[self.rootq] = [0, 0, 0, 1, 0, 0, 0]
        self.rootlinear = np.arange(va, va + 3)
        self.rootangular = np.arange(va + 3, va + 6)
        self.rootbody = int(model.jnt_bodyid[j])
        self.trunk = model.body('trunk_assembly').id
        gyro = model.sensor('gyro').id
        if model.sensor_type[gyro] != mujoco.mjtSensor.mjSENS_GYRO:
            raise ValueError('Original gyro sensor required')
        self.site = int(model.sensor_objid[gyro])
        if int(model.site_bodyid[self.site]) != self.trunk:
            raise ValueError('Gyro must be rigidly attached to original trunk')
        self.legs = np.array([model.actuator(side + '_' + part).id
                              for side in ('left', 'right')
                              for part in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')])
        self.head = np.array([i for i in range(14) if i not in self.legs])
        np.testing.assert_array_equal(self.legs, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13])

    def measure(self, sensor34):
        x = np.asarray(sensor34, dtype=np.float32)
        if x.shape != (34,) or not np.isfinite(x).all():
            raise ValueError('Finite actual native34 required')
        # Canonical geometry includes all measured joints: head motion also
        # changes whole-robot inertia. None of these values is a root truth.
        d, m = self.data, self.model
        d.qpos[:] = self.canonical
        d.qpos[self.qadr] = self.home + x[6:20].astype(float)
        d.qvel[:] = 0.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        matrix = np.zeros((3, m.nv))
        mujoco.mj_angmomMat(m, d, matrix, self.rootbody)
        jr = np.zeros_like(matrix)
        mujoco.mj_jacSite(m, d, None, jr, self.site)
        velocity = np.zeros(m.nv)
        velocity[self.vadr] = x[20:34].astype(float) / .05
        site_rotation = d.site_xmat[self.site].reshape(3, 3)
        # Recover free-joint angular coordinates from measured site gyro,
        # including any joint contribution, instead of assuming world axes.
        velocity[self.rootangular] = np.linalg.solve(
            jr[:, self.rootangular], site_rotation @ x[:3].astype(float)
            - jr[:, self.vadr] @ velocity[self.vadr])
        trunk_rotation = d.xmat[self.trunk].reshape(3, 3)
        body_matrix = trunk_rotation.T @ matrix
        momentum = body_matrix @ velocity
        return momentum, body_matrix[:, self.vadr[self.legs]].copy(), body_matrix, velocity


def momentum_request(current, nominal, initial, initial_nominal, geometry, control, enabled):
    """Instantaneous kinematic allocation, no contact/dynamics/safety prediction.

    Momentum units kg m^2/s; matrix units kg m^2; the virtual joint velocity
    allocation is converted to a position request by the original .02 s step.
    Original joint/reference/slew limits must be applied by a future wrapper.
    """
    if not isinstance(control, (int, np.integer)) or control < 0:
        raise ValueError('Nonnegative integer original phase required')
    xs = [np.asarray(v, dtype=np.float32) for v in (current, nominal, initial, initial_nominal)]
    if any(v.shape != (34,) or not np.isfinite(v).all() for v in xs):
        raise ValueError('Finite current and causal initial native34 required')
    if not enabled or control >= 529:
        return np.zeros(14), np.zeros(3), np.zeros((3, 10)), np.zeros(3)
    x, n, i, z = xs
    hm, allocation, _, _ = geometry.measure(x)
    hn = geometry.measure(n)[0]
    hi = geometry.measure(i)[0]
    hz = geometry.measure(z)[0]
    error = (hm - hn) - (hi - hz)
    request = np.zeros(14)
    if np.any(error != 0.):
        request[geometry.legs] = -DT * np.linalg.pinv(allocation, rcond=PINV_RCOND) @ error
    residual = error + allocation @ (request[geometry.legs] / DT)
    return request, error, allocation, residual
