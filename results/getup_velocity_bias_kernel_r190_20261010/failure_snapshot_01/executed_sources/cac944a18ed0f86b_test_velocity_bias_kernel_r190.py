"""Offline algebra and causal interface, never a recovery episode."""
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from diagnostics import velocity_bias_kernel_r190 as run
from diagnostics import probe_getup_centroidal_momentum_r188b as old


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def independent_velocity_bias(m, d):
    """Body spatial recursion with zero gravity, independently assembled I.

    Uses kinematic comPos/comVel fields, not mj_rne or saved kernel output.
    Array arithmetic intentionally is not claimed bitwise equal to C.
    """
    acceleration = np.zeros((m.nbody, 6)); forces = np.zeros_like(acceleration)
    for b in range(1, m.nbody):
        adr = int(m.body_dofadr[b]); count = int(m.body_dofnum[b])
        acceleration[b] = acceleration[m.body_parentid[b]]
        if count:
            acceleration[b] += d.cdof_dot[adr:adr+count].T @ d.qvel[adr:adr+count]
        r = d.xipos[b] - d.subtree_com[m.body_rootid[b]]
        rot = d.ximat[b].reshape(3, 3); mass = m.body_mass[b]
        inertia = rot @ np.diag(m.body_inertia[b]) @ rot.T + mass * (np.dot(r, r)*np.eye(3)-np.outer(r, r))
        first = skew(mass*r)
        spatial = np.block([[inertia, first], [-first, mass*np.eye(3)]])
        momentum = spatial @ d.cvel[b]
        w, v = d.cvel[b, :3], d.cvel[b, 3:]
        transport = np.r_[np.cross(w, momentum[:3])+np.cross(v, momentum[3:]), np.cross(w, momentum[3:])]
        forces[b] = spatial @ acceleration[b] + transport
    for b in range(m.nbody-1, 0, -1):
        parent = int(m.body_parentid[b])
        if parent: forces[parent] += forces[b]
    return np.sum(d.cdof * forces[m.dof_bodyid], axis=1)


class BiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(old.local.prior.program.SCENE))
        cls.model.opt.timestep = .002
        with np.load(old.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz', allow_pickle=False) as z:
            cls.nominal = z['observations'][:529, :34].copy()
        cls.proof = {}; cls.rng = np.random.default_rng(290)
        destination = os.environ.get('OPEN_DUCK_R190_EXECUTED_SOURCES')
        if destination:
            manifest = capture_all_sources(Path(destination))
            Path(destination).parent.joinpath('test_source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def setUp(self):
        self.g = run.VelocityBias(self.model); self.x = np.zeros(34, np.float32)

    @classmethod
    def tearDownClass(cls):
        destination = os.environ.get('OPEN_DUCK_R190_PROOF')
        if destination: Path(destination).write_text(json.dumps(cls.proof, indent=2)+'\n')

    def call(self, x=None, n=None, i=None, z=None, k=1, enabled=True):
        return run.velocity_bias_request(self.x if x is None else x, self.x if n is None else n,
            self.x if i is None else i, self.x if z is None else z, self.g, k, enabled)

    def sample(self):
        return self.rng.uniform(-.3, .3, 34).astype(np.float32)

    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(run.velocity_bias_request).parameters),
            ['current','nominal','initial','initial_nominal','geometry','control','enabled'])
        self.assertEqual(list(inspect.signature(run.VelocityBias.measure).parameters), ['self','sensor34'])

    def test_independent_spatial_body_recursion(self):
        errors=[]
        for _ in range(12):
            x=self.sample(); measured=self.g.measure(x)[0]
            independent=independent_velocity_bias(self.model,self.g.data)[self.g.vadr]
            np.testing.assert_allclose(measured,independent,atol=1e-14,rtol=3e-12)
            errors.append(float(np.max(np.abs(measured-independent))))
        self.proof['spatial_recursion_12_max_error_nm']=max(errors)

    def test_static_zero_and_gravity_subtraction(self):
        x=self.sample(); x[:3]=0; x[20:34]=0
        bias,moving,static,_=self.g.measure(x)
        np.testing.assert_array_equal(bias,np.zeros(14))
        np.testing.assert_array_equal(moving,static)
        self.assertGreater(np.max(np.abs(static)),.001)

    def test_quadratic_velocity_scaling(self):
        x=self.sample(); y=x.copy(); y[:3]*=2; y[20:34]*=2
        np.testing.assert_allclose(self.g.measure(y)[0],4*self.g.measure(x)[0],atol=1e-14,rtol=3e-12)

    def test_velocity_reversal_even(self):
        x=self.sample(); y=x.copy(); y[:3]*=-1; y[20:34]*=-1
        np.testing.assert_array_equal(self.g.measure(y)[0],self.g.measure(x)[0])

    def test_gyro_site_coordinates(self):
        x=self.sample(); velocity=self.g.measure(x)[3]
        jr=np.zeros((3,self.model.nv)); mujoco.mj_jacSite(self.model,self.g.data,None,jr,self.g.site)
        sensor=self.g.data.site_xmat[self.g.site].reshape(3,3).T @ (jr @ velocity)
        np.testing.assert_allclose(sensor,x[:3].astype(float),atol=1e-15,rtol=0)

    def test_gyro_actual_weld_not_same_body(self):
        self.assertNotEqual(self.g.gyrobody,self.g.trunk)
        self.assertEqual(self.model.body_weldid[self.g.gyrobody],self.model.body_weldid[self.g.trunk])
        self.g.measure(self.x); jr=np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.g.data,None,jr,self.g.site)
        np.testing.assert_array_equal(jr[:,self.g.vadr],np.zeros((3,14)))

    def test_native_units(self):
        x=self.sample(); velocity=self.g.measure(x)[3]
        np.testing.assert_array_equal(velocity[self.g.vadr],x[20:34].astype(float)/.05)

    def test_all_529_actual_nominal_scalar_zero(self):
        for k,n in enumerate(self.nominal):
            raw,e,_=self.call(n,n,self.nominal[0],self.nominal[0],k)
            np.testing.assert_array_equal(raw,np.zeros(14)); np.testing.assert_array_equal(e,np.zeros(14))

    def test_causal_initial_control_zero(self):
        x=self.sample(); raw,e,_=self.call(x,self.x,x,self.x,0)
        np.testing.assert_array_equal(raw,np.zeros(14)); np.testing.assert_array_equal(e,np.zeros(14))

    def test_upvector_unused(self):
        x=self.sample(); y=x.copy(); y[3:6]=[9,8,7]
        for a,b in zip(self.call(x),self.call(y)): np.testing.assert_array_equal(a,b)

    def test_head_inertia_without_new_head_commands(self):
        x=self.sample(); a=self.g.measure(x)[0]; x[20+self.g.head]+=.1
        b=self.g.measure(x)[0]
        self.assertGreater(np.max(np.abs(a[self.g.legs]-b[self.g.legs])),1e-8)
        np.testing.assert_array_equal(self.call(x)[0][self.g.head],np.zeros(4))

    def test_actual_affine_servo_positive_torque_increment(self):
        x=self.sample(); raw,e,_=self.call(x)
        np.testing.assert_array_equal(raw[self.g.legs],e[self.g.legs]/self.g.kp[self.g.legs])
        # SAME state affine torque algebra, explicitly before torque clipping.
        target=self.sample()[:14].astype(float); q=self.g.home+x[6:20].astype(float)
        velocity=x[20:34].astype(float)/.05; bias=self.model.actuator_biasprm
        base=self.g.kp*target+bias[:,0]+bias[:,1]*q+bias[:,2]*velocity
        new=self.g.kp*(target+raw)+bias[:,0]+bias[:,1]*q+bias[:,2]*velocity
        np.testing.assert_allclose((new-base)[self.g.legs],e[self.g.legs],atol=1e-14,rtol=3e-12)
        self.proof['actual_kp_nm_per_rad']=self.g.kp.tolist()

    def test_home_and_disabled_zero(self):
        for kwargs in ({'k':529},{'enabled':False}):
            for value in self.call(self.sample(),**kwargs): np.testing.assert_array_equal(value,np.zeros_like(value))

    def test_original_merge_zero_object_and_limits(self):
        target=np.zeros(14); raw=self.call(self.sample())[0]
        zero=old.local.merge_target(target,target,np.zeros(10),self.g.legs,-np.ones(14),np.ones(14))
        self.assertIs(zero,target)
        raw[self.g.legs]*=1000
        merged=old.local.merge_target(target,target,raw[self.g.legs],self.g.legs,-np.ones(14),np.ones(14))
        self.assertLessEqual(np.abs(merged).max(),.18+1e-12)
        np.testing.assert_array_equal(merged[self.g.head],target[self.g.head])
        planned=old.planner.apply_limits(merged,target,-np.ones(14),np.ones(14))
        self.assertLessEqual(np.abs(planned-target).max(),5.24*.02+1e-12)

    def test_private_no_forward_contact_or_integration(self):
        forbidden=('mj_forward','mj_step','mj_inverse','mj_fwdPosition','mj_fwdVelocity',
                   'mj_contactForce','mj_collision','mj_rnePostConstraint','mj_passive','mj_fwdActuation')
        with ExitStack() as stack:
            for name in forbidden: stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError(name)))
            self.call(self.sample())
        self.assertEqual(self.g.data.time,0)
        np.testing.assert_array_equal(self.g.data.qpos[self.g.rootq],[0,0,0,1,0,0,0])
        np.testing.assert_array_equal(self.g.data.qvel[self.g.rootlinear],np.zeros(3))
        np.testing.assert_array_equal(self.g.data.ctrl,np.zeros(14))

    def test_full_compiled_model_immutable(self):
        before=old.model_digest(self.model); self.call(self.sample())
        self.assertEqual(before,old.model_digest(self.model))
        self.proof['compiled_model_sha256']=before

    def test_global_rotation_and_translation_invariance(self):
        x=self.sample(); expected=self.g.measure(x)[0]
        d=self.g.data; m=self.model
        d.qpos[self.g.rootq]=[.7,-.2,.4,.5,.5,.5,.5]
        mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
        jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,None,jr,self.g.site)
        d.qvel[:]=0; d.qvel[self.g.vadr]=x[20:34].astype(float)/.05
        d.qvel[self.g.rootangular]=np.linalg.solve(jr[:,self.g.rootangular],d.site_xmat[self.g.site].reshape(3,3)@x[:3].astype(float)-jr[:,self.g.vadr]@d.qvel[self.g.vadr])
        mujoco.mj_comVel(m,d); moving=np.empty(m.nv); mujoco.mj_rne(m,d,0,moving)
        d.qvel[:]=0; mujoco.mj_comVel(m,d); static=np.empty(m.nv); mujoco.mj_rne(m,d,0,static)
        np.testing.assert_allclose((moving-static)[self.g.vadr],expected,atol=1e-14,rtol=3e-12)

    def test_invalid_inputs(self):
        for x in (np.zeros(33),np.full(34,np.nan)):
            with self.assertRaises(ValueError): self.call(x)
        for k in (-1,1.5,True):
            with self.assertRaises(ValueError): self.call(k=k)
        with self.assertRaises(ValueError): self.call(enabled=1)


def capture_all_sources(destination):
    import hashlib
    import shutil
    import sys
    destination.mkdir(exist_ok=False)
    paths={Path(__file__).resolve(),Path(run.__file__).resolve()}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).suffix=='.py' and Path(name).is_file(): paths.add(Path(name).resolve())
    manifest={}
    for path in sorted(paths):
        sha=old.prior.digest(path); name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name
        shutil.copy2(path,destination/name); assert old.prior.digest(destination/name)==sha
        manifest[str(path)]={'sha256':sha,'copy':name}
    return manifest


if __name__=='__main__': unittest.main(verbosity=2)
