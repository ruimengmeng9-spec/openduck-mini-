"""Read-only approximate LIPM capture point relative to the two foot centers.

Separate mjData avoids refreshing/changing the live simulator's sensor/contact
buffers. This is a simulation-state diagnostic, not a real-robot estimator.
"""
import math
import mujoco
import numpy as np


class CapturePointObserver:
    def __init__(self,model):
        self.model=model
        self.data=mujoco.MjData(model)
        self.sites=[model.site(name).id for name in ('left_foot','right_foot')]
        free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        if len(free)!=1 or model.body_subtreemass[0]<=0:
            raise ValueError('one floating robot required')
        self.qaddr=int(model.jnt_qposadr[free[0]])

    def forward_offset(self,live):
        d=self.data
        d.qpos[:]=live.qpos
        d.qvel[:]=live.qvel
        mujoco.mj_kinematics(self.model,d)
        mujoco.mj_comPos(self.model,d)
        mujoco.mj_comVel(self.model,d)
        mujoco.mj_subtreeVel(self.model,d)
        com=d.subtree_com[0]
        vel=d.subtree_linvel[0]
        omega=math.sqrt(abs(float(self.model.opt.gravity[2]))/float(np.clip(com[2],.08,.4)))
        if not omega>0:
            raise ValueError('capture observer requires gravity')
        cp=com[:2]+vel[:2]/omega
        center=d.site_xpos[self.sites,:2].mean(axis=0)
        w,x,y,z=d.qpos[self.qaddr+3:self.qaddr+7]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        offset=cp-center
        return float(math.cos(yaw)*offset[0]+math.sin(yaw)*offset[1])
