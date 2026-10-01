"""Kinematic replay around failure, without guessing cause from an endpoint."""
import json
from pathlib import Path
import mujoco
import numpy as np
import math
import argparse

root=Path('/data/shijinsheng/open_duck')
p=argparse.ArgumentParser()
p.add_argument('--folder',type=Path,default=root/'outputs/phase_feedback_r6_probe_heldout_30s')
p.add_argument('--seeds',default='15,18')
args=p.parse_args()
model=mujoco.MjModel.from_xml_path(str(root/'projects/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml'))
data=mujoco.MjData(model)
sites=[model.site(name).id for name in ['left_foot','right_foot']]
output=[]
for seed in [int(s) for s in args.seeds.split(',')]:
    trajectory=np.load(args.folder/f'seed_{seed}.npz',allow_pickle=False)
    times=trajectory['time']
    qpos=trajectory['qpos']
    rows=[]
    for i in range(max(0,len(times)-76),len(times),5):
        data.qpos[:]=qpos[i]
        mujoco.mj_forward(model,data)
        quat=qpos[i,3:7]
        up=1-2*(quat[1]**2+quat[2]**2)
        knees=[float(data.qpos[model.joint(n).qposadr][0]) for n in ['left_knee','right_knee']]
        w,x,y,z=quat
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=math.asin(np.clip(2*(w*y-z*x),-1.,1.))
        rows.append(dict(time=float(times[i]),height=float(qpos[i,2]),up_z=float(up),roll_deg=math.degrees(roll),pitch_deg=math.degrees(pitch),feet_z=data.site_xpos[sites,2].tolist(),knees=knees,contacts=int(data.ncon)))
    output.append(dict(seed=seed,last_1p5_seconds=rows))
path=args.folder/'failure_kinematics.json'
path.write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2),flush=True)
