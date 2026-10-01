"""Fit phase-dependent approximate capture offset, then inspect a paired failure."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from diagnostics.backward_phase_search import NativeRollout, body_pitch


def basis(times,dt,steps):
    phi=(np.rint(times/dt)%steps)/steps*2*np.pi
    return np.stack([np.ones_like(phi)]+[v for h in (1,2,3) for v in (np.cos(h*phi),np.sin(h*phi))],axis=1)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--failure',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('existing capture template preserved')
    engine=NativeRollout('/data/shijinsheng/open_duck')
    trajectory=np.load(args.trajectory,allow_pickle=False)
    times=trajectory['time']
    cp=trajectory['capture_forward_offset_m']
    keep=times>=3.
    design=basis(times,engine.dt,engine.sim.PRM.nb_steps_in_period)
    weights=np.linalg.lstsq(design[keep],cp[keep],rcond=None)[0]
    errors=cp[keep]-design[keep]@weights
    config=json.loads(args.controller.read_text())
    config.update(capture_template_weights=weights.tolist(),capture_template_period_s=engine.dt*engine.sim.PRM.nb_steps_in_period,
                  capture_template_source=str(args.trajectory),capture_template_source_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
                  capture_template_fit_rms_m=float(np.sqrt(np.mean(errors**2))),capture_template_fit_p95_abs_m=float(np.percentile(abs(errors),95)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,indent=2))
    failure=np.load(args.failure,allow_pickle=False)
    ft=failure['time']
    fe=failure['capture_forward_offset_m']-basis(ft,engine.dt,engine.sim.PRM.nb_steps_in_period)@weights
    pitch=np.asarray([body_pitch(q) for q in failure['qpos']])
    bad_cp=np.flatnonzero((ft>=3)&(fe<-.01))
    bad_pitch=np.flatnonzero((ft>=3)&(pitch<-.3))
    rows=[dict(time_s=float(ft[i]),capture_error_cm=float(fe[i]*100),pitch_deg=math.degrees(float(pitch[i]))) for i in range(max(0,len(ft)-101),len(ft),5)]
    diagnosis=dict(template_rms_m=config['capture_template_fit_rms_m'],template_p95_abs_m=config['capture_template_fit_p95_abs_m'],
                   first_capture_error_below_minus_1cm_s=float(ft[bad_cp[0]]) if len(bad_cp) else None,
                   first_pitch_below_minus_0p3_rad_s=float(ft[bad_pitch[0]]) if len(bad_pitch) else None,
                   failure=str(args.failure),last_two_seconds=rows,
                   interpretation='Approximate LIPM diagnostic relative to both foot centers, not an exact support-polygon certificate.')
    (args.output.parent/'capture_diagnosis.json').write_text(json.dumps(diagnosis,indent=2))
    print(json.dumps(diagnosis,indent=2),flush=True)
