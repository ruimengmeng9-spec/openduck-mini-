"""Fit normal body-pitch oscillation from a successful training-side trajectory.

The resulting template is a feature-preprocessing contract, not a motor policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from diagnostics.backward_phase_search import NativeRollout, body_pitch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',default='/data/shijinsheng/open_duck')
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('existing template preserved')
    engine=NativeRollout(a.root)
    config=json.loads(a.controller.read_text())
    trajectory=np.load(a.trajectory,allow_pickle=False)
    qpos,times=trajectory['qpos'],trajectory['time']
    keep=times>=3.
    pitch=np.asarray([body_pitch(q) for q in qpos])[keep]
    period=engine.dt*engine.sim.PRM.nb_steps_in_period
    phi=np.rint(times[keep]/engine.dt)%engine.sim.PRM.nb_steps_in_period
    phi=phi/engine.sim.PRM.nb_steps_in_period*2*np.pi
    design=np.stack([np.ones_like(phi)]+[v for h in (1,2,3) for v in (np.cos(h*phi),np.sin(h*phi))],axis=1)
    coefficients=np.linalg.lstsq(design,pitch,rcond=None)[0]
    errors=pitch-design@coefficients
    config.pop('balance_weights',None)
    config.update(pitch_template_weights=coefficients.tolist(),pitch_template_period_s=period,
                  pitch_template_source=str(a.trajectory),pitch_template_source_sha256=hashlib.sha256(a.trajectory.read_bytes()).hexdigest(),
                  pitch_template_fit_rms_rad=float(np.sqrt(np.mean(errors**2))),
                  pitch_template_fit_p95_abs_rad=float(np.percentile(abs(errors),95)),
                  balance_deadband_rad=.04)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(config,indent=2))
    print('TEMPLATE:',a.output,'period:',period,'pitch range:',[float(min(pitch)),float(max(pitch))],
          'RMS rad:',config['pitch_template_fit_rms_rad'],'p95:',config['pitch_template_fit_p95_abs_rad'],flush=True)


if __name__=='__main__':
    main()
