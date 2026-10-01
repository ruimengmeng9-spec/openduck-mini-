"""Create a simulation candidate only after all three distinct gates pass."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from diagnostics.summarize_phase_result import summarize


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--steady',type=Path,required=True)
    p.add_argument('--long',type=Path,required=True)
    p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('existing candidate preserved')
    contract=json.loads(a.contract.read_text())
    digest=hashlib.sha256(a.model.read_bytes()).hexdigest()
    if digest!=contract['onnx_sha256'] or contract.get('target_smoothing_tau_s')!=.01:
        raise ValueError('model/filter contract mismatch')
    steady=summarize(a.steady)
    long=summarize(a.long)
    if steady['tested']<50 or steady['qualified']!=steady['tested'] or steady['duration_s'][0]<60:
        raise ValueError('steady gate has not passed')
    if long['tested']<20 or long['qualified']!=long['tested'] or long['duration_s'][0]<120:
        raise ValueError('long gate has not passed')
    sequence=json.loads(a.sequence.read_text())
    rows=sequence['rows']
    if len(rows)<50 or not all(row['completed'] and row['qualified'] for row in rows) or sequence.get('stop_blend_s')!=1.:
        raise ValueError('full-sequence gate has not passed')
    for path in (a.steady,a.long):
        result=json.loads(path.read_text())
        if result['corrector_onnx']!=str(a.model) or result['controller']!=str(a.contract):
            raise ValueError('validation used a different model/contract')
    if sequence['model']!=str(a.model) or sequence['contract']!=str(a.contract):
        raise ValueError('sequence used a different controller')
    manifest=dict(status='validated_fixed_plane_simulation_candidate',simulation_only=True,hardware_readiness=False,
                  baseline_actor='models/backward_reference_residual_r2/final.onnx',baseline_sha256=contract['baseline_sha256'],
                  corrector_sha256=digest,control_dt_s=.02,target_smoothing_tau_s=.01,stop_crossfade_s=1.,
                  command_backward_mps=-.074,reference_dx=-.0925,reference_interpolation=True,
                  model_output='14 joint deltas in radians; not legacy normalized servo actions',
                  requires_matching_reference_decoder=True,legacy_motor_action_decoder_allowed=False,
                  scope='fixed flat terrain and friction; exact home pose; initial qvel uniform +/-0.02; no added observation noise, delay, pushes or terrain variation; no hardware test',
                  steady_gate=steady,long_gate=long,sequence_gate=dict(tested=len(rows),qualified=sum(r['qualified'] for r in rows),schedule=sequence['schedule'],stop_blend_s=sequence['stop_blend_s']),
                  result_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.steady,a.long,a.sequence)})
    a.output.mkdir(parents=True)
    shutil.copy2(a.model,a.output/'corrector.onnx')
    shutil.copy2(a.contract,a.output/'controller_contract.json')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SIMULATION CANDIDATE SAVED:',a.output,flush=True)
