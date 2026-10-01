"""Package a decoder-only filter experiment with the unchanged exact ONNX model."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--tau',type=float,required=True)
    a=p.parse_args()
    if a.output.exists() or not 0 < a.tau <= .1:
        raise ValueError('existing artifact preserved or invalid filter time constant')
    data=json.loads(a.source.read_text())
    if 'rows' in data:
        data['source_training_rows']=data.pop('rows')
    if 'score' in data:
        data['source_search_score']=data.pop('score')
    data.update(simulation_only=True,target_smoothing_tau_s=a.tau,
                filter_contract='50Hz alpha=dt/(tau+dt); initialized from previous physical motor target; after joint clamp and before unchanged slew limit; history uses filtered preslew target',
                source_controller=str(a.source),filter_not_PPO_training=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2))
