"""Explicit simulation-XY outer-loop trial with an unchanged inner ONNX actor."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gain',type=float,required=True)
    args=p.parse_args()
    if args.output.exists() or not 0 <= args.gain <= 2:
        raise ValueError('existing output or invalid gain')
    data=json.loads(args.controller.read_text())
    data['source_search_score']=data.pop('score',None)
    data['source_training_rows']=data.pop('rows',[])
    data.update(controller_type='phase_contact_path_probe_v1',source_controller=str(args.controller),
                path_heading_gain_rad_per_m=args.gain,path_heading_limit_rad=.15,
                requires_simulated_xy=True,simulation_only=True,
                path_heading_contract='goal=h0+clip(gain*initial-frame-cross-track,-0.15,0.15); backward motion only')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2))
    print('PATH PROBE:',args.output,'GAIN:',args.gain,flush=True)
