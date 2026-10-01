"""Create an explicit, recoverable scaled-gait experiment; never overwrite input."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scale',type=float,required=True)
    a=p.parse_args()
    if not 0 <= a.scale <= 1 or a.output.exists():
        raise ValueError('invalid scale or existing output')
    data=json.loads(a.input.read_text())
    data['weights']=(np.asarray(data['weights'])*a.scale).tolist()
    data.pop('feedback_weights',None)
    data['controller_type']='phase_leg_correction_v1'
    data['phase_scale']=a.scale
    data['source_controller']=str(a.input)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
