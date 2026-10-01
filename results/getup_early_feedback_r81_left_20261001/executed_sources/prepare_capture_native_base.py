"""Prepare zero additive feedback, preserving the R11 inner controller exactly."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('existing native capture base preserved')
    config=json.loads(a.source.read_text())
    config.update(capture_weights=[0.,0.,0.,0.],simulation_only=True)
    config.pop('onnx_sha256',None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(config,indent=2))
