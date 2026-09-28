"""Freeze an early candidate explicitly; never confuse it with final search output."""
import argparse
import hashlib
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('existing snapshot preserved')
    payload=a.source.read_bytes()
    data=json.loads(payload)
    data.update(snapshot_source=str(a.source),snapshot_source_sha256=hashlib.sha256(payload).hexdigest(),snapshot_not_final=True)
    a.output.write_text(json.dumps(data,indent=2))
