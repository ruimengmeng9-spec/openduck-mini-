"""Unique publication receipt; original strict repo-only publisher unchanged."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,'/data/shijinsheng/open_duck/tmp')
import push_openduck_server_20261009 as publisher
original_run=subprocess.run


def upload_timeout(args,*positional,**kwargs):
    if args==['git','-c','gc.auto=0','push','--progress','server-publish','HEAD:refs/heads/main']: kwargs['timeout']=3600
    return original_run(args,*positional,**kwargs)


if __name__=='__main__':
    receipt=Path('/data/shijinsheng/open_duck/tmp/getup_publication_r191_terminal_20261010_result.json')
    assert not receipt.exists() and '--push' in sys.argv
    expected=sys.argv[sys.argv.index('--expected')+1]
    subprocess.run=upload_timeout; publisher.main()
    receipt.write_text(json.dumps(dict(published_terminal=expected,helper_remote_after_verified=True,
        upload_only_timeout_s=3600,private_key_exported=False,force_push=False),indent=2)+'\n')
