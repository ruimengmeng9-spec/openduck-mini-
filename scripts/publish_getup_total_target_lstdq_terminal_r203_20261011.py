"""Unique R203 terminal normal-FF publisher; unchanged strict repository-only helper."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,'/data/shijinsheng/open_duck/tmp')
import push_openduck_server_20261009 as publisher
original_run=subprocess.run
def upload_timeout(args,*positional,**kwargs):
    if args==['git','-c','gc.auto=0','push','--progress','server-publish','HEAD:refs/heads/main']:kwargs['timeout']=3600
    return original_run(args,*positional,**kwargs)
if __name__=='__main__':
    root=Path('/data/shijinsheng/open_duck')
    receipt=root/'tmp/getup_publication_r203_lstdq_terminal_20261011_result.json'
    assert not receipt.exists() and '--push' in sys.argv
    expected=sys.argv[sys.argv.index('--expected')+1]
    archive=json.loads((root/'tmp/getup_total_target_lstdq_r203_archive_result_20261011.json').read_text())
    archive_exit=json.loads((root/'tmp/getup_total_target_lstdq_r203_archive_exit_20261011.json').read_text())
    assert archive_exit['natural_exit'] and archive_exit['exit_code']==0 and archive['commit']==expected and archive['repository_clean']
    subprocess.run=upload_timeout;publisher.main()
    with receipt.open('x') as f:f.write(json.dumps(dict(published_terminal=expected,helper_remote_after_verified=True,upload_only_timeout_s=3600,private_key_exported=False,force_push=False,new_dynamic_attempts=0),indent=2)+'\n')

