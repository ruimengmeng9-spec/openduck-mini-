"""Run existing native recovery search with clean spawned worker processes.

Only process creation changes. Does not change mechanics, objectives, candidate
definitions or success gates. Record this wrapper alongside the executed source.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

from diagnostics.getup_independent_native import digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',choices=('train_getup_fullpath_r27','search_getup_contact_archive_r30'),required=True)
    args,remaining=parser.parse_known_args()
    if '--output' not in remaining:
        parser.error('target search needs --output')
    out=Path(remaining[remaining.index('--output')+1])
    if out.exists():
        raise RuntimeError('Refusing to overwrite existing search')
    out.parent.mkdir(parents=True,exist_ok=True)
    contract=dict(worker_start_method='spawn',module=args.module,wrapper_sha256=digest(__file__),
                  arguments=remaining,only_process_creation_changes=True,
                  mechanics_objective_and_success_gates_unchanged=True)
    (out.parent/(out.name+'_spawn_launcher.json')).write_text(json.dumps(contract,indent=2))
    module=importlib.import_module('diagnostics.'+args.module)
    module.ProcessPoolExecutor=partial(ProcessPoolExecutor,mp_context=mp.get_context('spawn'))
    sys.argv=[args.module,*remaining]
    module.main()
    shutil.copy2(__file__,out/'run_getup_spawn.py')
    (out/'spawn_launcher.json').write_text(json.dumps(contract,indent=2))
    print('SPAWN_WRAPPER_RETURNED',flush=True)


if __name__=='__main__':
    main()
