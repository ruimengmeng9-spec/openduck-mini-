"""Frozen-model full-fall validation using unchanged R32 physical acceptance.

Development and qualification seed namespaces are separate. All trajectories
are retained; no partial-state library can enter these trials.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil

from diagnostics.getup_fullfallen_contract_r32 import POSES, completion_summary
from diagnostics.getup_independent_native import digest
from diagnostics.validate_getup_fullfallen_r32 import trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-base',type=int,required=True)
    p.add_argument('--count',type=int,default=3)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--qualification',action='store_true')
    args = p.parse_args()
    if args.seed_base < 3000000 or args.count < 1 or args.count > 20:
        p.error('Use reserved fresh seeds and at most 20 trials per pose')
    if args.qualification and args.count != 20:
        p.error('Qualification requires 20 trials per orientation')
    args.output.mkdir(parents=True,exist_ok=False)
    frozen = args.output/'tested_actor.onnx'
    shutil.copy2(args.actor,frozen)
    source = args.output/'executed_sources'
    source.mkdir()
    names = (Path(__file__).name,'validate_getup_fullfallen_r32.py',
             'getup_fullfallen_contract_r32.py','getup_fullfallen_env_r32.py',
             'validate_getup_fullpath_r27.py','getup_independent_native.py',
             'train_getup_fullpath_r27.py')
    hashes = {}
    for name in names:
        path = Path(__file__).with_name(name)
        shutil.copy2(path,source/name)
        hashes[name] = digest(path)
    scene = args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    hashes.update(actor=digest(frozen),scene=digest(scene),standing_actor=digest(stand))
    jobs = [(str(args.root),str(frozen),pose,args.seed_base+1000*k+i,12.,30.,
             str(args.output/f'{pose}_{args.seed_base+1000*k+i}.npz'))
            for k,pose in enumerate(POSES) for i in range(args.count)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(trial,jobs,chunksize=1):
            rows.append(row)
            (args.output/'partial_results.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    summary,passed = completion_summary(rows)
    report = dict(experiment='R98 corrected mixed PPO',simulation_only=True,
                  hardware_readiness=False,qualification=args.qualification,
                  seed_base=args.seed_base,hashes=hashes,summary=summary,results=rows,
                  full_task_completed=bool(args.qualification and passed),
                  development_successes=sum(r['success'] for r in rows),
                  uninterrupted_strict_hold_s=30.,root_edits_after_initialization=0,
                  development_seeds_must_not_be_reused_as_unseen=True,
                  remaining_noise_delay_and_expanded_validation=True)
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('R98_VALIDATION_TERMINAL',json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
