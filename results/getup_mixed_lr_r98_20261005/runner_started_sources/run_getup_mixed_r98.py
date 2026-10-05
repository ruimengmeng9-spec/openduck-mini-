"""Bounded, reproducible R98 learning-rate A/B, not unbounded blind training."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path('/data/shijinsheng/open_duck')
PROJECT = ROOT/'projects/Open_Duck_Playground'
BASE = ROOT/'training/getup_prefix_stage3_smoke_20261002/final.msgpack'
LIBRARY = ROOT/'outputs/getup_prefix_state_audit_r73_20261002/prefix_states.npz'
COMPLETE_POSES = ('prone','supine','left_side','right_side')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_audit():
    source = PROJECT/'diagnostics/train_getup_prefix_ppo.py'
    tree = ast.parse(source.read_text())
    adam = next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='adam')
    records = {}
    for name in ('getup_mixed_p05_smoke_20261002b','getup_mixed_p05_extended_20261002'):
        directory = ROOT/'training'/name
        contract = json.loads((directory/'controller_contract.json').read_text())
        episodes = json.loads((directory/'episodes.json').read_text())
        records[name] = dict(recorded_lr=contract['optimizer_learning_rate'],
            actual_mixed_lr=1e-4,episode_count=len(episodes),
            mislabeled_fullfall_episodes=sum(e['pose']=='prefix_library' and e.get('actual_fallen_start',False) for e in episodes),
            mislabeled_fullfall_discoveries=sum(e['pose']=='prefix_library' and e.get('actual_fallen_start',False) and e['training_success'] for e in episodes),
            actual_complete_fall_episodes=sum(e['pose'] in COMPLETE_POSES for e in episodes),
            actual_complete_fall_discoveries=sum(e['pose'] in COMPLETE_POSES and e['training_success'] for e in episodes),
            episodes_by_pose=dict(Counter(e['pose'] for e in episodes)),
            contract_sha256=digest(directory/'controller_contract.json'),
            episode_sha256=digest(directory/'episodes.json'))
    library_audit = ROOT/'outputs/getup_prefix_state_audit_r73_20261002/prefix_state_audit.json'
    audit = json.loads(library_audit.read_text())
    return dict(actual_adam_expression=ast.unparse(adam.args[0]),
                source_sha256=digest(source),historical_runs=records,
                prefix_library_sha256=digest(LIBRARY),
                prefix_audit_sha256=digest(library_audit),prefix_rows=audit['rows'],
                prefix_up_z_range=[min(r['up_z'] for r in audit['states']),max(r['up_z'] for r in audit['states'])],
                prefix_all_body_contact_false=all(not r['body_contact'] for r in audit['states']),
                prefixes_are_training_only=True)


def execute(args, log, cpu=False):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    env.update(CUDA_VISIBLE_DEVICES='' if cpu else '6',JAX_PLATFORMS='cpu' if cpu else 'cuda')
    print('START',str(log),flush=True)
    with open(log,'x') as stream:
        subprocess.run([sys.executable,'-u',*args],cwd=PROJECT,env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('END',str(log),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source = args.output/'executed_sources'
    source.mkdir()
    for name in ('run_getup_mixed_r98.py','train_getup_mixed_ppo_r98.py',
                 'getup_mixed_curriculum_env_r98.py','test_getup_mixed_r98.py','validate_getup_mixed_r98.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    audit = historical_audit()
    (args.output/'historical_audit.json').write_text(json.dumps(audit,indent=2))
    print('HISTORICAL_AUDIT',json.dumps(audit),flush=True)
    free = int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    if free < 8000:
        raise RuntimeError('GPU6 not available; never evict another process')
    experiment = dict(seed=198,iterations=args.iterations,envs=16,workers=4,horizon=128,
        rates={'a_legacy_lr':1e-4,'b_corrected_lr':5e-6},prefix_probability=.5,
        warm_start_checkpoint=str(BASE),warm_start_sha256=digest(BASE),
        prefix_library_sha256=digest(LIBRARY),warm_start_not_exact_resume=True,
        development_seed_base=3105000,qualification_seed_base=3115000,
        seed_stride_per_pose=1000,physics_and_acceptance_unchanged=True,
        simulation_only=True,hardware_readiness=False,
        validation_after_each_run=True,full_task_completed=False,
        hypothesis='Correcting 20x mixed LR may preserve skill and improve actual full-fall discovery; not assumed sufficient')
    (args.output/'experiment.json').write_text(json.dumps(experiment,indent=2))
    execute(['-m','unittest','diagnostics.test_getup_mixed_r98',
             'diagnostics.test_getup_fullfallen_r32','diagnostics.test_getup_temporal_noise_r33','-v'],
             args.output/'tests.log',cpu=True)
    execute(['-m','diagnostics.train_getup_mixed_ppo_r98','--output',str(args.output/'smoke'),
        '--iterations','2','--envs','4','--workers','2','--horizon','16','--seed','198',
        '--initialize-actor',str(BASE),'--mixed-library',str(LIBRARY),'--learning-rate','5e-6'],
        args.output/'smoke.log')
    outcomes = {}
    for label,rate in experiment['rates'].items():
        run = args.output/label
        execute(['-m','diagnostics.train_getup_mixed_ppo_r98','--output',str(run),
                 '--iterations',str(args.iterations),'--envs','16','--workers','4','--horizon','128',
                 '--seed','198','--initialize-actor',str(BASE),'--mixed-library',str(LIBRARY),
                 '--learning-rate',str(rate)],args.output/f'{label}.log')
        execute(['-m','diagnostics.validate_getup_mixed_r98','--actor',str(run/'final.onnx'),
            '--output',str(args.output/f'{label}_development'),'--seed-base','3105000',
            '--count','3','--workers','4'],args.output/f'{label}_development.log',cpu=True)
        result = json.loads((args.output/f'{label}_development/results.json').read_text())
        outcomes[label] = dict(successes=result['development_successes'],
            summary=result['summary'],model_sha256=result['hashes']['actor'])
        (args.output/'comparison_partial.json').write_text(json.dumps(outcomes,indent=2))
    # Selection is on development only. Frozen final qualification cannot tune.
    chosen = max(outcomes,key=lambda label:outcomes[label]['successes'])
    report = dict(outcomes=outcomes,selected_on_development_only=chosen,
                  selection_is_not_qualification=True,full_task_completed=False,
                  simulation_only=True,hardware_readiness=False)
    if outcomes[chosen]['successes'] >= 9:
        execute(['-m','diagnostics.validate_getup_mixed_r98','--actor',str(args.output/chosen/'final.onnx'),
            '--output',str(args.output/'independent_qualification'),'--seed-base','3115000',
            '--count','20','--workers','4','--qualification'],args.output/'qualification.log',cpu=True)
        report['qualification'] = json.loads((args.output/'independent_qualification/results.json').read_text())['summary']
        report['remaining_expanded_noise_delay_validation'] = True
    else:
        report['qualification_not_run_reason'] = 'Development full-fall success below 9/12; preserve failures and analyze before more search'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('R98_TERMINAL',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
