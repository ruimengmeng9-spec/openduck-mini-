"""Evidence-backed summary; successes are not automatically learning gains."""
import argparse
import json
from pathlib import Path


def load(path): return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training',type=Path,default=Path('/data/shijinsheng/open_duck/training'))
    args=p.parse_args()
    t=args.training; a=t/'getup_native_ppo_r10_stage1'; b=t/'getup_native_ppo_r11_stage1'
    summaries={name:load(d/'training_summary.json') for name,d in [('r10',a),('r11',b)]}
    histories={name:load(d/'training_history.json') for name,d in [('r10',a),('r11',b)]}
    evals={
        'r10_tilt025_test_44000':load(a/'evaluation.json')['aggregates'],
        'r10_tilt055_test_44000':load(a/'evaluation_tilt055.json')['aggregates'],
        'r11_final_tilt055_test_74000':load(b/'evaluation.json')['aggregates'],
        'r11_final_tilt025_test_74000':load(b/'evaluation_tilt025.json')['aggregates'],
        'r11_selected_tilt055_test_95000':load(b/'selected_candidate/evaluation.json')['aggregates'],
        'r11_selected_tilt025_test_95000':load(b/'selected_candidate/evaluation_tilt025.json')['aggregates'],
    }
    report=dict(stage='loaded low start / tilted near-standing recovery; NOT full fallen recovery',
        main_environment_steps=sum(r['environment_steps'] for r in summaries.values()),
        summaries=summaries,training_episodes={name:dict(episodes=sum(r['completed_episodes'] for r in rows),
            successes=sum(r['successful_episodes'] for r in rows)) for name,rows in histories.items()},
        evaluations=evals,selection=load(b/'checkpoint_selection/selection.json')['winner'],
        simulation_only=True,hardware_readiness=False,promoted_to_motion_runtime=False,
        conclusion='No clear improvement on independent tests; preserve original motion controllers.',
        caveats=['20 seeds per comparison; no population-level reliability guarantee',
                 'safe_runs means sampled finite/penetration/joint checks, NOT absence of falls',
                 'actor has 50 inputs and 14 independent normalized joint targets, not the walking 101-input graph',
                 'reset qualification does not imply full fallen recovery',
                 'R11 reuses R10 weights but resets optimizer and exploration standard deviation'])
    output=t/'getup_native_ppo_r10_r11_summary.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
