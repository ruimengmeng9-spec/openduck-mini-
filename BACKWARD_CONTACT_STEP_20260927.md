# Backward contact-aware stepping: first sustained reverse (2026-09-27)

All tests are native MuJoCo on flat terrain. No hardware commands were sent.

## The gate

An earlier note (2026-09-26) diagnosed the reverse fall as **the body moving back
faster than the support foot is repositioned, with no pitch recovery**, and
predicted that penalising the COM-behind-support precursor and rewarding a rear
swing foot would help. This round tested that prediction against an independent
acceptance gate:

> 10 s upright on 5 perturbed seeds **and** mean reverse speed at least half the
> `-0.074 m/s` command (i.e. `>= 0.037 m/s`).

## Results

| Candidate | Completed 10 s | Mean speed | Durations (s) | Verdict |
| --- | ---: | ---: | --- | --- |
| v29 rear-support-shortfall | 0/5 | -0.1115 | 1.56, 1.54, 1.56, 1.54, 1.56 | Fast but falls at ~1.55 s |
| **v30 contact-aware stepping** | **5/5** | **-0.0276** | 10.0 x5 | **First reverse policy to stay up the full window** |
| v31 v30 + heading/yaw/lin-vel | 1/5 | -0.0449 | 10.0, 4.56, 6.34, 7.36, 5.80 | Regression: heading group breaks it |

Raw data: [`results/backward_contact_step_gate_20260927.json`](results/backward_contact_step_gate_20260927.json).

## What this establishes

1. **The stability half of the gate is now achievable.** v30 is the first reverse
   policy that never falls in the acceptance window on any seed. The mechanism is
   the reward group that pays attention to *which* foot is down and where the rear
   foot is swinging (`BACKWARD_REAR_SUPPORT_SHORTFALL`, `BACKWARD_SINGLE_SUPPORT`,
   `BACKWARD_SWING_REAR`). This confirms the placement/pitch diagnosis: the earlier
   failures were a foot-placement problem, not a reward-magnitude problem.
2. **The remaining gap is speed, and it is a trade.** v30 reaches only `-0.028`
   against `-0.074`. Stability and speed move in opposite directions here, which is
   the same speed/stability cliff the action-blend sweep found in v26, now with a
   usable stable endpoint to move from.
3. **Heading constraints are counterproductive here.** v31 added `heading_error`,
   `yaw_error` and a stronger `lin_vel_xy_error` on top of v30 and dropped
   completion to 1/5. They should be left off while chasing speed.

## Next

`scripts/launch_backward_v32.sh` keeps v30's contact group, omits the v31 heading
group, and raises only the two speed-pressure terms (`BACKWARD_NORMALIZED_PROGRESS`
/ `BACKWARD_PROGRESS_SHORTFALL`, added as knobs this round) by 2x and 4x from the
v30 checkpoint. Acceptance still requires 10 s x 5 seeds with
`|mean speed| >= 0.037 m/s`; nothing is promoted or deployed on the strength of a
few seconds of motion.

## Reproducibility

Acceptance script: `validate_backward_sustained.py`. v31 launcher kept as
`scripts/launch_backward_v31.sh`. **v30 was run ad hoc from the shell and its
exact command was not saved**; its reward group is recoverable from the three
`BACKWARD_*` terms quoted above and is reproduced in
`scripts/launch_backward_v32.sh`, which uses precisely that set. The live reward
implementation is in `playground/open_duck_mini_v2/focused_skill.py`
(`rear_contact_margin` plus the `reverse_*` / `rear_support_shortfall` scales).
