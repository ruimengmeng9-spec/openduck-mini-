# Backward speed sweep: stability is solved, speed is structurally capped (2026-09-27)

Native MuJoCo, flat terrain. No hardware commands were sent.

## Question

v30 became the first reverse policy to pass the stability half of the gate
(10 s, 5/5 perturbed seeds). It sat at `-0.028 m/s` against a `-0.074` command, so
the next question was whether simply paying more for speed closes the gap.

## Result

| Candidate | Speed-pressure terms | 10 s completed | Mean reverse speed |
| --- | --- | ---: | ---: |
| v30 | 1x (80 / 80) | 5/5 | -0.0276 m/s |
| v32 speed2x | 2x (160 / 160) | 5/5 | -0.0290 m/s |
| v32 speed4x | 4x (320 / 320) | 5/5 | -0.0251 m/s |

Raw data: `results/backward_speed_sweep_20260927.json` and the per-run
`results/sus_backward_v32_*_20260927.json`.

**Doubling and then quadrupling the speed reward did not increase speed.** The three
values are within run-to-run noise of each other, and all stay at 5/5 stable. The
`-0.074 m/s` target was never approached.

## What this means

1. **Speed is not reward-limited.** A term that is worth roughly +1.6/step when
   tracking and is scaled 4x still does not buy a longer stride, so the bottleneck is
   not how strongly speed is incentivised.
2. **The cap is structural.** The most economical reading is that at this command the
   contact/foot-placement objective that keeps the robot balanced (rear support
   margin, single-support, rear swing) also grips the stride, so the stable gait it
   can find is a short one. v32 raised the speed reward without relaxing those terms,
   which is why nothing moved.
3. **The stable endpoint is robust.** 5 consecutive seeds, two different reward
   weights, 10 s each, zero falls. v30's mechanism is reproducible, not a lucky run.

## Next

Test the trade-off from the other side: relax the contact/stability group itself and
keep the v32 speed pressure. Launched as `backward_v33_relaxed` (single-support 25,
swing-rear 20, rear-shortfall 40) and `backward_v33_released` (10 / 8 / 20), both from
the v30 checkpoint. The gate is unchanged: 10 s x 5 seeds with
`|mean speed| >= 0.037 m/s`. Nothing is promoted or deployed until a candidate meets
both halves.

## Reproducibility

`scripts/launch_backward_v32.sh` (parameterised by speed pressure), driven with the
stability group from v30 and no heading terms. Speed-pressure knobs
`BACKWARD_NORMALIZED_PROGRESS` / `BACKWARD_PROGRESS_SHORTFALL` were added to
`playground/open_duck_mini_v2/focused_skill.py` this round.
