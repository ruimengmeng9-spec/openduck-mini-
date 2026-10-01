# Open Duck left side get up timing diagnosis

The completed early-motion timing search did not improve independent
get-up performance. The next simulation experiment tests whether changing
the duration of a constant home-pose wait can prevent divergence before
the terminal flip. This is a hypothesis, not an established cause.

## Completed early motion search

R76 completed thirteen generations and retained the original timing.
Short training success was 13/24. Paired independent full-fall qualification
was 13/40 for both candidate and baseline, requiring thirty continuous
strict-standing seconds with the original physical audit. All candidate
metrics match their paired baseline metrics. The method therefore produced
no improvement and is not ready for deployment.

Thirty-eight of forty candidate audits are valid. Two failures, seeds
778001 and 778021, stop at control step 45 with approximately 0.030 m peak
self-penetration. The other twenty-five unsuccessful independent cases
need separate explanation; the collision failures cannot explain them all.
The prior user-requested pause and verified continuation did not reset the
experiment, replace its checkpoint or launch another training process.

## Current wait duration experiment

R77 uses the frozen R73 checkpoint and the same twenty-four actual-fall
training seeds: 769000–769007 and 773000–773015. Its eighteen requested wait
durations are 2, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5,
1.75, 2.25, 2.5, 3, 4 and 5 seconds. Durations are rounded to the existing
0.02-second control grid; the saved target step count is authoritative.

Only identical phase-1 home commands are repeated or removed. The initial
recovery path, its timing, wait target, later pulse and terminal commands,
feedback gains, joint limits, torque limits, target slew limit, collision
model and qualification thresholds remain unchanged. Each trial begins
from its own physically settled fallen state and runs the whole path
without an intermediate root-state reset or teleport. Failed trajectories
are retained. The nominal reference and nominal rollout must first pass
the existing mandatory short gate before a wait is tested on the cases.

This diagnostic uses a one-second continuous strict-standing short training
gate, as the preceding training searches do. It does not replace the final
thirty-second gate. The pool uses six CPU workers and no robot hardware or
GPU. Twenty-two control-contract tests passed before launch. Results are
written to `/data/shijinsheng/open_duck/outputs/getup_wait_library_r77_20261001`.

## Interpretation and next decision

A better fixed wait would justify fresh independent full-fall validation,
not a success claim from training alone. If different waits rescue different
training cases, their retrospective union is only a best-case ceiling. A
sensor-based choice must be trained from available pre-decision observations
and independently tested; choosing a wait using its eventual outcome is
not a deployable policy. If the library adds no coverage, do not repeat a
timing search without a new mechanism supported by the failures.

No independent seed is used to select the waits. Future fitting must not
use previously inspected independent outcomes as a new test set. Left-side
qualification still requires at least eighteen of twenty fresh perturbations
with a valid physical audit and thirty continuous strict-standing seconds,
followed by a larger independent set and noise and delay testing. Right-side,
prone and supine recovery remain separate unqualified stages. Neither R76
nor this diagnostic establishes hardware readiness.
