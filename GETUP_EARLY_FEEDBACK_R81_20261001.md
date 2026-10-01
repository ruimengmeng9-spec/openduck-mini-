# R81 early asymmetric sensor feedback

Simulation only. Real-robot debugging remains paused. No hardware
connection, motor enable, deployment or walking-controller replacement.

## Evidence and hypothesis

R80 completed thirteen generations and retained exactly zero early-pose
correction. Fresh full-fall 30 s qualification is 13/40 for candidate and
baseline, with no paired change. Uniform early offsets therefore did not
improve this search. R79's training-only diagnosis found body/contact
divergence not explained solely by average encoder tracking or fixed wait
length. This does not establish a unique cause or prove fixed poses can
never work; it supports a new controlled policy test.

R81 adds independent responses of the eight leg joints to the pre-control
body-frame upvector X/Y errors relative to the frozen nominal sensor
reference. The learned law is an 8-by-2 matrix. It operates only within the
same 0.5--3.5 s early window as R80, with a smooth envelope rising to one at
1.5 s and returning to zero at 3.5 s. Unlike R80's offsets, the correction
depends on each trial's current observed tilt. It allows asymmetric leg
responses, unlike the earlier shared bilateral phase-0 feedback mapping.
The experiment therefore tests a localized asymmetric feedback architecture,
not a claim that either localization or asymmetry alone is the cause.

The original feedback and added feedback are summed, then clipped together
to the original +/-0.18 rad residual cap. Their budgets are not added. All
reference commands, timing, home wait, terminal targets, head targets,
existing gain values, collision model, joint/torque/slew limits and strict
qualification rules remain frozen. Feedback uses only current IMU features,
not future outcomes, root pose, contact force or privileged state. Zero
learned gain delegates literally to the original R64 evaluator. No injected
reference state or intermediate reset occurs during a rollout.

## Training and validation

Train cases remain 769000--769007 and 773000--773015. The nominal actually
fallen case is mandatory. Zero-gain training must reproduce the completed
R80 short baseline before fitting can continue. Fixed seed 181, population
24, at most 24 generations, six CPU workers, twelve stale generations stop.
The short one-second standing gate is training-only.

After fitting, forty fresh seeds 781000--781039 and nominal are evaluated
from full physically settled falls. The new split excludes the prior
780000--780039 qualification cases. Candidate and frozen baseline are paired
from identical initial states. Full completion, unchanged physical audit
and thirty continuous strict-standing seconds in a 35 s final hold are
required. Training and independent baseline/candidate traces, including
failures, are saved. Checkpoints, RNG state, executed sources and hashes
are preserved. These are not hardware readiness tests.

Twenty-two contract/regression unit tests passed before launch. At this
archive the actual main process and six workers are live; no improvement
or independent success is yet claimed.

Output: `/data/shijinsheng/open_duck/outputs/getup_early_feedback_r81_left_20261001`.
Code: `diagnostics/train_getup_early_feedback_r81.py`.
Launcher: `scripts/launch_getup_early_feedback_r81_20261001.sh`.

## Continuation

Check the actual R81 process, log, checkpoint and result before another
launch. R80 is complete and must not be restarted. If R81 does not improve,
compare eligible candidates, independent rescues/regressions and physical
failures before a new hypothesis; do not repeat the same negative search.
No four-pose qualification gate has passed. Preserve the existing staged
18/20 fresh complete recoveries per pose, 30 s continuous strict standing
and valid audit, then expand unseen/noise/delay testing. Do not deploy.

## Completed result (supersedes running wording above)

R81 finished thirteen generations. Every generation retained zero added
gains, training remained 13/24, and all twenty-four candidates per generation
passed the nominal gate. Forty fresh full-fall qualification trials yielded
8/40 for both candidate and paired baseline, with no rescues or regressions.
The zero candidate delegates to the baseline, so this is a negative search
result, not a new validated controller. R80's 13/40 uses different seeds;
the difference between cohorts is not evidence that R81 worsened the policy.

Thirty-nine of forty independent trials passed the physical audit; seed
781004 stopped at control 45 with self penetration 0.029894956 m. Its failed
trace and the unchanged rejection threshold are retained. No hardware test
or four-pose gate passed. On the training traces, early IMU discrepancies
are nonzero while the original residual cap is not generally active; this
does not establish a unique cause or prove that the added gains can work.

Next experiment: `GETUP_SENSOR_CLOCK_R82_20261001.md`. Check its actual
process and saved result rather than restarting this completed R81 run.
