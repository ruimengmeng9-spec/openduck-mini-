# R81 negative result and R82 causal sensor-clock probe

Simulation only. Real-robot work stays paused. No robot connection, motor
enable, walking-controller replacement or hardware deployment is authorized.

## Evidence and new hypothesis

R81 ended after thirteen generations with zero learned early feedback,
13/24 short training recoveries, and 8/40 independent full-fall recoveries
for both candidate and paired baseline. There are no paired rescues or
regressions. Its baseline cohort is distinct from R80's 13/40 cohort.

Prior fixed time-warp searches, fixed pose offsets and localized feedback
did not establish an improvement. R79's saved training traces show early
body divergence not explained solely by average encoder tracking. R82 asks
whether actual body progress and the prescribed phase-0 clock become
misaligned. It is a *sensor-dependent* clock test, not a repetition of the
case-independent R75/R76 timing search. This hypothesis is not yet proven.

## Controlled intervention

Only phase-0's reference cursor changes. At each 50 Hz control boundary,
body upvector X/Y and scaled gyro features are measured before the command.
Within a local window of ten or twenty stored reference controls, weighted
feature distance plus a small index-distance penalty estimates reference
progress. The reference cursor's increment is clipped to 0.25--1.75 controls
per control. It never rewinds, skips the phase boundary or resets physics.
The first twenty-five controls are fixed. Every post-phase-0 command and
its duration is executed unchanged, including the final hold.

Stored reference values, including later reference samples, are constants;
no later trial observation or outcome is used by this causal policy. Targets
and reference features are linearly interpolated at the same cursor. Existing
feedback is computed against this reference with its original residual cap.
The actual motor target still goes through unmodified StrictSim substeps,
joint limits, force limit, target slew limit and physical audit. Changing
reference progress does not increase allowed target velocity.

Zero clock gain delegates literally to the original evaluator and must
reproduce nominal success and 13/24 training success before any selection.
The remaining eight grid policies combine gains 0.05, 0.15, 0.3, 0.6 with
the two reference matching radii. All evaluated nominal/training failures,
cursor rates and paired input hashes are saved. A clock reaching the end
does not imply success: valid physics, completed full path and the existing
continuous strict-standing gate remain mandatory.

## Search and validation contract

Training cases are still 769000--769007 and 773000--773015. The nominal full
fall must pass. Six CPU workers are bounded; no GPU or hardware is used.
The short one-second training tail is not qualification.

Only a higher training success count promotes a candidate to forty fresh
full-fall paired tests, seeds 782000--782039, with a 35-second final hold
and at least thirty continuous strict-standing seconds. If there is no
training improvement, the independent seeds remain unused and this fact
is explicit in the result. Inspected R81 seeds cannot become fresh tests.
Nineteen contract/regression tests passed before launch; these are not
physical recovery evidence. All physical limits and acceptance rules stay
unchanged. No complete four-pose simulation qualification has passed.

Output: `/data/shijinsheng/open_duck/outputs/getup_sensor_clock_r82_left_20261001`.
Source: `diagnostics/probe_getup_sensor_clock_r82.py`.
Launcher: `scripts/launch_getup_sensor_clock_r82_20261001.sh`.

## Continue safely

Read current results and check actual process identity before a new launch.
R67--R81 are finished; do not act on their old PIDs. If R82 fails, inspect
the saved cursor, feedback and body trajectory to distinguish ineffective
clock changes, wrong phase estimates and a genuinely inadequate path.
Do not rerun unchanged negative grids or describe training success as
30-second independent recovery. Preserve the staged per-pose 18/20 gate,
then larger unseen and noise/delay tests; never automatically deploy.

## Completed result

The identity clock reproduced 13/24. Eight adaptive settings all preserved
nominal success but obtained only 1, 2, 5, 2, 4, 1, 3, 3 training recoveries
in grid order. Their physical-audit rejects were 2, 1, 2, 2, 1, 1, 1, 1.
The saved cursor data confirm that the intervention changed reference
progress, so this is not simply a zero-effect search. Some settings rescue
individual baseline failures but lose many previous successes. No candidate
was promoted; qualification seeds 782000--782039 were not simulated.

Across settings an oracle could select successes for 20/24, but an oracle
uses future trial outcomes and is not a deployable policy. R83 tested causal
selection using the shared four-dimensional pre-intervention IMU errors at
control 25. Training-only leave-one-out fitting excludes each target case
from both neighbor labels and feature normalization. Neighbor counts 1, 3,
5, 7 yield 8, 6, 7, 9 successes, respectively, below baseline 13. The
selector is therefore rejected rather than promoted or called validated.
No fresh physical trials occurred in R83. Raw inputs and prediction rows
are retained in `outputs/getup_clock_selector_audit_r83_20261001`.

Next experiment: `GETUP_INITIAL_SETTLING_R84_20261001.md`. R82 and R83 are
complete and must not be restarted.
