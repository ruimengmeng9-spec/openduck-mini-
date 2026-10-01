# Short-wait get-up continuation R69–R70 (2026-09-30)

## Latest continuation 2026 10 01 R77 and R78 finished without improvement

R77 completed all eighteen waits. Fifteen preserve the mandatory nominal
gate; 0.1, 0.6 and 1.0 seconds do not. The best fixed wait remains 2 seconds,
13/24 short training successes. The retrospective union of successes across
eligible waits is 16/24, not a policy or independent qualification.

R78 fitted a sensor-only nearest-neighbour wait selector from those training
outcomes. Inputs are upvector, gyro, encoder offsets and joint velocities
after the original 308 early controls, before any wait command. All eligible
saved traces have bit-exact matching pre-decision positions, velocities and
observations. Feature normalization is fitted separately inside each
leave-one-case-out fold. A nominal sensor anchor and conservative baseline
fallback are included. Twelve neighbour-count and margin settings were
compared on training cases only; the best cross-check is **13/24**, equal to
the frozen baseline. Five selector unit tests passed, in addition to the
twenty-two control-contract tests. These are not independent rollout tests.

No new physical policy validation or long-running training was launched
from that negative result. R76, R77 and R78 are finished; do not restart
their obsolete PIDs. First read the final evidence in
`/data/shijinsheng/open_duck/outputs/getup_wait_library_r77_20261001` and
`/data/shijinsheng/open_duck/outputs/getup_wait_selector_r78_20261001`.
The selector is a recorded training artifact, not a validated replacement.
Further work should examine the eight training cases that no eligible wait
rescues and whether their contact/tilt divergence requires a genuinely
different feedback action or recovery path. This is a proposed direction,
not an established cause; diagnose the saved failures before another search.
Preserve original physics, complete-fall starts and thirty-second strict
qualification. No real-robot connection, deployment or walking-file change.
See `GETUP_WAIT_LIBRARY_R77_20261001.md`.

## Previous continuation 2026 10 01 R76 finished and R77 started

R76 has completed thirteen generations. The retained four timing factors
are all ones, with 13/24 short training successes. Full-fall seeds
778000 through 778039 qualified **13/40 versus 13/40** for the baseline;
every paired candidate result equals its baseline result. This is a negative
timing experiment, not a qualified skill. Thirty-eight candidate physical
audits are valid; seeds 778001 and 778021 fail the audit at control step 45
with about 0.030 m peak self-penetration. Those two early failures do not
explain the remaining twenty-five unsuccessful independent cases.

The earlier pause was user requested, not a training crash. The verified
process family resumed without a new training run, and the log advanced
from generation seven to eight before normal completion. Preserve the
resume record, final checkpoint, all paired replays and failures. Do not
resume the obsolete R76 PIDs or restart R67 through R76.

R77 now tests eighteen home-wait durations from zero to five seconds on the
same twenty-four actual-fall training cases, using the frozen pre-R74
checkpoint. Only the count of identical phase-1 home commands changes;
early motion, later commands, feedback, physical parameters, motor limits
and standing criteria remain fixed. Twenty-two unit tests passed. The
main Python PID was 1923572 at launch, with six CPU workers. Recheck its
command and output rather than relying on this historical PID.

Check `/data/shijinsheng/open_duck/outputs/getup_wait_library_r77_20261001`
and its matching `.log` first. This is a training-only diagnostic, not PPO
training or independent qualification. A rejected nominal gate is not
twenty-four simulated failures. A retrospective per-case best-wait count
is an oracle ceiling, not an executable policy. No held-out seed is used
to select waits. Any next controller needs fresh full-fall verification
with thirty continuous strict-standing seconds before claiming improvement.
No real-robot connection or deployment. See
`GETUP_WAIT_LIBRARY_R77_20261001.md`.

## Previous continuation 2026 10 01 R75 finished and R76 running

R75's 15-setting early timing grid is complete. Identity preserves nominal
success and 13/24 short training successes. Global 1.2 timing preserves
nominal but reduces training successes to 4/24; the other thirteen variants
fail the mandatory nominal gate. The selected factors remain all ones.
This is a training-only negative ablation, not independent qualification.

Read-only server inspection subsequently found an existing R76 CEM run at
`/data/shijinsheng/open_duck/outputs/getup_prefix_timing_cem_r76_20261001`.
Do not duplicate or stop this run. PID 1886896 was its parent Python process
at inspection, with six workers and a `flock` execution guard. Verify the
command as well as PID. The observed log had four completed generations,
13/24 training successes and mandatory nominal success; no final independent
result existed at inspection. Eighteen control-contract unit tests passed.

R76 uses the frozen pre-R74 checkpoint, seed 176, up to 32 generations,
population 24, six CPU workers and twelve-stale stopping. Its automatic paired
qualification reserves complete-fall seeds 778000 through 778039, excluded
from fitting, and requires 30 continuous strict-standing seconds in a
35-second final hold with the original physical audit. Check its actual
process, log, checkpoint and final results first on continuation. Running
training and short training success are not a stage gate or hardware readiness.
R67, R70, R73, R74 and R75 are finished. No real-robot connection or deployment.
See `GETUP_EARLY_TIMING_R75_R76_20261001.md`.

## Previous continuation R74 finished and R75 prepared

R74 ended after 21 generations, with twelve stale generations and 14/24
short training successes. Independent full-fall seeds 777000–777039 qualified
**14/40 versus 13/40** for the frozen baseline; there were four rescues and
three regressions. All forty candidate physical audits were valid and the
nominal case passed. This small paired difference is not a reliable skill
or a stage gate pass. R74 must not be restarted or deployed.

In the saved training replays, all ten remaining candidate failures are
already side-on by the end of the home wait. R75 prepares a single-factor
early-motion timing probe, returning to the pre-R74 baseline instead of
combining interventions. Its source and tests are uploaded but remote test
commands twice timed out in automatic approval; no launch is confirmed.
Before proceeding, verify tests, actual processes and output existence.
See `GETUP_EARLY_TIMING_PREPARATION_R75_20261001.md`. Do not describe R75
as running or repeat R74. R67/R70/R73/R74 are all finished.

## Previous continuation: R73 finished; R74 launch (historical)

R73 ended after 16 generations with 13/24 short training successes. Its
retained 32 pose-correction parameters are all zero. Paired 30-second strict
qualification on new full-fall seeds 776000–776039 was **10/40 versus 10/40**;
nominal passed, no improvement or stage gate pass. Do not restart R73.

R74 tests smooth wait-phase target corrections after a training-only audit
found ten of eleven failures already tilted at the end of that wait. It
changes only ten bounded joint offsets during the wait, not physics, timing,
earlier/later commands or IMU gains. New independent seeds 777000–777039 are
reserved for automatic long-hold qualification. See
`GETUP_WAIT_POSE_R74_20260930.md` for the hypothesis and frozen contract.

Next check the actual process, log, checkpoint and results in
`/data/shijinsheng/open_duck/outputs/getup_wait_pose_r74_left_20260930`
and its matching `.log`. It is an active experiment, not a qualified skill.
R67, R70 and R73 are finished. No hardware connection or deployment.

## Previous continuation: R67/R70 finished; R73 launch (historical)

R67 completed 24 generations, retained 6/8 short training successes, but
independent complete-fall 30-second standing qualification was **0/20**, versus
3/20 for its baseline. R70 stopped after 13 generations at **13/24** training
successes and qualified **9/40**, identical to its baseline. Neither is a
qualified replacement get-up skill. Do not restart these completed runs.

R71 recorded nominal and all 24 training trajectories, comparing post-control
joint states to a correctly time-aligned nominal reference. Joint errors were
small even in failures (phase means approximately 0.010–0.046 rad), while body
tilt deviation grew strongly. Feedback clipping was not pervasive: failed
phase 1 and phase 7 had cap-hit fractions approximately 7.2% and 7.6%, with
the other phase averages zero. These observations do not support a blanket
claim that joint tracking or residual saturation is the main failure cause.

Several failures were already in the wrong body/contact configuration during
the home wait, before the terminal flipping phases. For example, seed 769001
ended the initial motion with up-z 0.303, then the wait with up-z -0.252;
the nominal values were 0.342 and 0.487. This is a mechanistic hypothesis from
training trajectories, not proof that every failure shares the same cause.
Seed 773005 reached the terminal flip upright and failed afterward.

R72 froze commands, feedback, and physics and swept terminal phase-5 dwell
times 0.36/0.46/0.60/0.74/0.90 seconds and phase-6 times 0.60/0.80/1.00 seconds.
Each timing regenerated its nominal IMU reference by physical simulation.
Seven settings preserved nominal success; all remained **13/24** on the same
training falls. Others failed the mandatory nominal gate. No timing variant
was adopted. This training-only negative result rules out improvement from
the tested simple terminal dwell adjustments, not all state-driven timing.

R73 now tests a different intervention: learn four smooth eight-leg-joint
target correction knots over the final two seconds of initial recovery,
before the wait. Preserve all earlier commands, the wait, pulse, terminal
path, IMU gains, motor limits and physical acceptance criteria. Candidate
target deltas are bounded to 0.35 rad and then clipped to existing joint
limits; actuator target slew is still enforced at the original 50 Hz.
The nominal full-fall case must remain successful. No mid-path body-state
reset or teleportation is performed. All candidates are evaluated from
actual physically settled fallen starts, not captured near-standing states.

R73 training seeds remain 769000–769007 and 773000–773015. After bounded
CEM training (seed 173, 40-generation budget, population 24, six CPU workers,
15-stale-generation stop), automatic paired qualification uses **new seeds
776000–776039**, excluded from training, with a 35-second final hold and at
least 30 continuous strict-standing seconds. Preserve all failed traces.
Previous test sets inspected in R67/R70 are not reused as fresh independent
validation. Fifteen control-contract unit tests passed before launch; this
does not establish physical skill success.

Next continuation should check `outputs/getup_prefix_pose_r73_left_20260930`
and its matching `.log` first, including actual process, checkpoint and
results. It is a running experiment, not a proven improvement. Its launch
PID was 1630476; identify by command as well because PIDs can be reused.
Check for a final result before restarting or launching another run. Publish
the final checkpoint, independent results, physical audits and necessary
failure traces when complete. Current completed diagnostic outputs are
`getup_tracking_audit_r71_20260930` and `getup_terminal_timing_r72_20260930`.

The sections below record the earlier R69/R70 launch state historically;
the latest status above supersedes any earlier statement that R67/R70 run.

## Evidence and limits

This work is simulation-only, remains separate from deployed walking control,
and does not authorize hardware deployment. The full get-up skill is not yet
complete. The user's request to keep improving the skill is implemented with
bounded training runs followed by independent qualification, not a claim that
any finite experiment proves perfection.

R67 was still running when this continuation started, with 6/8 training falls
successful. Its original command prefix contains a 31-second constant home
wait inherited from an earlier failure-validation rollout. R69 explicitly
verified that this entire removed segment consisted of identical home commands
before changing its duration. Other commands, actuator limits, collision model,
and strict-standing thresholds were unchanged.

## R69 training-only wait ablation

Freeze an R67 checkpoint, regenerate each nominal reference by physical
simulation, then test the same eight previously used R67 training seeds.
Only the wait duration differs; no intermediate body-state reset is used.

| Home wait | Nominal successful | Training successes |
|---|---|---:|
| 31 s | yes | 6/8 |
| 5 s | yes | 4/8 |
| 2 s | yes | 6/8 |
| 1 s | no | 0/8 |
| 0.5 s | yes | 5/8 |
| 0 s | yes | 4/8 |

Select 2 seconds as the shortest wait preserving both nominal success and the
highest observed training success count. This removes 29 seconds of simulated
waiting per rollout. It is a **training-only ablation**; the six-of-eight
result is not independent validation and does not prove the skill reliable.
The non-monotonic results also show that a shorter wait cannot be assumed safe
without a physical rollout.

## R70 expanded short-path training

New script: `diagnostics/train_getup_shortpath_feedback_r70.py`.

- Freeze the R69 selected prefix targets, R59 terminal target offsets, and R64
  terminal feedback gains. Train prefix IMU feedback gains only.
- Full left-side fallen starts: the original eight training seeds plus sixteen
  new training seeds, with the nominal fallen start as a mandatory success.
- Train split: 769000–769007 and 773000–773015. Independent test split:
  774000–774039, excluded from training and candidate selection.
- CEM: population 24, at most 32 generations, six CPU workers; stop after
  twelve generations without objective improvement rather than endlessly
  increasing the step count.
- Physical collision, force, joint-position and target-slew audits remain in
  force. No intermediate root-state reset or nominal-transition teleport.
- Post-training qualification: forty independent complete fallen-start trials
  plus nominal, paired with the frozen R69 warm-start baseline. Each must
  complete the rollout and end with at least thirty continuous strict-standing
  seconds in a 35-second final hold.
- Contract and checkpoints are saved from the start. A running checkpoint is
  not a completed or validated skill.

Eight control-contract unit tests passed (three wait-compression tests and
five reference-feedback tests). These tests establish code contracts only,
not physical recovery success.

At the first observed R70 generation, 13/24 training falls passed and the
nominal case remained successful. Training and independent qualification were
still pending at this archive; no success claim is made for the forty test
seeds or for any other fall orientation.

## Continued-work handoff

Server project: `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`.
Use its existing virtual environment. Current outputs:

- `outputs/getup_fullfall_tracking_r67_left_20260930`: earlier full prefix,
  finished; preserves the twenty-seed 770000–770019 held-out split.
- `outputs/getup_fullfall_tracking_r67_pin_r69_20260930`: frozen input for R69.
- `outputs/getup_wait_compression_r69_20260930`: finished wait ablation and
  `selected_prefix.npz`.
- `outputs/getup_shortpath_feedback_r70_left_20260930`: finished expanded
  short-path training and long-hold qualification; 9/40, not selected.

An hourly continuation is attached to the existing chat. Check live processes
before launching anything; do not duplicate runs or kill other users' jobs.
Investigate failures using saved traces and controlled ablations, keep fresh
test splits, save new source/configuration/checkpoints/results to GitHub, and
preserve the dirty local checkout. If the left-side gate passes, broaden
independent testing before developing and testing other fallen orientations.

The established staged completion gate is at least 18/20 independent full-fall
perturbed starts for each of left side, right side, prone and supine, with
thirty continuous strict-standing seconds and valid physical audits. This is
a staged simulation gate, not universal perfection. Larger unseen test sets,
sensing and execution-delay robustness, hardware-model agreement and explicit
real-robot safety validation remain additional gates. Never enable or deploy
the skill on the real robot from a scheduled continuation.

## Latest continuation: R80 (2026-10-01)

R76--R79 are complete. R76's timing change and R78's wait selector did not
improve their respective paired baseline. The current live experiment is
R80 early leg-pose correction, not another R67/R70 run. Read
`GETUP_EARLY_POSE_R79_R80_20261001.md`, then inspect
`outputs/getup_early_pose_r80_left_20261001` and its actual process/log before
launching anything. Its first generation reproduced the frozen 13/24 short
baseline, preserved nominal success and saved a checkpoint. This is not
independent qualification. Forty fresh 780000--780039 full-fall tests with
thirty continuous strict-standing seconds run automatically after fitting.
No hardware deployment is authorized by this continuation.
