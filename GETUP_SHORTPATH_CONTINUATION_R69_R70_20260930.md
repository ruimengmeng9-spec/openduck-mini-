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

## Latest continuation: R81 (2026-10-01)

R80 is complete: thirteen generations retained zero correction, with paired
13/40 independent long-hold successes and no rescues or regressions. Current
training is R81 early asymmetric IMU feedback, described in
`GETUP_EARLY_FEEDBACK_R81_20261001.md`. Inspect
`outputs/getup_early_feedback_r81_left_20261001` and live processes. Its
forty fresh test seeds are 781000--781039, not the previously inspected R80
test cases. The combined residual cap, physics and strict gates are unchanged.
Do not restart R80 or rely on older heartbeat wording about it being live.

## Latest continuation: R82 (2026-10-01)

R81 is complete: thirteen generations retained zero added gains, training
13/24 and fresh paired long-hold qualification 8/40 for both branches. No
paired recovery changed. R82 is a new bounded causal IMU reference-clock
alignment probe, described in `GETUP_SENSOR_CLOCK_R82_20261001.md`.
Inspect `outputs/getup_sensor_clock_r82_left_20261001`, its actual process
and result before acting. Only a training-count improvement promotes it
to fresh paired 782000--782039 continuous 30-second qualification; otherwise
those seeds remain unused. Do not restart R81 or older completed jobs.
All prior hardware prohibitions and physical/standing gates remain in force.

## Latest continuation: R84 (2026-10-01)

R82 completed without improvement: identity 13/24, adaptive clocks 1--5/24.
R83's offline causal selector also failed its training-only leave-one-out
screen (best 9/24); the oracle 20/24 is not a policy. Neither used fresh
qualification seeds. Current experiment R84 tests physical initial settling
before the unchanged complete get-up stream; read
`GETUP_INITIAL_SETTLING_R84_20261001.md` and inspect
`outputs/getup_initial_settling_r84_left_20261001` before launching anything.
Only training-count improvement promotes it to fresh paired 784000--784039
continuous 30-second qualification. No intermediate state reset, hardware
connection, physics changes or relaxed acceptance are allowed. R67--R83
are complete; never resume their old PIDs.

## Latest continuation: R86 (2026-10-01)

R84 is complete: extra initial dwell failed nominal reference physical audit
at all nonzero settings; training cases for those settings and fresh test
seeds were not simulated. R85 observed actual transient trunk-to-knee/ankle
substep contact, explaining those rejected references but not all valid
failed get-up cases. R86 trains early head/neck target coordination with
all leg references and physics frozen. Read `GETUP_EARLY_HEAD_R86_20261001.md`
and inspect `outputs/getup_early_head_r86_left_20261001`, its actual process,
log and checkpoint. Twenty-four generations maximum, seed 186, six CPU
workers. Only increased training count triggers fresh paired 786000--786039
continuous thirty-second qualification. R67--R85 are finished. Keep every
hardware prohibition, no-intermediate-reset requirement and physical gate.

## Latest continuation: R87 (2026-10-01)

R86 is complete: thirteen generations retained zero early head offsets,
training stayed 13/24 and independent seeds were unused. Saved candidates
rescue different failures but do not form a common improved policy. R87
tests causal early head/neck feedback from current IMU reference errors;
read `GETUP_HEAD_FEEDBACK_R87_20261001.md` and inspect
`outputs/getup_head_feedback_r87_left_20261001`, current process, log,
checkpoint and final result. Search seed 187, six CPU workers, population
24, at most 24 generations; twelve stale generations stop. Only training
count improvement triggers fresh paired 787000--787039 complete falls,
valid physical audit and thirty continuous strict-standing seconds.
R67--R86 are finished; do not resume old PIDs or duplicate live R87 work.
All hardware prohibitions, unchanged physics and no-midpath-reset rules
remain in force. Training or oracle counts are not independent qualification.

## Latest continuation R89 2026 10 01

R87 is finished after thirteen generations, retaining zero gains and 13/24
development success. Reserved 787000 through 787039 independent seeds were
unused. R88's observable candidate selector also failed its development
screen, best 9/24. Do not restart either job or any earlier completed run.

Current R89 trains nonlinear head feedback from successful R87 teachers.
Read `GETUP_HEAD_DISTILLATION_R88_R89_20261001.md` and inspect
`outputs/getup_head_distillation_r89_left_20261001`, its actual processes,
log, saved networks and results. Seed 189, six CPU workers, 300 supervised
epochs and nine closed-loop settings. Teachers reproduced 25/25 short
development successes; this is NOT common-policy or thirty-second success.
Only improved closed-loop training count above 13/24 with nominal preserved
triggers fresh paired 789000 through 789039 complete-fall qualification.
All physical audit, thirty-second standing, hardware prohibition and
no-midpath-reset requirements remain unchanged. Avoid duplicate launch.

## Latest continuation R91 2026 10 01

R89 is finished: case-specific teachers reproduced 25/25 short-gate success,
but the common student achieved at most 9/24. R90 preserved the complete
nominal trajectory exactly by output centering, yet achieved at most 7/24.
Neither passed development promotion, so their independent seeds remained
unused. R67 through R90 are complete. Never resume their old process IDs.

Current R91 is a seventeen-arm head-command tolerance diagnostic, not a
promotion search. Read `GETUP_HEAD_TOLERANCE_R91_20261001.md` and inspect
`outputs/getup_head_tolerance_r91_left_20261001`, its actual processes, log,
progress and results. It uses the same nominal and twenty-four development
falls, six CPU workers, and separate neck/head raw biases of plus/minus
0.0005, 0.001, 0.002 and 0.005 radians under the original envelope and cap.
Identity must reproduce 13/24. Retain both rescues and regressions. No fresh
validation seeds, hardware action, physics change or relaxed gates are
authorized by this diagnostic. All original four-pose and thirty-second
qualification requirements stay in force. Inspect results and design a
different evidence-based training method rather than duplicate a live job.

## Latest state after R91 completion 2026 10 01

R91 finished all seventeen diagnostic arms. Identity remained 13/24; only
three of sixteen nonzero settings retained nominal success, and none
exceeded 13/24 development recovery. A head-pitch bias of minus 0.0005 rad
rescued six cases but lost six, so equal count is not equal reliability.
This supports a narrow command-tolerance problem, not a validated fix.
All R67 through R91 jobs are finished; the final process check found no
live get-up job. Do not resume old PIDs or duplicate any completed run.

Next read complete R91 results and paired traces, locate when tiny command
deviations alter body motion and contact/support transition, and distinguish
that mechanism from sensor-selection ambiguity or student distribution
shift. Design a genuinely new robust whole-path or closed-loop training
hypothesis from this evidence; do not simply increase epochs or repeat
head-gain search. Preserve full actual-fall starts, every physical limit,
independent seed separation and all thirty-second/four-pose acceptance
conditions. No hardware action or controller replacement is authorized.

## Latest continuation R93 2026 10 01

R92 exactly replayed nine R91 trajectories while recording actual physical
substep contacts. Early head-ground force and topology differences appeared
around 0.62 through 0.77 s, before gyro and leg-feedback divergence. This
does not prove gyro feedback caused the initial change. R67 through R92 are
complete; do not resume their old process IDs.

Current R93 independently attenuates early pitch-rate and roll-rate feedback
over a twenty-five-setting deterministic grid. Read
`GETUP_MICRO_CONTACT_R92_GYRO_R93_20261001.md` and inspect
`outputs/getup_gyro_attenuation_r93_left_20261001`, actual processes, log,
grid progress, checkpoint and results. Six CPU workers; existing twenty-four
development starts. Identity must reproduce 13/24. Only actual count
improvement preserving nominal enters a five-arm micro-command-bias screen;
only no-worse mean and worst-case results promote to fresh paired 793000
through 793039 full-fall, thirty-second strict-standing qualification.
Do not call training scores, contact temporal ordering or launch a recovery
success. No hardware action, physics change, larger actuation budget or
midpath state reset is allowed. Inspect and wait for a healthy live job;
do not duplicate it or blindly restart older heartbeat-listed experiments.

## Latest continuation R94 2026 10 01

R93 is complete: all twenty-five nominal settings passed the short gate,
but no nonidentity setting exceeded baseline 13/24. Best nonidentity was
11/24; no bias screen or fresh 793000 cohort was run. R67 through R93 are
finished and must not be resumed using stale process IDs.

R94 tests independent head/neck lead and lag relative to unchanged leg
commands from 0 to 1.2 seconds. Read
`GETUP_HEAD_RELATIVE_PHASE_R94_20261001.md`, then inspect
`outputs/getup_head_phase_r94_left_20261001`, current processes, log,
grid progress, checkpoint and results. Twenty-five prescribed settings,
six CPU workers, and the same twenty-four development starts. Only count
improvement preserving nominal enters a development micro-bias robustness
screen; only no-worse mean and worst recovery plus all nominal successes
promote to fresh paired seeds 794000 through 794039 with thirty-second
strict continuous standing. All physical limits and four-pose acceptance
conditions remain unchanged. No hardware action is authorized. Do not
duplicate the running experiment or restart older heartbeat-listed jobs.

## Latest continuation R95 2026 10 01

R94 is now complete and rejected. Identity remained 13/24; nine eligible
nonidentity settings recovered only 2 to 8 of twenty-four development
cases. Fifteen other settings failed nominal and were not evaluated on
the development cohort. No micro-bias screen or fresh 794000 qualification
cohort ran. R67 through R94 are finished; do not resume their old PIDs.

Read `GETUP_COORDINATED_PRELOAD_R95_20261001.md` and inspect
`outputs/getup_preload_r95_left_20261001`. R95 jointly trains ten leg and
head/neck offsets over 0 to 0.9 seconds, while allowing nominal standing
failures only for intermediate search scoring, never selected acceptance.
Six CPU workers, seed 195, population 24, thirty-two-generation budget.
Each physically valid nominal reference is evaluated on all twenty-four
development starts; selected candidate must retain nominal and improve
13/24 before micro-command robustness screening and any paired fresh
795000 through 795039 thirty-second qualification. Check live identity,
log, history, checkpoints and results before continuing. No physics-limit
or gate changes, midpath state resets, hardware connection or controller
replacement is authorized. Wait if live and healthy; do not duplicate.

## Latest continuation R95 final and R96 running 2026 10 01

R95 completed seventeen generations with no feasible improvement. Selected
preload was exactly zero; both final reviews recovered 13/24 with nominal
success and valid physical audits in all twenty-four development cases.
No fresh 795000 qualification or micro-bias screen ran. R67 through R95
are complete; never resume their old PIDs.

Read `GETUP_INITIAL_ENCODER_COMPENSATION_R96_20261001.md`. R96 passed
forty-seven code/regression tests and is running with six CPU workers.
Identity reproduced nominal success and 13/24. It tests shared
initial-encoder compensation gains over seventy-three settings, fading
within 0.4 to 1.2 seconds under the unchanged combined residual cap. The
largest observed settled joint deviation in the development cohort was
about 0.00710 rad; this is a hypothesis input, not proof of a root cause.
Actual full-fall starts, all physical limits and strict acceptance are
unchanged. No seed, root pose, outcome or future state is a policy input.

Check actual server processes, log, grid progress and results before any
continuation. Increasing the SSH timeout recovered access; the existing
bundle download completed. R96 main 2144366 and six workers were observed
with the expected command, but verify current identity rather than acting
on these stored IDs. Output is `outputs/getup_initial_encoder_r96_left_20261001`.
Do not infer a training crash from a connection timeout or launch a duplicate.
Only nominal-preserving count improvement can enter the comparative
five-arm bias development screen; candidate mean/worst counts and each
baseline nominal success must be retained. This is not the all-biased
nominal gate previously used in R94/R95 and is not robustness qualification.
Only a promoted candidate uses fresh paired seeds 796000 through 796039
with thirty uninterrupted strict-standing seconds. If not promoted, those
seeds remain unused. No hardware action or walking controller replacement
is authorized, and all four-pose and larger-cohort acceptance gates remain.
