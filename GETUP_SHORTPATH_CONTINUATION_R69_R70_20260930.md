# Short-wait get-up continuation R69–R70 (2026-09-30)

## Latest continuation: R67/R70 finished; R73 running

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
