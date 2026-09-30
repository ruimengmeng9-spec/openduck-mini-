# Short-wait get-up continuation R69–R70 (2026-09-30)

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
  running; preserves the twenty-seed 770000–770019 held-out split.
- `outputs/getup_fullfall_tracking_r67_pin_r69_20260930`: frozen input for R69.
- `outputs/getup_wait_compression_r69_20260930`: finished wait ablation and
  `selected_prefix.npz`.
- `outputs/getup_shortpath_feedback_r70_left_20260930`: new expanded short-path
  training, followed automatically by long-hold qualification.

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
