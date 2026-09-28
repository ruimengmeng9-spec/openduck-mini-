# Backward drift: new-method investigation — 2026-09-28

This turn performed diagnostic simulation and literature review, not new PPO
training or production-controller changes. Friction, physical joint limits,
reference motion and actor weights were not modified by the timing probe.

## A concrete train/runtime difference

In `joystick.py`, `step` constructs the next observation before shifting
`last_act`, `last_last_act`, and `last_last_last_act`. In the native
`DuckSimulation` loop those history variables have already been updated when
the next policy observation is constructed. Consequently the three action
blocks at observation indices 41:83 differ by one control tick (20 ms).
The motor-target block and gait phase are separate and are not delayed by the
probe. This does not establish that every other part of the two loops matches.

`diagnostics.probe_residual_history_timing` compares untouched native history
against a shim that delays only those three blocks. A unit test verifies that
it neither changes the input in place nor alters any other observation block.
The warm-start probe cannot recover the fourth pre-probe action and uses zero
for that first sample only. Cold-start history is zero, eliminating that
particular first-sample ambiguity.

## Results

Mean signed heading changes below are unwrapped, sampled every 0.2 s; speed is
net displacement projected onto the initial heading divided by elapsed time.
All rows in these experiments remained upright. Sampling minimum up-vector
every 0.2 s is not the same as checking its minimum at every physical substep.

| Actor / protocol | History | Runs | Mean speed m/s | Mean heading change deg |
|---|---|---:|---:|---:|
| R2, warm-up 3 s, rollout 10 s | Native | 3 | -0.085236 | -25.476 |
| R2, warm-up 3 s, rollout 10 s | Training-style lag | 3 | -0.089864 | -57.996 |
| R3, warm-up 3 s, rollout 10 s | Native | 3 | -0.038976 | -142.437 |
| R3, warm-up 3 s, rollout 10 s | Training-style lag | 3 | -0.095559 | -32.126 |
| R3, no warm-up, rollout 10 s | Native | 3 | -0.042232 | -141.920 |
| R3, no warm-up, rollout 10 s | Training-style lag | 3 | -0.094993 | -35.745 |
| R3, warm-up 3 s, rollout 30 s | Training-style lag | 5 | -0.058285 | -108.190 |

The R3 regression is strongly sensitive to action-history timing, including
under cold start. Yet aligning history is not a universal fix (R2 worsens),
and R3 still curves substantially over 30 seconds. Do not deploy the shim or
promote R3 on the basis of its improved short rollout. This is evidence for
an interface sensitivity, not a proven single cause of all backward drift.

## Recommended next experiment: observable direction correction

1. Define one explicit observation/control-tick contract and check it in both
   engines. Include history age, phase, sensor timestamps, target slew and
   action-before/after-physics conventions; do not change all of them at once.
2. Preserve the stable R2 gait as a frozen baseline. Add a small direction
   correction branch initialized to zero, rather than relearning the whole
   gait while increasing heading penalties.
3. Feed relative target-heading error as sine/cosine, yaw rate and projected
   gravity to the corrector; make direction error visible to the critic too.
   Current 101-dimensional actor observations have instantaneous gyro but not
   accumulated heading error. This limits direct error-based correction, but
   does not imply that a reactive yaw-rate policy can never walk straight.
4. Bound corrections and retain joint-limit projection and 50 Hz target slew.
   Use leg/contact-aware corrections, not arbitrary enlargement of yaw commands
   in an actor that was trained only on zero-yaw backward commands.
5. Train with paired positive/negative heading errors while holding terrain
   friction fixed. Test zero-input behavior preservation, correction from both
   signs, 10/30/60-second rollouts, held-out seeds and stop/restart.

This branch and observation extension have **not** been implemented/trained by
this investigation. Relative simulated heading is available now; a hardware
implementation needs a consistent relative-yaw estimator (gyro integration has
drift), not an assumption that a six-axis IMU provides perfect absolute yaw.

## Other methods worth testing

- **Symmetry during learning, not inference-time action averaging.** Augment
  paired observations/actions or use a soft mirror loss, with correct joint
  parity and gait-phase transformation. First check robot/reference symmetry;
  the reference, joint limits and neutral posture need not be exactly symmetric.
  Averaging two opposite-leg gait actions can suppress the gait itself and is
  not equivalent to symmetry-aware PPO.
- **Native-engine constrained teacher search.** Optimize a low-dimensional,
  bounded steering/phase correction around R2 in native MuJoCo, then distill a
  teacher only after it passes stable straight-backward tests. This is a
  proposed fallback for a persistent MJX/native gap, not a tested solution.
- **Retarget the reference coherently.** Earlier knee-limit violations warrant
  a joint/velocity-limit-constrained gait reference rather than independently
  clipping peaks forever. Retargeting must preserve foot contacts and support
  balance. It is distinct from reversing a forward gait in time, which is not
  guaranteed dynamically feasible in contact-rich locomotion.

## Primary-source reading

- [RuN: Residual Policy for Natural Humanoid Locomotion](https://arxiv.org/html/2509.20696v1):
  separates a frozen motion prior from learned dynamical corrections and uses
  proprioceptive/task observations. Our small steering branch is an adaptation
  of the decoupling principle, not a reproduction or guaranteed transfer of its
  Unitree G1 results.
- [Symmetry Considerations for Learning Task Symmetric Robot Policies](https://arxiv.org/html/2403.04359v1):
  studies mirror loss and symmetry-based augmentation, including limits and
  trade-offs when symmetry is only approximate.
- [Leveraging Symmetry in RL-based Legged Locomotion Control](https://arxiv.org/html/2403.17320v2):
  compares symmetry-aware augmentation and equivariant architectures in legged
  control. Results on those robots do not establish Open Duck performance.

Raw results: `results/history_timing*_20260928.json`. Existing agent/hardware
configuration remains unchanged.
