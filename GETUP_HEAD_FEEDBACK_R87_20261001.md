# R86 negative outcome and R87 causal head feedback

Simulation only. No real-robot connection, enable, deployment or change to
the original walking controller. No physics or acceptance relaxation.

## Evidence and new controlled hypothesis

R86 early head/neck pose fitting ended after thirteen generations with
exactly zero offsets and 13/24 training recoveries. No held-out qualification
was run. Across 312 candidates, 98 passed the mandatory nominal gate;
nonidentity eligible candidates achieved at most 10/24, yet collectively
rescued each of the eleven original failures at the cost of other cases.
This oracle diversity motivates a state-dependent test; it is not proof
that a causal selector exists or that the common policy is 24/24 reliable.

R81 tested only added leg feedback and kept head joints fixed. R87 tests
early head/neck feedback rather than fixed head offsets: a learned 2-by-4
matrix maps *pre-control* deviations from the frozen nominal sensor reference
to neck_pitch and head_pitch corrections. Inputs are body upvector X/Y and
0.15-scaled gyro Y/X errors. No root state, contact force, future observation,
trial seed, outcome label or privileged state enters the deployed policy.

## Controlled implementation

The same envelope is zero at 0.5 s, one at 1.5 and 2.5 s, and zero at
3.5 s. Gain search bounds are +/-2. Corrections are added to the original
feedback and clipped together to the unchanged +/-0.18 rad residual cap.
No added actuation budget is created. All reference targets, timing, initial
settling, leg feedback gain values, collision, torque, joint limits and
50 Hz target-slew limit stay unchanged. Existing leg feedback may respond
differently to an altered body state, but its law and reference stay fixed.

Zero gains delegate literally to the original evaluator and must reproduce
nominal success and the completed R86 baseline 13/24 before fitting continues.
Every trial restores only its own complete physically fallen initial state;
no midpath state reset or reference-state injection is allowed.

## Training and qualification

Existing 24 training seeds, fixed search seed 187, population 24, six CPU
workers, at most 24 generations and twelve stale-generation stopping.
The first population contains each positive and negative single-gain probe
at magnitude 0.4. Nominal eligibility, then actual success count, then
continuous quality define the ranking; reward cannot replace a recovery.
Candidate parameters and summaries, rejected nominal traces, final selected
training failures, checkpoints, RNG state and source/model hashes are saved.

Training's one-second strict tail is not qualification. Only improved
training count promotes the candidate to forty fresh full-fall seeds
787000--787039 plus nominal, paired with the frozen baseline from identical
initial states. Require complete continuous execution, valid physical audit,
and thirty uninterrupted strict-standing seconds in the 35-second final
hold. Save both branches' failed trajectories and paired changes. Otherwise
the independent test seeds remain unused.

Twenty-five code-contract/regression tests passed before launch; these do
not demonstrate physical recovery or readiness. Inspect actual process,
logs, checkpoint and result before another launch. Wrapper PID at launch
was 2042822, not permanent process identity. R67--R86 are complete.
No per-pose or complete four-pose qualification has passed. Maintain the
existing 18/20 fresh full-fall gate per pose, then larger unseen/noise/delay
tests; never automatically deploy.

Output: `/data/shijinsheng/open_duck/outputs/getup_head_feedback_r87_left_20261001`.
Source: `diagnostics/train_getup_head_feedback_r87.py`.
Launcher: `scripts/launch_getup_head_feedback_r87_20261001.sh`.

## First observed generation

Generation one reproduced 13/24 and nominal success. All twenty-four
candidate feedback laws passed the nominal gate, unlike the 98/312
eligibility in R86's fixed offsets. The retained head gain is zero so far;
eligibility alone is not a recovery improvement. Training and independent
qualification remain unfinished at this observation.

## Completed result

R87 ended after thirteen generations and twelve stale generations. The
retained gains are exactly zero, with 13/24 development recoveries and
nominal success. All 312 candidates passed the nominal gate, but no
nonidentity candidate exceeded 13/24. Selected and baseline development
replays agree, with all twenty-five physical audits valid. Because recovery
count did not improve, the reserved 787000 through 787039 independent seeds
were not used. No new thirty-second qualification or hardware readiness is
established.

Different candidates collectively rescued all eleven baseline failures.
Their union is an outcome-aware oracle, not a runnable 24/24 policy. R88
therefore screened an observable selector; its best development result was
only 9/24. R89 instead tests nonlinear head-feedback distillation from the
successful case-specific teachers. Inspect the latest continuation note
before starting another experiment. R87 is finished and must not be resumed.
