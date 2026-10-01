# Open Duck early contact divergence and gyro feedback attenuation

R92 located a physical transition that differs before large body tilt is
visible: very small head-command biases change head-ground loading and
contact patterns, followed by gyro-error and leg-feedback differences.
R93 tests whether the subsequent gyro feedback amplifies that already
changed transition. This is not a claim that gyro feedback caused the first
contact change, and no recovery skill has yet passed independent acceptance.

## Exact replay and actual contact evidence

R92 replayed nominal and development seeds 769003 and 773014 under identity,
neck bias minus 0.0005 rad, and head-pitch bias minus 0.0005 rad. A temporary
observer called the original MuJoCo integrator exactly once per substep,
then read actual contacts and forces. It changed no model arrays, state,
commands or physical-audit thresholds. All nine results and complete qpos
and qvel traces exactly matched the previously recorded R91 trajectories.
Each replay contained 6290 physical substeps.

Across the six biased comparisons, floor-force differences exceeding 3 N
for twenty milliseconds began at 0.618 through 0.734 seconds. Persistent
contact-topology differences began at 0.664 through 0.774 seconds, and
scaled gyro differences began at 0.82 through 0.96 seconds. Leg-feedback
differences exceeding 0.01 rad began at 0.82 through 2.16 seconds; large
root up-vector-Z differences appeared later, at 2.2 through 10.44 seconds.
These are diagnostic detection thresholds, not replacement standing gates.

In nominal neck-bias replay at 0.682 s, neither foot was carrying load;
head-floor normal force differed from about 12.877 N to 16.103 N. In the
head-pitch-bias replay at 0.718 s, head-floor force differed from zero to
8.824 N, while other-body floor loading also changed. This establishes
actual early head-contact sensitivity in those comparisons. The observed
time ordering alone does not prove a unique cause for every failed seed,
and it does not justify changing friction or disabling collisions.

## Controlled feedback fitting

R93 retains every reference joint target, original tilt-feedback coefficient,
phase duration and initial settling command. Only the original pitch-rate
gain columns 3 through 5 and roll-rate column 7 are multiplied by attenuation
factors. The factors vary independently over 1, 0.75, 0.5, 0.25 and 0, making
twenty-five prescribed settings. The same early envelope is zero at 0.5 s,
one at 1.5 and 2.5 s, and zero at 3.5 s. Feedback outside that window is
unchanged. Factors cannot amplify or reverse the original gains.

The scalar reference errors come from current pre-control IMU readings;
no seed, outcome, root state or future trial observation is a policy input.
Identity literally delegates to the baseline evaluator and must reproduce
nominal success and 13/24 development recoveries. A setting first needs
nominal success before its twenty-four development cases are simulated.
Rejected settings retain their nominal failure trace and must not be
described as twenty-four evaluated failures. Fixed development seeds and a
deterministic grid are used; there is no new randomized optimizer sampling.

Every trial starts from its own actual complete fallen state, with no
intermediate restore or teleport. Collision, masses, friction, torque,
joint bounds, 50 Hz target slew, combined 0.18-rad feedback cap and strict
standing thresholds stay unchanged. Six CPU workers run the experiment.
No real robot is connected or enabled, and the original walking controller
is not modified.

## Robustness screen and independent qualification

If no nonidentity setting improves on 13/24, fresh validation seeds are
unused. Otherwise, the three strongest improved settings and baseline are
compared under zero bias and separate positive/negative 0.0005-rad head and
neck biases. Every setting uses the same nominal and twenty-four development
starts. The biases are commanded-tolerance nuisance tests, not a change to
actuator limits, physical noise or acceptance thresholds. Complete traces
and both rescues and regressions are retained.

Promotion additionally requires no worse mean and worst-case development
recovery count over that bias grid compared with baseline. This short
one-second strict-tail screen is not independent acceptance. A promoted
setting runs against baseline on forty fresh seeds 793000 through 793039
and nominal, with identical initial-state hashes. Require valid physical
audit, complete continuous execution and thirty uninterrupted strict-standing
seconds in a thirty-five-second final hold. Preserve every paired trajectory.
The fresh cohort cannot subsequently be used for tuning and still be called
unseen. All four pose gates, larger independent sets and sensor/execution
delay testing remain unpassed.

## Completed attenuation result

Thirty-one code-contract and regression tests passed before launch. R93
completed all twenty-five settings. All retained nominal short-gate success;
identity reproduced 13/24 development recoveries and all twenty-four valid
physical audits. No nonidentity setting improved the count. The strongest
nonidentity setting removed early pitch-rate feedback, retained roll-rate
feedback, and recovered 11/24, with twenty-three valid physical audits.
The other settings recovered between 3/24 and 10/24; valid physical counts
ranged from twenty-two to twenty-four. Thus removing or reducing this
feedback does not establish a fix for the observed contact sensitivity.

The selected checkpoint retains identity factors. The conditional bias
screen and independent qualification were not run; seeds 793000 through
793039 remain unused. All process checks after completion found no R93
worker still running. The negative result does not prove that gyro feedback
is irrelevant in every trajectory, only that this prescribed intervention
did not improve the shared policy on the development cohort.

R67 through R93 are complete. Completed R93 output is
`/data/shijinsheng/open_duck/outputs/getup_gyro_attenuation_r93_left_20261001`.
Source: `diagnostics/train_getup_gyro_attenuation_r93.py`.
Launcher: `scripts/launch_getup_gyro_attenuation_r93_20261001.sh`.
R92 contact evidence is in
`/data/shijinsheng/open_duck/outputs/getup_micro_contact_r92_20261001`.
Next read `GETUP_HEAD_RELATIVE_PHASE_R94_20261001.md`. R94 tests relative
head-to-leg timing before the contact change, not attenuation or another
shared whole-body clock. Do not resume an old R93 PID or repeat a completed
negative grid.
