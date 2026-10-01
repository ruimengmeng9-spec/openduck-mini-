# Open Duck head feedback selector audit and nonlinear distillation

R87 did not improve the common get-up policy. R88 also rejected choosing a
head-feedback law once from the first half-second of sensor data. R89 now
tests a different hypothesis: a nonlinear feedback model can learn from
successful trajectories while observing the robot throughout the motion.
This is simulation-only development, not independent qualification or
permission to operate a real robot.

## Completed selector audit

R88 used 287 distinct R87 feedback candidates and the same twenty-four
development cases. The first twenty-five commands have no added head
feedback. Inputs therefore came from the baseline trajectory before any
candidate intervention: IMU reference errors at 0.1, 0.3 and 0.5 seconds,
measured joint encoders and joint velocities. Floating-base position,
orientation and velocity were discarded.

Sixty-four nearest-neighbor settings varied input sets, neighbor counts
and global-prior weights. Each left-out query was excluded from feature
normalization and success-utility fitting. The best screen achieved 9/24,
with one rescue and five regressions, below the baseline 13/24. The selector
was rejected. The library itself was previously optimized on all twenty-four
cases, so even these leave-one-out scores are development evidence rather
than independent validation. The outcome-aware candidate union of 24/24
does not represent any common causal policy.

## Distillation hypothesis and teacher evidence

R89 chooses teachers offline only. Baseline-successful cases and the nominal
case use zero extra head feedback. For each of the eleven baseline failures,
the smallest-norm audited successful R87 head-gain candidate supplies a
teacher. The teacher choice may use development outcomes; the eventual
controller cannot use a trial seed, outcome or teacher identity.

Twenty-five complete physical teacher replays reproduced their short
development success, including nominal. Their trajectories, initial-state
hashes, candidate provenance and sensor-action datasets are retained. This
25/25 figure concerns different teachers on previously used cases, with a
one-second strict tail. It is not a common policy success rate or the
thirty-second qualification result.

The student receives thirty-five pre-control values: four IMU reference
errors, fourteen joint-position offsets, fourteen scaled joint velocities,
two of its own previous head targets and elapsed time. It has a 35 by 64 by
2 tanh-hidden network and learns neck-pitch and head-pitch feedback only.
The teacher supplies a raw feedback target before the original time
envelope and combined 0.18-radian residual cap. An internal prediction
bound of two radians is not an actuator allowance: the same 0.18-radian
combined cap, joint clipping and 5.24-rad/s target slew apply to every
physical command. No contact force, root state or future trial observation
enters the student.

## Training and physical verification

Fixed training seed 189, six CPU workers and 300 supervised epochs are used.
Saved networks at epochs 50, 150 and 300 are tested at feedback strengths
0.25, 0.5 and 1.0. Loss improvement cannot promote a policy. Each candidate
must first preserve nominal success, then run from every original complete
fallen development state with no intermediate state reset. The baseline
must reproduce 13/24. Leg reference commands, original leg-feedback law,
phase timings, initial settling and all physical and strict-standing limits
remain unchanged. All closed-loop traces are saved, including failures.

Only a student with more than thirteen development recoveries proceeds to
forty fresh seeds 789000 through 789039 and nominal, paired with baseline
from matching complete initial states. Require a valid physical audit,
full uninterrupted execution and thirty continuous strict-standing seconds
in the final thirty-five-second hold. Those seeds cannot later be reused
for tuning and still described as unseen. None of the four pose gates has
passed; the existing per-pose 18/20 threshold and subsequent larger-seed,
sensor-noise and execution-delay checks remain required.

Seventeen source-contract tests passed before launch. Wrapper PID 2069015
and main PID 2069017 were observed at launch, but future checks must verify
current command identity rather than act on these numbers blindly. R67
through R88 are complete. No hardware connection, enable, deployment or
original walking-controller edit occurred.

Output: `/data/shijinsheng/open_duck/outputs/getup_head_distillation_r89_left_20261001`.
Source: `diagnostics/train_getup_head_distillation_r89.py`.
Launcher: `scripts/launch_getup_head_distillation_r89_20261001.sh`.

The latest observed stage reproduced all teachers and completed supervised
fitting. Closed-loop development screening is in progress. No student
improvement or independent qualification is claimed at this stage.
