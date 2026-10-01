# Open Duck initial encoder compensation experiment

R95 finished without improving the 13/24 development baseline. R96 tests
whether a small correction based on the actual initial joint encoders can
reduce sensitivity during the first ground-contact transition. This is
a simulation-only hypothesis, not a demonstrated cause or validated skill.

## Evidence and proposed correction

A read-only reconstruction of the twenty-four original physically settled
development falls found small differences from the nominal joint readings.
The largest absolute difference among the ten controlled joints was about
0.00710 rad. R92 previously showed that even smaller head command changes
can change ground contacts before gyro and feedback diverge. Those two
observations motivate the test but do not establish that initial encoder
differences cause the failures. R95 already rejected a shared fixed preload.

Each trial reads its ten actual initial joint encoders once and subtracts
the frozen nominal encoder vector recorded from a separate physical fall.
One shared gain scales the eight leg differences, and another scales neck
and head pitch. A half-cosine envelope fades the correction to zero over
0.4, 0.8 or 1.2 seconds. Gains are chosen from 0, 1, 0.5, minus 1 and 2,
giving seventy-three settings including one literal unchanged baseline.
Both signs are tested rather than assuming the compensation direction.

The controller uses no seed, outcome label, future state or root pose as
input. It does not align or reset the robot to the nominal pose. Every
trial starts from its complete actual settled fall and continues without
state resets. Reference targets, nominal IMU reference, feedback gains,
phase timing, collision, mass, friction, joint range, torque and target
speed limits remain unchanged. The correction shares the existing total
0.18 rad residual cap with feedback; it does not add actuation budget.

## Development and independent validation

The deterministic grid uses six CPU workers and the existing development
seeds 769000 through 769007 and 773000 through 773015. Every setting is
evaluated on nominal and all twenty-four starts, with complete trajectories
and physical audits saved. Identity must reproduce nominal success and
13/24. Only nominal-preserving settings exceeding that count can advance.

At most three improving settings enter the five-arm micro-head-command
bias comparison against baseline. Candidate mean and worst recovery counts
must not be worse, and every biased nominal success of baseline must be
retained. This screen is explicitly comparative development evidence:
it does not require baseline-failing biased nominal cases to pass, and
does not establish sensor or actuator robustness. In particular it must
not be reported as the stricter all-nominal-bias gate used in R94 and R95.

A promoted setting then runs paired baseline and candidate trials on fresh
seeds 796000 through 796039. Identical actual initial-state hashes are
required. A thirty-five-second final hold is evaluated for thirty
uninterrupted strict-standing seconds with the original physical audit.
Rescues and regressions and all paired failed trajectories are preserved.
Without development promotion, these fresh seeds remain unused.

All original left, right, prone and supine acceptance gates remain required,
including at least 18/20 unseen disturbances per posture, larger independent
sets and sensor/execution-delay tests. A short development tail or training
count is not qualification. The walking controller and hardware are outside
this experiment and must not be changed or enabled.

## Execution state

Forty-seven code-contract and regression tests passed before launch. After
increasing the connection timeout, a live server check found no earlier
get-up job and a clean publication working tree. R96 then launched with
one main process and six CPU workers. Identity reproduced nominal success
and 13/24; the first five nonidentity settings recovered 6, 7, 10, 11 and
9 of twenty-four starts. This is initial negative development evidence,
not a final result. Verify actual process identity and saved progress before
any continuation; do not duplicate a healthy running job.

Source is `diagnostics/probe_getup_initial_encoder_r96.py`; tests are
`diagnostics/test_getup_initial_encoder_r96.py`; launcher is
`scripts/launch_getup_initial_encoder_r96_20261001.sh`. Planned output is
`/data/shijinsheng/open_duck/outputs/getup_initial_encoder_r96_left_20261001`.
