# Open Duck early head and leg timing test

R93 did not improve the shared get-up policy. R94 tests a different,
unproven hypothesis: the early relative timing of head and leg commands
may determine which head-ground loading transition occurs. This experiment
is simulation only and has not passed independent recovery acceptance.

## Evidence and distinction from earlier searches

R92 measured actual head-ground loading and contact changes before gyro
and leg-feedback differences in exact paired replays. R93 independently
attenuated the subsequent gyro response across twenty-five settings, but
its best nonidentity result was 11/24 versus baseline 13/24. No setting
was promoted and no fresh qualification seed was simulated.

Inspection of the frozen prefix shows a head/neck hold around 0.5 to
0.6 seconds and a substantial coordinated target transition around 0.64
to 0.88 seconds, overlapping the observed contact divergence. This is
command-path evidence, not proof of a universal failure cause. Prior
R75/R76 time warps changed all joints together in four coarse quarters;
R82 used a shared sensor-dependent clock with its first 0.5 seconds fixed.
R86/R87 added head offsets or feedback beginning at 0.5 seconds. None of
those interventions independently shifted head timing relative to unchanged
leg commands from the beginning of the path.

## Prescribed intervention

Only neck-pitch and head-pitch reference cursors change. Each lead is chosen
from zero, minus/plus 0.06 seconds and minus/plus 0.12 seconds, making
twenty-five settings. A triangular-ended envelope is zero at the start,
one at 0.3 and 0.9 seconds, and zero at 1.2 seconds. Positive lead samples
the saved head trajectory earlier; negative lead delays it. Both reference
cursors remain monotonic, with rates between 0.6 and 1.4 of the original
reference clock. These are reference timing bounds, not motor-speed limits.

All nonhead reference targets and every target at or after 1.2 seconds
remain byte-identical. Interpolation follows the existing commanded head
value path, rather than inventing additional angular waypoints. All phase
durations remain fixed. The nominal IMU reference is regenerated in a
separate physical rollout for each setting. Invalid reference rollouts
retain their failure traces and are rejected before development cases.
Identity delegates to the original physical path and must recover nominal
and 13/24 before selection.

Each independent trial starts from its own actual settled full fall; no
root or joint state is injected during recovery. Feedback gains, collisions,
mass, friction, torque, joint range, combined residual cap, target slew,
control frequency and strict standing thresholds remain unchanged. No
robot connection, motor enable, deployment or original walking-controller
edit is authorized.

## Promotion and acceptance

Six CPU workers evaluate the existing twenty-four development seeds.
Complete nominal and evaluated development traces, initial hashes and
physical audits are saved. A rejected nominal setting must not be reported
as twenty-four evaluated development failures. The one-second strict tail
is only a development screen.

Only settings exceeding 13/24 while preserving nominal enter a five-arm
micro-command-bias comparison with baseline. Candidate promotion requires
nominal success on every bias arm and no worse mean or worst-case recovery
count than baseline. These nuisance tests reuse development seeds and are
not independent qualification. If no candidate passes, fresh seeds stay
unused and the retained checkpoint is identity.

A promoted candidate and baseline run paired full-fall trials on forty
fresh seeds 794000 through 794039. Initial-state hashes must match. Valid
physics and thirty uninterrupted strict-standing seconds during the final
thirty-five-second hold are required. All four pose gates, larger fresh
cohorts and sensor/execution-delay tests are still required; no model has
yet completed those conditions or been approved for hardware.

## Completed result and continuation

Thirty-six contract and regression tests passed before launch. All twenty-five
settings have now finished. Identity reproduced 13/24 with twenty-four valid
physical audits. Nine nonidentity settings preserved nominal and were each
tested on all twenty-four development starts, but recovered only 2 to 8
cases. The best nonidentity result was 8/24 with seventeen physically valid
cases; a nominal short-gate pass alone did not guarantee perturbed physical
validity. The remaining fifteen settings failed nominal standing and were
not evaluated on the development cohort. There were no reference-audit
rejections in this grid. Complete evaluated failure traces were retained.

No setting was promoted. The micro-bias screen and fresh 794000 through
794039 qualification cohort were not run, and the selected checkpoint is
identity. These results reject this prescribed relative-phase correction,
not every possible coordinated path. R94 has ended and its old process IDs
must not be resumed. Next read `GETUP_COORDINATED_PRELOAD_R95_20261001.md`.

Completed output is
`/data/shijinsheng/open_duck/outputs/getup_head_phase_r94_left_20261001`.
Source is `diagnostics/probe_getup_head_phase_r94.py`; launcher is
`scripts/launch_getup_head_phase_r94_20261001.sh`. Wrapper 2100974 was
reported at launch and is no longer active. Do not repeat head-offset,
shared-clock or attenuation searches under a new label.
