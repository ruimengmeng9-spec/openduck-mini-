# R84/R85 evidence and R86 early head/neck coordination training

Simulation only. Hardware debugging remains paused; no robot connection,
motor enable, deployment or walking-controller change is allowed.

## Evidence and controlled hypothesis

R84's extra home dwell did not yield an eligible alternative. Every nonzero
dwell failed nominal substep self-penetration audit, and therefore no training
case or independent qualification rollout was performed for those settings.
The unchanged zero-dwell baseline remains 13/24 short training recoveries.

R85's read-only substep observer identified trunk-to-knee/ankle contacts in
three rejected dwell references. It preserved the original nominal result
exactly. Two existing valid failed training paths had no such deep contact.
Thus the contact issue explains the tested dwell failures, not a universal
cause of insufficient recovery. No physical limit or collision is changed.

Early-window R80 pose search and R81 sensor feedback deliberately froze the
head joints. R73's correction was later in the path, 4.16--6.16 s. R86 tests
whether head/neck coordination during 0.5--3.5 s can improve early body
rotation or mass/support coupling while keeping every leg reference target
fixed. This is a new controlled variable, not a proven diagnosis or a claim
that additional head motion must work.

## Training contract

Learn four offsets: neck_pitch and head_pitch at 1.5 and 2.5 s, with zero
offset at 0.5 and 3.5 s. Smooth interpolation returns to the original path
outside that window. Offset search bounds are +/-0.35 rad; actual commands
are still clipped to original joint limits and sent through the original
50 Hz slew limiter, torque limits and physical substep audit. This search
bound does not increase a physical joint or velocity limit.

Original leg commands, other head joints, initial settling, phase durations,
existing feedback gain values, collisions and standing gates stay frozen.
Nominal sensor references for each candidate are generated through actual
physical simulation. Rejected reference traces and failed nominal trials
are saved, not hidden. A trial restores only its own actual initial fall;
no favorable intermediate state or velocity is injected.

Training uses the existing 24-case split, fixed seed 186, population 24,
six CPU workers and at most 24 generations. Twelve stale generations stop
the search. The first population includes explicit +/-0.025 and +/-0.075
single-variable probes. Zero head offsets must reproduce nominal success
and 13/24 before fitting proceeds. Actual success count is prioritized
lexicographically above continuous reward, with the mandatory nominal gate.
All candidate parameters and summaries are retained, alongside atomic
checkpoints, search history, RNG state, executed sources and hashes.

## Qualification and continuation

Training's one-second tail is not qualification. After fitting, preserve
baseline/candidate training traces, including failures. Only a higher
training count promotes the candidate to forty fresh independent full-fall
seeds 786000--786039 and nominal, paired from identical initial snapshots.
Require full execution, unchanged physical audit and thirty continuous
strict-standing seconds in a 35 s final hold. Save paired failures and
report rescues and regressions, not just aggregate counts. If training count
does not improve, independent seeds stay unused.

Twenty-three contract/regression tests passed before launch. Inspect actual
process identity, current logs and checkpoints before a new experiment.
The launch wrapper printed PID 2032363, which is not permanent identity.
R67--R85 are complete; do not resume their old PIDs. No four-pose gate or
hardware qualification has passed. Preserve the staged 18/20 fresh full-fall
gate per pose, then expand unseen and noise/delay tests; never deploy.

Output: `/data/shijinsheng/open_duck/outputs/getup_early_head_r86_left_20261001`.
Source: `diagnostics/train_getup_early_head_r86.py`.
Launcher: `scripts/launch_getup_early_head_r86_20261001.sh`.

## Completed result (supersedes running wording)

R86 completed thirteen generations, retained exactly zero head offsets,
and remained at 13/24. Twelve stale generations ended the search. Training
review reproduced the retained baseline. No increased training count was
found, so independent seeds 786000--786039 were not simulated.

There are 312 saved candidate summaries, of which 98 pass the nominal gate.
Eligible nonidentity candidates reach at most 10/24, below the common
baseline. Their successes collectively include all eleven baseline failures,
but that is an oracle union over different actions and future outcomes,
not a causal controller or 24/24 common-policy success. There is no ground
to claim improvement, qualification or hardware readiness. Rejected nominal
reference traces and paired retained-policy training failures are preserved.

Next: `GETUP_HEAD_FEEDBACK_R87_20261001.md`, testing current-IMU-dependent
head response instead of universal head offsets. Check actual R87 state
before a new launch. R86 is complete and must not be restarted.
