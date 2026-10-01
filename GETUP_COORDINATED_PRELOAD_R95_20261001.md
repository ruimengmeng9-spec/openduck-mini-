# Open Duck coordinated early recovery preparation

R95 trains a shared early preparation pose for legs and head/neck, beginning
before the contact divergence observed in R92. It also removes nominal
standing rejection from intermediate search scoring, while retaining it
for final candidate selection. This is a new, unproven hypothesis and not
a validated recovery skill. No hardware work is performed.

## Evidence and hypothesis

R93 gyro attenuation and R94 head-relative timing both failed to exceed
baseline 13/24 development recoveries. In R94, fifteen of twenty-four
nonidentity settings failed the nominal standing gate and therefore never
provided development-cohort feedback. This is correct for promotion but
may hide useful intermediate search information. Whether broader scoring
can improve optimization remains to be tested.

Prior R80 and R86 pose searches began at 0.5 seconds and changed legs or
head separately. R95 instead changes all ten controlled leg and head/neck
joints together from time zero to 0.9 seconds, covering the approach to
the first measured contact divergence around 0.62 to 0.77 seconds. The
working hypothesis is that shared early preparation can avoid a fragile
head-loading transition. Neither the temporal contact evidence nor earlier
failures establishes that hypothesis as fact.

## Controlled search

A single ten-dimensional angular offset is multiplied by an envelope:
zero at time zero, one at 0.3 and 0.6 seconds, and zero again at 0.9 seconds.
Each offset is bounded to 0.12 rad. The first commanded target and every
target at or after 0.9 seconds are unchanged. Motor target slew, joint
bounds, force limit, combined feedback residual cap, collision geometry,
mass, friction, feedback gains, phase duration and strict standing gates
remain frozen. The offset is a reference-pose parameter, not an increase
in the feedback or actuator budget.

A separate physical nominal rollout regenerates each reference. Invalid
reference paths are rejected and their traces saved. Otherwise every
candidate is evaluated on nominal and all twenty-four development full
falls, even when nominal does not finish standing. Intermediate CEM fitness
uses actual recovery counts, the worst six scores, mean scores and nominal
performance. Failed physical audits remain failures. No trial resets its
root or joint state after the original settled full-fall start.

The search distribution can learn from physically valid intermediate paths
which fail nominal standing, but the stored best candidate must pass nominal.
Actual success count dominates its ranking, with quality only breaking ties.
Identity is included in each generation and must reproduce nominal and
13/24 before continuing. Six CPU workers, random seed 195, population 24,
at most thirty-two generations and sixteen stagnant generations bound this
experiment. Full evaluated traces, including failures, are retained, with
source/model hashes, initial-state hashes, search and RNG checkpoints.

## Validation and acceptance

The one-second training tail is not qualification. Only a nominal-feasible
candidate exceeding 13/24 enters a five-arm micro-head-command-bias comparison
against baseline on the development cohort. All candidate nominal bias arms
must pass, and its mean and worst-case recovery counts must not be worse
than baseline. A promoted candidate then runs paired fresh full-fall tests
on seeds 795000 through 795039, with identical initial-state hashes and a
thirty-five-second final hold requiring thirty uninterrupted strict-standing
seconds. Every paired trajectory and physical audit is preserved.

If no candidate passes the development screens, fresh qualification seeds
stay unused. No acceptance threshold is changed merely because optimization
temporarily explores a failed nominal path. All left/right/prone/supine gates,
larger fresh sets and sensor/execution-delay tests remain required. The
original walking controller is not modified; no robot connection, enabling
or deployment is authorized.

## Continuation

Forty-one code-contract and regression tests passed before launch. First
generation identity reproduced 13/24 and nominal success. A separate
comparison confirmed all twenty-five complete nominal/development traces
match R93 identity exactly in qpos, qvel, strict flags and residual arrays.
Eight of twenty-four first-generation candidates preserved nominal standing;
the selected best still recovered 13/24. These are initial observations,
not proof of improvement or independent qualification. Main process 2106295
and wrapper 2106293 were observed; verify current identity before any process
operation rather than relying on these launch IDs.

The next successful process check found the same main and six workers
running, with no stopped T state. Five generations had completed, each
retaining best 13/24 and a nominal-successful search best. Generation five
had seven nominal-feasible candidates and four stagnant generations. This
is continuing negative development evidence, not a final result. The
previous permission-review timeout did not stop the training process.

Source is `diagnostics/train_getup_preload_r95.py`; launcher is
`scripts/launch_getup_preload_r95_20261001.sh`. Output is
`/data/shijinsheng/open_duck/outputs/getup_preload_r95_left_20261001`.
Before any further launch, check actual processes, log, search history,
checkpoint and final results. Do not act on stale PIDs or duplicate a live
job. If the best feasible count does not improve, distinguish the search
distribution from the selected feasible candidate and inspect actual failed
contact trajectories before selecting another method. Do not describe reward
growth or short development success as thirty-second independent recovery.
