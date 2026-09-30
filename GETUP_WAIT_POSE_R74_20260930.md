# R73 negative result and wait-pose hypothesis R74 (2026-09-30)

## Completed R73, not a trained improvement

The pre-wait last-two-second leg-knot search ended after 16 generations,
including 15 stale generations. Training stayed at 13/24. The retained knot
corrections were exactly zero (all 32 parameters), so the candidate is the
unchanged baseline, not a new learned recovery controller.

Fresh complete fallen-start seeds 776000–776039 qualified 10/40 for both
candidate and baseline. The nominal case passed. Each qualification requires
valid original physical audits, complete execution and 30 continuous strict
standing seconds in a 35-second final hold. No improvement and no stage gate
pass. Failed trajectories, checkpoint, history and reports remain on server.

## Training-only failure localization

R71 saved post-control trajectories cover nominal plus 24 training falls.
Ten of eleven failed training cases have root up-Z below 0.5 at the end of
the home-wait phase; the remaining failure (773005) occurs later. For example,
769001 changes from root up-Z about 0.303 at initial recovery completion to
about -0.252 after the wait; the nominal values are about 0.342 and 0.487.

`audit_getup_wait_tilt_r74.py` uses only these existing training trajectories.
It derives root vertical projection from normalized floating-base
quaternions and retains per-phase endpoints and source/trace hashes. It does
not reconstruct accurate contact forces from saved poses, use independent
test failures for fitting, or imply all failures have the same cause.

## New bounded experiment, not yet a validated skill

Hypothesis: maintaining the unmodified home targets during the transient
wait lets some perturbed cases fall into the wrong contact basin. Unlike R73,
R74 changes the wait itself, not the final two seconds before it. Learn ten
joint target offsets (eight leg joints, neck pitch, head pitch), bounded to
±0.25 rad, smoothly ramped in and out over 0.2 seconds. The last wait command,
all pre/post-wait targets, phase durations, IMU feedback, physics, actuator
force/position/slew constraints and final qualification thresholds stay fixed.

Train from physically settled complete left-side falls using the existing
24 training seeds. Nominal recovery remains mandatory. CEM seed 174, at most
32 generations, population 32, six CPU workers, twelve stale generations.
Fourteen control-contract tests passed before launch; tests are not evidence
of recovery success. Record paired training replays for failure diagnosis.

Automatic post-training qualification uses **new seeds 777000–777039**,
excluded from search. Both candidate and baseline receive the same fallen
start snapshot at the start of their separate trials. No root-state reset or
teleportation occurs within a trial. Keep all paired failure trajectories.
Qualification again requires 30 continuous strict standing seconds and valid
physical audits, not just a favorable short training return.

## Continuation

Check actual process, checkpoint, log and final results in
`/data/shijinsheng/open_duck/outputs/getup_wait_pose_r74_left_20260930`
and its `.log` before starting any new experiment. The publication may contain
only the immutable launch contract while this run is active; do not treat
a running checkpoint as final. R67/R70/R73 are completed and must not restart.

The left-side skill is still unqualified. Right-side, prone, supine, larger
independent sets and sensor/actuation delays remain pending. This experiment
does not touch the deployed walking controller or robot. Real-hardware IMU
and supply faults are a separate unresolved gate, not fixed by this training.
