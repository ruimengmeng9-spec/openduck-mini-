# R74 outcome and prepared early timing probe R75 (2026-10-01)

## Completed R74

R74 stopped after 21 generations (twelve stale), retaining 14/24 training
successes. Forty fresh seeds 777000–777039 produced 14/40 candidate successes
and 13/40 baseline successes, with four rescues and three regressions. All
forty candidate physical audits were valid. The nominal candidate passed.
No stage gate pass, no reliable improvement claim, no hardware deployment.

Output: `/data/shijinsheng/open_duck/outputs/getup_wait_pose_r74_left_20260930`.
This completed run includes final checkpoint, history, contract, paired
training replays and all paired independent trajectories, including failures.
Keep these records. Do not restart or overwrite the run.

Training-only saved quaternion endpoint inspection shows all ten remaining
candidate training failures have root up-Z below 0.5 at the end of wait.
The earliest recovery phase itself is unchanged by R74. This motivates a
different timing intervention, not a claim that timing is proven causal.

## Prepared R75, NOT running

New source: `diagnostics/search_getup_prefix_timing_r75.py` and
`diagnostics/test_getup_prefix_timing_r75.py`. Uploaded to the simulation
project. Tests have **not** been verified: two remote commands timed out
during automatic approval, before confirmed execution. No R75 launch is
confirmed. Inspect processes/output and pass tests before any launch.

Use the frozen R73 zero-correction checkpoint (not R74's wait offsets), so
only early timing differs. Split the initial 308-control recovery stream
into four quarters. Duration multipliers are bounded to 0.7–1.3. Resample
the original target path within each quarter with endpoints preserved.
Unit factors must reproduce the original target stream bit-for-bit.
The original 50 Hz target-slew limiter still applies; never increase it.
Freeze later wait duration and targets, head pulse, terminal targets, IMU
gains, physics and qualification thresholds. Regenerate nominal IMU
references by physical simulation for each timing variant.

First run the training-only grid: identity, six global duration scales and
eight one-quarter scales, on the same 24 training falls plus mandatory
nominal. Prospective directory:
`/data/shijinsheng/open_duck/outputs/getup_prefix_timing_probe_r75_20261001`.
Do not assume it exists or that a result was generated.

Only after inspecting that probe, decide whether to run the optional bounded
R76 CEM mode (`--train`), in a new directory. Its seed is 176, maximum 32
generations, population 24, six workers, twelve-stale stopping criterion.
Automatic independent qualification reserves new seeds 778000–778039,
excluded from fitting. Preserve paired traces, complete actual-fall paths,
valid physical audits and at least 30 continuous strict-standing seconds
in a 35-second final hold. Restore happens only at each trial start, never
mid-trial. No exact spatial target equality claim is made for non-unit
scales: interpolation changes time sampling, not the intended target path.

This work is simulation-only, separate from walking control. Left-side
qualification remains incomplete; other fall orientations and robustness
gates remain pending. Real-robot sensor and supply faults are unresolved.
