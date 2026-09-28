# Backward locomotion: native phase search, heading feedback and pitch balance

Simulation only. Production agent and hardware decoder are unchanged. R2 remains frozen and recoverable; SHA256 `05c1219a1403831deead152a6cff89b30a1d20fea3ebef285ca5c07279cfbdc8`.

## Why change methods

R4 shared hip-yaw/roll PPO correction did not resolve curving. This experiment uses **derivative-free cross-entropy parameter search directly in ordinary MuJoCo**, not additional millions of PPO updates. Separately controlled left/right hip yaw, roll and pitch depend on gait phase. Ankle pitch opposes each hip-pitch correction. This is a bounded joint-space correction, not full Cartesian foot-placement IK.

The decoder, reference, joint limits and 50 Hz motor slew limit are retained. Flat terrain and friction are unchanged throughout. Startup is the exact home pose with seeded initial qvel sampled uniformly within ±0.02, not arbitrary pose recovery. No observation noise or variable delay is introduced.

Native execution: infer target → limit joints and motor slew → advance 10 physics steps → advance reference phase. The observation uses the freshest three motor action histories. Reference dx −0.0925 with interpolation, backward command −0.074 m/s, R2 residual gain 0.12 rad and one-second reference startup ramp. This is a separate opt-in simulation contract.

## Search and export

`diagnostics/backward_phase_search.py` freezes R2. R5 trains 18 phase coefficients; R6/R7 freeze phase coefficients and train six sine-heading-error feedback gains. R8/R9 additionally freeze those 24 coefficients and train two body-pitch/finite-difference-pitch-rate feedback gains.

Each stage uses eight generations of 20 candidates, CEM random seed 87, elite fraction 1/4 and four CPU workers. Phase coefficients are bounded to ±2; heading/balance gains to ±3. A fall receives a large penalty; speed and progress penalties prevent solving the objective by standing still. Launcher: `scripts/launch_backward_phase_search.sh`. It refuses to overwrite existing experiments.

The pitch correction has a 0.10 rad deadband, a pitch scale of 0.2 rad, a pitch-rate scale of 0.8 rad/s and rate clipping at ±4 rad/s. Its output is bounded to ±0.06 rad common hip pitch with opposite ankle compensation. It reacts to orientation, not contact forces. Simulated heading/pitch inputs are **not evidence that the current real-robot runtime can supply identical signals**.

`diagnostics/export_phase_controller.py` exports a tiny ONNX corrector. Outputs are **14 joint deltas in radians**, to be combined with the frozen R2 reference-residual target before limits/slew. Do not feed these outputs to a legacy normalized 101-to-14 action decoder. Export checks 1,000 randomized NumPy/ONNX input pairs. Closed-loop ONNX validation is mandatory even after numerical parity passes.

## Completed isolation results

| Experiment | Training conditions | Independent result | Decision |
|---|---|---|---|
| R5 phase only | 10 s, seeds 0/1 | Early candidate: 10/10 survived 30 s, but yaw/lateral drift remained; final candidate later failed one 30 s training-control run | Not accepted |
| R6 phase + heading | 30 s, seeds 0/1/2 | Exported early candidate: 8/10 survived seeds 10–19 for 30 s; seeds 15/18 failed at 10.64/21.90 s | Not accepted |
| R6 orientation gate | Same phase/heading; gate fades correction as up-Z drops from .975 to .94 | Known seeds 10–19: 10/10 survived; new 20–29: 8/10; 60 s seeds 20–24: 3/5 | Local improvement, not general recovery |
| R7 70% phase amplitude + retrained heading | 30 s, seeds 0/1/2/15/18 | Final ONNX, 60 s seeds 20–29: 9/10 survived, 8/10 met combined straight-backward criteria | Improved but not solved |

R7 final seed 21 fell at **42.20 s**. Seed 28 survived but ended at −27.25° heading error. Other final R7 60 s runs had initial-heading-axis speed about −0.078 to −0.081 m/s. A 9/10 survival count does not establish a solved gait.

Matched zero-correction R2 control, same native clock and cold-start procedure, seeds 30–34 for 60 s: all survived, none met straight-backward criteria. Maximum unwrapped heading change was 376.38°, maximum lateral displacement 3.056 m; initial-axis velocity ranged −0.0195 to +0.0210 m/s. The robot continued stepping backward in its own frame but looped, losing useful backward progress.

Kinematic replays of failed R6 tests show progressive backward body pitch and eventual tipping. That motivates active pitch feedback; it does not prove toe collision or insufficient ground friction. Contact counts in pose-only replay are not measured dynamic contact forces. Both successful and failed gates are retained.

## Acceptance boundaries

For this fixed-plane simulation experiment, `diagnostics/summarize_phase_result.py` requires no fall, initial-axis speed −0.10 to −0.05 m/s, |final unwrapped heading| ≤15°, heading RMS ≤10°, |lateral displacement| ≤0.25 m and minimum up-Z ≥0.94. These are experiment gates, not a hardware safety standard. Require additional unseen-seed ONNX tests and longer durations; do not promote on training return alone.

R8 completed all eight generations (1,120 candidate rollouts, 1,561,409 actual control steps, not PPO updates). Its exported final corrector survived 9/10 unseen 60 s tests, but only 7/10 met combined criteria. Seed 38 fell at 5.44 s; two survivors had excessive lateral displacement. R8 is not promoted.

R9 completed all eight generations (640 candidate rollouts, 1,693,171 actual control steps). An early exported candidate met all criteria in 9/10 unseen 60 s tests; seed 31 fell at 28.96 s with progressively increasing backward pitch. **Final R9, seeds 30–49 for 60 s: 17/20 survived and only 9/20 met all criteria**, with falls in seeds 35/42/47. Training-side score improvement did not generalize. R9 is rejected. Do not confuse the early and final models or promote the final model because its search score is better.

R10 hypothesis: absolute-pitch feedback interferes with **normal phase-dependent gait lean**. A successful R7 seed-20 training-side trajectory shows pitch −0.2066 to −0.0775 rad after startup. A seven-coefficient, three-harmonic phase template fits that oscillation with RMS error 0.01784 rad and p95 absolute error 0.03797 rad. R10 subtracts the ramped nominal pitch and its finite-difference rate before forming balance features, with a 0.04 rad error deadband. Only the two balance gains are optimized; phase and heading coefficients remain frozen. Previously failed seeds 21/31 are explicitly included in training, so future acceptance must use new seeds, not call those independent validation.

R10 uses 60 s training rollouts for seeds 0/20/21/25/31. It is under evaluation; no solved-gait or hardware readiness claim is made.

Reproduction requires the resource-full runtime checkout and reference-motion assets already used for R2. The lean publication clone is not a replacement for those assets. Pitch-template preprocessing and its phase-period contract must travel with the ONNX model.

Full checkpoints and trajectories remain under `/data/shijinsheng/open_duck/training/` and `/data/shijinsheng/open_duck/outputs/`. GitHub receives source, exact controller contracts, small ONNX models, search logs and result JSONs; no caches, virtual environments or large training checkpoints.
