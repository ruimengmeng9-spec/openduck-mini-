# R4: frozen backward gait and a small heading corrector

Simulation-only experiment. No change to production agent or hardware decoder.

## Hypothesis and boundaries

R2 survives backward motion but curves. R3 increased heading reward while updating the whole gait and regressed. The timing probe also showed that stale action history changes behavior. R4 freezes the exact R2 ONNX weights and trains only a small two-output actor. It modifies four hip joints, not knees, ankles or head. Ground and friction are unchanged.

The new opt-in decoder selects a target before advancing physics, advances the reference phase after physics, and exposes the most recent three motor actions. Actor observations are 107 channels: the existing 101, sine/cosine of relative heading error, projected base up-vector, and startup ramp. The critic has 218 channels. This new interface is not interchangeable with a legacy 101-to-14 actor.

The deterministic corrector is initialized at zero. Exploration starts small. R2 weights are JAX constants, absent from the optimized parameter tree. Correction is limited to ±0.05 rad hip yaw and ±0.025 rad hip roll, followed by joint-limit projection and the existing 5.24 rad/s target slew limit at 50 Hz.

**Heading features currently use simulated base orientation. No claim is made that this heading signal is available from the real robot's current IMU/runtime.**

## Pre-training checks

- R2 SHA256: `05c1219a1403831deead152a6cff89b30a1d20fea3ebef285ca5c07279cfbdc8`.
- Compiled reset and step passed, with 107 actor observations, 218 critic observations and two trainable actions. Fresh action-history equality checked.
- Zero correction: seeds 0–2 all completed 10 s. Speeds −0.08134, −0.08656, −0.08956 m/s; heading changes −23.29°, −31.22°, −26.46°.
- Single-seed authority probes (not proof of general improvement): yaw correction +0.5 produced −18.98°, −0.5 produced −23.79°; roll +0.5 produced −31.20°, −0.5 produced −16.77°. All completed 10 s. This demonstrates some steering influence while retaining backward motion.

## Training protocol

Launcher: `scripts/launch_backward_heading_steering_r4.sh`. New actor/critic hidden widths (64,64)/(128,128), seed 86, 1024 environments, 64 evaluation environments, requested 4,096,000 PPO steps, learning rate 3e-4. Desired backward command −0.074 m/s, continuous reference dx −0.0925, residual baseline gain 0.12 rad, startup ramp 1 s. Initial relative goal offsets sampled within ±0.15 rad. Noise and sampled delays disabled for this first isolation experiment.

Heading penalty 20; other gait/contact/progress settings retain the R2 isolation experiment. A larger training return does not constitute acceptance.

## Acceptance and reproducibility

Run `diagnostics.validate_heading_steering` with the matching model contract, fixed reference configuration, cold start and identical seeds. Compare with zero correction under this **same new timing contract**, not with a different R2 runtime. Check falls, initial-axis backward speed, unwrapped heading change, lateral displacement and minimum up-vector over 10 s and 30 s. Test held-out seeds and ±0.15 rad heading goals if the basic gate improves.

Training is in progress; no R4 success or deployment claim yet. Preserve R2 as the rollback baseline. Save the final small actor, its controller contract, evaluation JSON and updated conclusions here when available. Do not upload dependency caches or large checkpoint directories.
