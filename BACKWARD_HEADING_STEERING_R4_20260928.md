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

## Completed results: R4 rejected

Training completed **4,587,520 actual steps** (PPO rounds requested steps to complete batches). Evaluation return increased from 2145.34 to 2257.94. The corrector ONNX is 48 KB, SHA256 `353a76a34a85c7ab10c5975c3485d0ac1757fb836da27b4f53047c5d6cf3dae0`. The baseline R2 hash is unchanged. Twenty normalized-input ONNX/JAX parity checks had maximum error 3.73e-7; zero deterministic correction initialization also passed.

All five R4 native tests completed 10 s and all five completed 30 s without falling. This does **not** establish successful straight backward walking:

| Same cold-start 30 s protocol, seeds 0–4 | Zero correction | R4 learned correction |
|---|---:|---:|
| Completed / tested | 5 / 5 | 5 / 5 |
| Mean initial-heading-axis velocity, m/s | −0.03908 | −0.03626 |
| Mean unwrapped heading change | −128.12° | −132.27° |
| Mean lateral displacement | 1.621 m | 1.558 m |

R4 seed 1 rotated −186.76°; its initial-axis displacement even became slightly positive, despite continued body-frame backward stepping. R4 is **not promoted** and no production agent is changed. No hardware readiness claim.

Additional non-learned authority controls, seeds 0–2 for 30 s, were also rejected. Proportional sine-heading feedback gain 4 averaged −185.92°; saturated constant combined yaw/roll correction (+1,−1) averaged −183.87°. All survived but nearly lost useful initial-axis backward progress. Therefore increasing a constant/shared hip correction based on a favorable short test is not a valid fix. Phase-dependent contact and the frozen gait's feedback response must be accounted for.

### Matched-initialization cross-engine diagnostic

With native seed-0 home qpos/qvel copied into MJX, zero correction remained upright for 30 s and rotated −99.91°, close to native zero correction seed 0 (−98.16°). Initial observations differed only in accelerometer channels 3–5, maximum 4.21 m/s². This is a real initial sensor discrepancy to audit, **not evidence that it explains the long-run failure**: both engines still reproduce curving under the matched clock.

Training resets base/joint velocities to zero, while the native robustness gate perturbs qvel within ±0.02. Thus this first-stage training does not cover the entire gate distribution. That mismatch and restricted shared two-axis steering are hypotheses for follow-up, not proven sole causes.

### Next experiment boundary

Keep R2 frozen and preserved. Before another PPO run, test phase/contact-dependent, separately controllable left/right foot-placement or hip corrections with a small constrained search directly in native MuJoCo. Require repeatable steering authority and stability across seeds first, then distill the successful controller and add matching initial-velocity perturbations to the learning curriculum. Do not increase gain, invert a sign, or lengthen PPO blindly.

Saved: `models/backward_heading_steering_r4/final.onnx`, controller contract and training log; native results, rejected authority probes and MJX diagnostic under `results/heading_steering_r4_*`. Full local trajectories and checkpoints remain under `/data/shijinsheng/open_duck/outputs/` and `/data/shijinsheng/open_duck/training/backward_heading_steering_r4/`; caches/checkpoint directories are not uploaded.
