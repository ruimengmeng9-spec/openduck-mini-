# Backward reference-residual R2 — 2026-09-28

Status: stable backward motion with substantial heading drift in tested flat
simulation. Not approved for hardware or the ordinary agent controller.

## Why this experiment

The latest main (`c7cf758`) added continuous reference-dx interpolation in the
JAX reference path. The native NumPy reference did not implement that opt-in
path. Previous native replay of `REFERENCE_DX=-0.0925` therefore cannot be used
as evidence that the continuous -0.0925 reference itself is stable.

`diagnostics.reference_residual_policy.ContinuousDxReference` now provides an
opt-in native counterpart without changing the production NumPy reference.
Across five dx values and 27 phase samples, maximum joint-target discrepancy
with JAX was 0.000173 rad. Maximum discrepancy across all 40 channels was
0.013267 (polynomial derivative channels included); those derivative channels
are not used by this decoder. This is numerical agreement of joint targets,
not bitwise equality of every channel.

The corrected reference reachability audit found:

- At legacy action scale 0.25, normalized reference actions span approximately
  -1.815 to +2.354. About 21.96% of action entries exceed [-1, 1], and 70.37% of
  phases contain an unreachable leg target under that actor interface.
- Knee reference peaks are approximately 1.784 and 1.968 rad, whereas the
  model joint limits are 1.571 rad. Each knee exceeds its physical limit in
  14.81% of sampled phases.
- Direct reference replay survived only 2/5 ten-second tests. Actor-clipped
  reference replay survived 5/5 but had mean initial-heading-axis speed
  approximately -0.001593 m/s. This audit used home-target settling and is not
  the same initialization protocol as the sustained policy gate.

These observations identify representation and physical-limit mismatches.
They do not prove that changing the representation alone solves backward
walking, nor that further reward optimization can never help.

## Controller contract

The R2 actor emits a small state-feedback residual rather than a legacy motor
action. The decoder ramps from home to the continuous gait reference over one
second, projects that reference into physical joint limits, adds a residual
of at most 0.12 rad, then projects into joint limits again. Projection before
adding the residual preserves correction authority when the raw reference
lies outside a knee limit. The existing 50 Hz motor-target velocity limiter
(5.24 rad/s) remains enabled.

The exported actor requires the matching reference decoder. It must **not** be
loaded into the ordinary `home + 0.25 * actor` agent/hardware controller.
`controller_contract.json` records the model hash and decoder parameters.

## Training protocol

- From scratch; no v30 checkpoint restoration.
- 1,024 parallel environments, seed 84, learning rate 1e-4.
- 8,192,000 requested steps, discount 0.995, entropy cost 0.002.
- Backward command -0.074 m/s; continuous reference dx -0.0925.
- No sensor noise or action/IMU delay in this first-stage feasibility test.
- Residual gain 0.12 rad; reference ramp 1 s.
- Imitation 25 / error 5, rear-support 30, single-support 20,
  rear-swing 15, termination cost 10,000, heading error 4,
  normalized progress 80 / shortfall 80.

Reproduce using `scripts/launch_backward_reference_residual_r2.sh` after loading
the project's environment setup. The launcher saves its complete log.

## Zero-residual control

Under the sustained gate (official-policy warm-up 3 s, phase reset, five seeds,
initial velocity perturbation +/-0.02, ten seconds, native MuJoCo), the matching
decoder with a zero actor survived 5/5 but moved at only -0.000332 m/s and turned
54.5–64.9 degrees. Stable standing/turning is therefore not a successful backward
baseline. R2 must improve both backward displacement and heading stability.

## Validation and decision

R2 completed 8,519,680 actual steps (PPO batches rounded above the requested
budget). Evaluation return rose from -425.38 to 1,341.79. This is a learning
indicator, not a hardware acceptance criterion.

Native MuJoCo with the matching decoder, physical joint clamp, 50 Hz slew
limiter, phase reset, 3 s official-policy warm-up and initial qvel perturbation
gave the following results. Speeds/displacements are projected onto the
**initial heading**, not the continuously rotating body frame.

| Test | Survived | Mean initial-axis speed | Mean absolute lateral displacement |
|---|---:|---:|---:|
| R2, seeds 0–4, 10 s | 5/5 | -0.083801 m/s | 0.225981 m |
| R2, seeds 0–4, 30 s | 5/5 | -0.038043 m/s | 1.453369 m |
| R2, held-out seeds 10–14, 30 s | 5/5 | -0.028543 m/s | 1.609607 m |
| R2, no warm-up, seeds 0–4, 10 s | 5/5 | -0.080274 m/s | 0.277142 m |
| Zero residual, seeds 0–4, 10 s | 5/5 | -0.000332 m/s | 0.013021 m |

The learned residual clearly produces backward motion instead of the
zero-residual control's nearly stationary turning. Nevertheless ten-second
heading changes range from -15.743 to -57.063 degrees. Longer trajectories
curve significantly; some net displacements even point forward along the
initial axis. Thirty-second endpoint heading values are wrapped into
[-180, 180] and must not be interpreted as total accumulated yaw.

Keep existing agent policies unchanged. R2 is not accepted as straight
backward walking and still lacks stop/restart and real-machine validation.

## R3 continuation

`scripts/launch_backward_reference_residual_r3.sh` resumes the R2 final
checkpoint with the same decoder, reference, speed target and other reward
weights. Heading-error cost increases from 4 to 20; the learning rate is
reduced to 5e-5 and seed is 85. This is continuation training, not a strictly
matched one-factor causal experiment (optimizer settings and seed also change).
The actor still observes the existing 101-dimensional sensor/history state;
absolute heading is not added to its input. R3 must be evaluated independently
before any claim of reduced drift. Its requested additional budget is
8,192,000 steps. Results pending.

Decoder unit tests and lateral-mirror unit tests pass (7 tests). Joint-reference
parity and JIT environment smoke tests also passed. The validator now rejects
a reference-residual actor loaded with legacy decoding or mismatched model
hash/decoder parameters when its controller contract is present.

Raw audit and control results are in `results/`; checkpoints remain on the
server outside Git. No hardware was actuated during this experiment.
