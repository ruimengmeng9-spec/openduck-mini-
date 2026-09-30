# Get-up feedback and whole-path verification R63–R67 (2026-09-30)

## Scope and selection

All experiments are simulation-only. No hardware file, deployed walking actor,
collision parameter, torque limit, target slew limit, or strict-standing gate
was changed. The full get-up task remains incomplete and hardware readiness
remains false. R59 remains the full-path comparison baseline; R64 is a useful
**transition-only** feedback candidate, not a replacement recovery controller.

## R63: phase-specific absolute IMU feedback (rejected)

Training reached 16/16. On forty independent transition perturbations it
decreased success from 36/40 to 31/40, and the canonical actual-fall replay
lost its strict standing tail (0 seconds). Phase-specific gains alone did not
prevent overfitting or nominal regression.

## R64: feedback relative to a successful reference trajectory

The residual uses deviations in body-frame up-vector X/Y and pitch/roll gyro
from the nominal R59 trajectory, sampled **before** each 50 Hz command.
Four phases have separate gains, and corrections are capped at ±0.18 rad and
passed through the unchanged actuator clipping and slew mechanism. The
canonical nominal case is mandatory during training.

- Zero-feedback evaluation matched the R59 evaluator before training.
- Training: 24/24 in six generations.
- Independent short-hold transition test: 38/40 versus R59's 34/40.
- Canonical replay: 1.68 seconds continuous strict standing at the short gate.
- Five control-contract unit tests passed: zero-error behavior, bilateral
  mapping/cap, feature order, pre-control sampling, and long-reference prefix.

## R65: paired long-hold verification

Require at least 30 seconds continuous strict standing at the end of a
35-second home hold, as well as complete execution and physical validity.
The same seeds are run with R59 and with R64 feedback.

| Group | R59 | Final R64 |
|---|---:|---:|
| R64 independent seeds (40) | 34/40 | 38/40 |
| Additional independent seeds (40) | 33/40 | 34/40 |
| Combined (80) | 67/80 | 72/80 |

Across the eighty cases there were twelve rescues and seven regressions.
The additional group alone improved only one case, so these results must not
be interpreted as universal robustness. An earlier frozen generation-two
checkpoint scored 67/80, equal to R59, and is archived separately. Its result
must not be attributed to the final generation-six checkpoint.

## R66: start from the actual perturbed left-side fall

The exact nominal 50 Hz prefix commands were recorded and frozen. Replaying
those commands reproduced the canonical captured-state hash exactly. Each new
case was instead initialized from a perturbed fallen pose, physically settled,
and then received that fixed stream. There was **no hindsight selection of
the best transition on held-out trajectories**, and no reset to the nominal
mid-path state. Each terminal branch began from its own physically reached
state. Final qualification again required thirty continuous strict seconds.

- Perturbations before settling: tilt ±0.05 rad, joint position ±0.02 rad,
  generalized velocity ±0.01.
- Twenty independent full fallen-start cases: R59 9/20; final R64 7/20.
- No rescues and two regressions; full task not passed.
- Eleven unsuccessful prefix states lacked two-foot support before the
  terminal continuation. Several ended approximately side-on, with up-vector
  Z near 0 or −0.265 and both foot loads at zero.
- A separate four-case prefix replay audit completed all 1889 prefix controls
  without violating existing physical audit limits. Therefore at least these
  failures were not caused by rejecting the path on the physical thresholds;
  the earlier fixed prefix reached the wrong recovery state.

This isolates a coverage gap: learning a robust continuation from a narrowly
perturbed nominal transient does not make the earlier fallen-to-transient
movement robust.

## R67: whole-path reference-error feedback training

New experiment: `diagnostics/search_getup_fullfall_tracking_r67.py`.

- Learns four prefix-phase IMU feedback gain sets across the full frozen
  command stream, while freezing the R64 terminal gains.
- Eight independently initialized, physically settled left-side training
  falls plus a mandatory nominal case; no intermediate root-state reset.
- CEM budget: 24 generations, population 24, six CPU workers.
- Independent seeds are fixed in advance and excluded from optimization.
  After training, twenty fresh full fallen-start cases and the nominal case
  are tested with a 35-second hold and thirty-second strict-standing gate.
- Both zero-prefix-feedback and learned-prefix-feedback branches are saved.
  Gains never bypass the original physical limits.

The first two R67 generations reached 4/8 training successes and preserved
the nominal success; this was not yet an improvement sufficient for selection.
At publication, R67 was running on the server. Its contract/checkpoint are
progress artifacts, **not evidence of successful full get-up**. Other fall
orientations, realistic sensing/delay robustness, hardware-model matching,
and real-robot deployment remain separate unpassed gates.

## Reproduction

Use the existing Open Duck Playground virtual environment. Inputs are the
audited decomposed R4 scene, the original `BEST_WALK_ONNX_2.onnx`, the archived
R27 left-side `best_reference.npz`, and the archived final R64 checkpoint.
The command-line scripts expose `--root`, `--reference`, `--checkpoint` or
`--terminal-checkpoint`, and fresh `--output` directories. Results and source
hashes are stored with each experiment; do not overwrite prior outputs.
