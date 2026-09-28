# Per-skill execution contract correction

The first showcase recording unintentionally applied the backward target
filter and filtered-action history to forward and turn policies too. This
changed the execution contract of otherwise unchanged policy weights.

The corrected simulation recording now separates the contracts:

- Forward, turn, and ordinary standing: no additional target low-pass filter;
  history stores the policy's original normalized action. Freshly shift all
  three history frames; retain joint clamps and the 5.24 rad/s slew limit.
- Backward and backward-to-stop: retain R22's 0.01 s target filter and filtered
  pre-slew action history, the matching R2 decoder, and the 1 s stop crossfade.
- On entry into backward, initialize the target filter from the previously
  applied motor target. Do not reset the physical state, gait phase or histories.

No neural-network weights, physical friction, or actuator force limits changed.
This correction is in the simulation recording pipeline, not a new hardware
deployment or a replacement of the production language-agent controller.

## Recording comparison (seed 1600)

Both recordings use the same 26 s schedule, same models and +0.15 m/s forward
command. Corrected warm-up uses the ordinary standing contract as well.

| Metric | Original recording | Corrected recording |
|---|---:|---:|
| Forward average axial speed over 5 s | 0.07426 m/s | 0.08526 m/s |
| Forward last-2-s mean local speed | 0.08407 m/s | 0.09671 m/s |
| Right-turn heading change over 5 s | −24.48 degrees | −43.44 degrees |
| Backward average axial speed over 7 s | −0.07858 m/s | −0.08039 m/s |
| Stop last-1-s drift | 0.09326 mm | 0.09522 mm |
| Falls | None | None |

The corrected forward average is about 14.8% higher. Its last-2-s speed is close
to the previous ordinary MuJoCo result of 0.0974 m/s. This does not claim that
the +0.15 m/s command is accurately tracked or that JAX and native MuJoCo
produce identical speed. Braking still takes time and has nonzero displacement.

Six unit tests cover filtering, raw-action history, fresh history shifting,
backward filter initialization, physical clamps/slew, and invalid-target
rejection. A ten-seed paired continuous-sequence regression is archived as
`results/showcase_contract_v2_regression_20260928.json`.

All ten seeds completed both execution modes without a fall. Across those
paired runs, mean forward axial speed rose from 0.07440 to 0.08517 m/s (14.5%).
Corrected last-2-s local forward speeds ranged from 0.09477 to 0.09719 m/s.
Mean backward axial speed was −0.08376 m/s; maximum stop last-second drift was
0.1119 mm. This is a bounded fixed-plane regression, not a hardware safety claim.

The corrected MP4, exact trajectory and metadata are in
`media/showcase_20260928_v2/`. The original video is preserved separately.

## Reproduce

```bash
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1 \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 \
.venv/bin/python -u -m diagnostics.record_motion_showcase \
  --output /data/shijinsheng/open_duck/outputs/new_showcase_directory
```

Choose an available EGL device. `--legacy-shared-controls` reproduces the old
recording's shared filter only for comparison; it is not the default.
