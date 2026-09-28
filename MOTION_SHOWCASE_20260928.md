# Continuous forward / right turn / backward / stop recording

**Historical recording:** this page describes the original shared-filter
recording. The current recording script defaults to corrected per-skill
execution; see `MOTION_EXECUTION_FIX_20260928.md`. Use
`--legacy-shared-controls` only to reproduce the original behavior.

This is one continuous fixed-flat-plane MuJoCo simulation, seed 1600. There are
no resets between skills, no friction changes, and no hardware commands.

The 26-second H.264 MP4 runs at 25 fps and normal speed. The left panel follows
the robot; the right panel provides a fixed top view with the recorded path.

| Video time | Phase | Observed result |
|---|---|---|
| 0–3 s | Stand | Initial settling |
| 3–8 s | Forward | Net horizontal displacement 0.374 m |
| 8–9 s | Settle | Zero command before policy switch |
| 9–14 s | Right turn | Heading change −24.485 degrees |
| 14–16 s | Settle | Zero command before backward start |
| 16–23 s | Backward | Net horizontal displacement 0.551 m |
| 23–26 s | Stop | Last-second drift 0.0933 mm |

All phases completed without a fall. Backward minimum up-vector Z was 0.9765.
Stop is a 1-second smooth target crossfade followed by standing; its entire
3-second phase has nonzero braking displacement. Last-second drift does not
mean instantaneous stopping.

Forward and standing use the official `BEST_WALK_ONNX_2.onnx`. Right turning
uses the V5 yaw-error actor plus V8 extended-band negative residual. Backward
uses the R22 candidate's matching frozen R2 reference/residual decoder and
8-input heading/contact/pitch corrector. The recording applies 0.01-second
target filtering throughout, preserves all motor slew limits, and updates the
three action histories freshly before each physics segment. These details can
change performance relative to earlier isolated turn tests.

The forward command was +0.15 m/s, but this run's average axial speed was only
+0.0743 m/s. The backward command was −0.074 m/s, with observed average axial
speed −0.0786 m/s. This is an observation recording, not proof of precise speed
tracking, general robustness, or hardware readiness.

Files are in `media/showcase_20260928/`: MP4, trajectory NPZ, results JSON and
video metadata. Exact policy and script hashes are in the results JSON.

Reproduce from the resource-complete server checkout:

```bash
source /data/shijinsheng/open_duck/env_walk.sh
cd /data/shijinsheng/open_duck/projects/Open_Duck_Playground
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1 \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 \
.venv/bin/python -u -m diagnostics.record_motion_showcase \
  --output /data/shijinsheng/open_duck/outputs/showcase_unique_directory
```

Select an available EGL device on shared servers. Encoding requires
`imageio-ffmpeg`; it does not change the simulation dependencies.
