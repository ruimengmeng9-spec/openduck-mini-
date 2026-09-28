#!/usr/bin/env bash
# Simulation only. Frozen R2 + small native-controller search, not PPO training.
set -euo pipefail
source /data/shijinsheng/open_duck/env_walk.sh
ROOT=/data/shijinsheng/open_duck
cd "$ROOT/projects/Open_Duck_Playground"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
export REFERENCE_DX=-0.0925 REFERENCE_DX_INTERPOLATION=1
STAGE="${1:?choose r5 through r12}"
ARGS=()
case "$STAGE" in
  r5)
    OUT="$ROOT/training/backward_phase_r5"
    ARGS=(--duration 10 --seeds 0,1)
    ;;
  r6)
    OUT="$ROOT/training/backward_phase_feedback_r6"
    ARGS=(--duration 30 --seeds 0,1,2 --feedback-base "$ROOT/training/backward_phase_r5/best.json")
    ;;
  r7)
    OUT="$ROOT/training/backward_phase_feedback_r7"
    mkdir -p "$OUT"
    test ! -e "$OUT/phase_base.json"
    .venv/bin/python -m diagnostics.scale_phase_controller \
      --input "$ROOT/training/backward_phase_r5/best.json" \
      --output "$OUT/phase_base.json" --scale .7
    ARGS=(--duration 30 --seeds 0,1,2,15,18 --feedback-base "$OUT/phase_base.json" --resume "$ROOT/training/backward_phase_feedback_r6/best.json")
    ;;
  r8)
    OUT="$ROOT/training/backward_phase_balance_r8"
    ARGS=(--duration 30 --seeds 0,1,2,15,18,20,25 --balance-base "$ROOT/training/backward_phase_feedback_r7/best.json")
    ;;
  r9)
    OUT="$ROOT/training/backward_phase_balance_r9_long"
    ARGS=(--duration 60 --seeds 0,20,21,25 --balance-base "$ROOT/training/backward_phase_feedback_r7/best.json")
    ;;
  r10)
    OUT="$ROOT/training/backward_phase_template_r10"
    mkdir -p "$OUT"
    test ! -e "$OUT/template_base.json"
    .venv/bin/python -m diagnostics.fit_pitch_template \
      --controller "$ROOT/training/backward_phase_feedback_r7/best.json" \
      --trajectory "$ROOT/outputs/phase_feedback_r7_final_heldout_60s/seed_20.npz" \
      --output "$OUT/template_base.json"
    ARGS=(--duration 60 --seeds 0,20,21,25,31 --balance-base "$OUT/template_base.json")
    ;;
  r11)
    OUT="$ROOT/training/backward_contact_balance_r11"
    ARGS=(--duration 60 --seeds 0,21,31,38,57 --balance-base "$ROOT/training/backward_phase_template_r10/snapshot_probe.json" --resume "$ROOT/training/backward_phase_template_r10/snapshot_probe.json" --ankle-balance)
    ;;
  r12)
    OUT="$ROOT/training/backward_contact_heading_r12"
    ARGS=(--duration 60 --seeds 0,21,31,38,57 --feedback-base "$ROOT/training/backward_contact_balance_r11/snapshot_probe.json" --resume "$ROOT/training/backward_contact_balance_r11/snapshot_probe.json" --initial-std .25)
    ;;
  *) echo "Unknown stage" >&2; exit 2 ;;
esac
if test -e "$OUT/best.json" || test -e "$OUT/search.json"; then
  echo "Existing experiment preserved: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
.venv/bin/python -m unittest diagnostics.test_backward_phase_search -v
.venv/bin/python -u -m diagnostics.backward_phase_search \
  --output "$OUT" --generations 8 --population 20 --workers 4 \
  "${ARGS[@]}" | tee "$OUT/train.log"
.venv/bin/python -m diagnostics.export_phase_controller \
  --controller "$OUT/best.json" --output "$OUT/export_final"
