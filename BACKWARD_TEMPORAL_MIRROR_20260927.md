# Backward temporal-mirror teacher probe — 2026-09-27

## Question and acceptance gate

Negative turning benefited from a left/right geometric reflection of a successful positive-turn teacher. This experiment asked whether a forward-walking policy could similarly provide a backward teacher. A teacher would need to complete at least 10 seconds on multiple perturbed seeds while producing meaningful reverse velocity at a requested -0.074 m/s. Merely moving backward before falling is not an acceptable distillation label.

The baseline actor was `official_seed_turn_balance_v5_yaw_error/final.onnx`. Tests used native MuJoCo, the unchanged flat-terrain XML and motor slew limit, a 3-second standing warm-up, and independent ±0.02 qvel perturbations. The policy-only transformations were: flip the forward command sign seen by the policy; reverse the sine component of the gait phase; invert selected sagittal joint action offsets; and blend the ankle inversion strength. This is **not** a true physical time reversal.

## Results

| Candidate | Duration / seeds | Survived | Mean signed forward speed | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Original v5 with negative command | 5 s / 3 | 3/3 | ~0.00002 m/s | Stands nearly still |
| Positive command fed to v5, reversed phase | 5 s / 3 | 3/3 | +0.00172 m/s | No reverse gait |
| Positive command, full hip/knee/ankle action inversion | 5 s / 3 | 0/3 | -0.318 m/s until fall | Falls in ~0.56 s |
| Positive command, ankle inversion strength 0.35 | 10 s / 3 | 3/3 | -0.00404 m/s | Stable, negligible reverse |
| Positive command, ankle inversion strength 0.39 | 10 s / 3 | 3/3 | -0.00382 m/s | Stable, negligible reverse |
| Positive command, ankle inversion strength 0.40 | 10 s / 3 | 0/3 | -0.08874 m/s until fall | Falls in 2.20–2.24 s |
| Reversed phase plus ankle strength 0.39 | 10 s / 3 | 0/3 | -0.12649 m/s until fall | Falls around 1.5 s |
| Ankle strength 0.40 with pitch guard to 0.30 | 10 s / 3 | 2/3 | -0.01190 m/s across unequal durations | Not robust; mean includes early failure |

The seven single-seed hip/knee/ankle subset probes also found no teacher meeting both safety and speed requirements. Knee-only inversion survived but moved **forward** at +0.02681 m/s; ankle-containing full inversions moved backward briefly but fell.

The sharp change between ankle strengths 0.39 and 0.40 suggests a closed-loop stability boundary, not a usable reverse gait. These runs do not establish a mathematical impossibility for backward mirror distillation; they reject the **simple command/phase/action transformations tested here**. Pitch guarding in this limited sweep did not repair the boundary.

## Decision

No backward mirror teacher passed the gate, so **no new student weights were trained or promoted**. Distilling the fast but falling trajectories would teach the known failure. The existing v5 forward/turn baseline and previous backward models remain unchanged. The next training route should first construct a dynamically stable backward teacher—e.g. a contact/foot-placement-aware reference and balance objective—then distill only verified long, multi-seed trajectories and retest the student in closed loop. Do not deploy the diagnostic adapter to hardware.

## Reproducibility

Script: `diagnostics/test_backward_temporal_mirror.py`. Adapter unit tests: `diagnostics/test_backward_temporal_mirror_policy.py` (3/3 pass). Representative raw summaries are in `results/backward_*_20260927.json`; all complete run logs remain under `/data/shijinsheng/open_duck/outputs/` on the server.
