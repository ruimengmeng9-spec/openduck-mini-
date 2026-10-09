# R175 recurrent readout terminal and R176 read-only audit — 2026-10-09

R175 has naturally exited. Its selected 14×16 readout is all zero. Candidate and frozen R157 baseline both achieve **18/24** disturbed full-fall recoveries, retain the standard recovery and have zero physical-invalid episodes. Development gate is false. No expanded development or unseen qualification was run. This is not full-task completion or hardware readiness.

## Commands, budget and evidence

Project: `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`; environment: existing `.venv`, OMP/OPENBLAS/MKL threads 1, CUDA_VISIBLE_DEVICES empty, JAX_PLATFORMS cpu.

`/data/shijinsheng/open_duck/projects/Open_Duck_Playground/.venv/bin/python -u -m diagnostics.train_getup_recurrent_readout_r175`

Seed 275, two generations, eight proposals per generation, six CPU workers. Fixed sensor/recurrent matrices and the 224-dimensional readout contract are unchanged from `GETUP_RECURRENT_READOUT_R175_20261009.md`. CEM bounds ±.02, initial standard deviation .0005, elite 2, .6/.4 update, standard deviation .0001–.002; zero and best retained, repeated proposals cached. There are 13 actual programs (12 nonzero), 325 full-path training attempts, 50 final paired development attempts and six formal startup consistency replays: 381 formal dynamic attempts. Six previously saved independent smoke attempts bring the total to 387, not 387 independent starting states. Physical-invalid attempts stop early under the original audit; not every attempt runs all 2279 controls.

Outputs: `/data/shijinsheng/open_duck/outputs/getup_recurrent_readout_r175_left_20261009`, corresponding `.log`, both closed generations, all candidates/failures, parameters/distributions/history/RNG, `training_closed.json`, `results.json`, frozen models/feature matrices and executed sources. Final 25 candidate/baseline initial hashes, every saved field and original peaks match exactly; baseline original fields reproduce R157 exactly. Zero additional readout does not suppress the internal hidden state on disturbed episodes, but leaves original actions unchanged.

## R176 audit

`.../.venv/bin/python -u -m diagnostics.audit_getup_recurrent_terminal_r176 --smoke`

`.../.venv/bin/python -u -m diagnostics.audit_getup_recurrent_terminal_r176`

Independent read-only smoke audits six existing startup trajectories plus 25 terminal pairs; after natural completion, formal auditing covers all 13 programs / 325 training attempts plus 25 terminal pairs. Feedback uses a field whitelist. Actual scalar causal error changes, phi, recursive hidden state, readout extra and frozen right-hip feedback reproduce bitwise. Control-zero, standard and home added signals are zero; hidden state is bounded. Double saved original/adjusted pre-slew targets independently yield the saved direct difference exactly. Whole-field parity reads recorded root arrays only in a separate comparison, never as control inputs. No environment/MjData/forward/integration, force inference, label or physical-validity relabeling occurs. Tracked execution/frozen sources and training evidence hashes remain unchanged. The three overlapping PROBE signal arrays also compare bitwise during archiving.

Limit: R176 does not independently reconstruct the standalone joint/slew planning decision; actual pre-integration planned targets are checked against recorded applied targets through terminal parity. Do not call this a dynamic counterfactual, future-action prediction or recovery proof.

Six of the 12 nonzero programs are physically valid across all training starts. Highest nonzero and highest all-valid are the same generation 2 / candidate 03, complete 10/24, invalid 0. It rescues 769002, 769004, 773004, 773015 but regresses 769000, 769005, 769006, 769007, 773000, 773002, 773003, 773010, 773011, 773012, 773013, 773014. These full-path labels show no overall improvement; they do not establish that all causal feedback or memory is ineffective. R176 saves this program's 25 signal arrays and three PROBE arrays, not 50 representative arrays.

## Hashes and immutable publication

Executed R175 source SHA256: `3e7115754a7cfccfa3eb3f015eb46e26de7d02b1df6367affe0d480afe2e143b`.

Fixed recurrence: `2cba4043d2f97fbf17ebd64ad75aad921b6e9a7f96332360a921d356bb178891`.

R157 frozen selector: `2aa2633dc048d8e9c8b262d088a7dbdab25897f7fd15e94119e69af72c7e7737`.

R122 frozen snapshot: `8fc66fda837ba7a2ef46fda66fad1eabcc986b9fd6e63424902290b9736fb3c9`.

Physical-array fingerprint: `4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae`.

Unique archive script `archive_getup_recurrent_terminal_r175_r176.py` starts from clean explicit base `288dc8ee1b9b9af880c613f5dd4688d4ea515cdf`. It preserves R175 full terminal and R176 formal/smoke terminal snapshots, logs and artifact manifests, without replacing the prior startup snapshot, source or failure evidence. Direct server publication uses only the repository-scoped deployment key and verified GitHub host, normal fast-forward push followed by separate read-only remote verification. No credential enters evidence.

## Remaining gates

Unified best remains R157 18/24, standard success, invalid 0. Do not extend R175's same readout budget or rerun the fixed PROBE. A different finite hypothesis must pass causal/boundary/actual scalar nominal-zero/initialization/full-original-parity regressions and independent full smoke before training. Original 14-joint total ±.18rad, joint/torque/slew/home, mesh/flags/physics/reward and acceptance remain unchanged. No extra settling or mid-episode root reset.

Original development requires ≥22/24, better than actual 18/24 baseline, standard retained and all physically valid; then frozen expanded 316 development ≥36/40 and better than fixedzero/R102 with all valid; only then unseen qualification. 3200000–3200039 remain unread/unexecuted; 316 and 318 are developed. Unseen paired candidate/zero initialization hashes, two disjoint 20-case groups each ≥18/20, entry ≤12s, strict continuous standing 30s and original 500Hz audits, followed by larger perturbations/delays and right/prone/supine validation, remain mandatory. Simulator-only authorization; no robot connection, enable or deployment. Continue ACTIVE, not complete.
