# R177 foot task terminal and R178 canonical kinematic audit

R177 completed its two-generation budget, full development and natural exit on2026-10-09. Selected left/right task amplitudes are `[0,0]`. Candidate and frozen R157 baseline both achieve **18/24**, standard retained and physical-invalid0. Original development gate is false. No improvement, expanded development or unseen qualification is claimed. Best proven unified controller remains R15718/24; continuation remains ACTIVE, full task and hardware readiness false.

## Complete path training and immutable inputs

Project `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`, existing `.venv`; OMP/OPENBLAS/MKL each1, CUDA_VISIBLE_DEVICES empty, JAX_PLATFORMS cpu.

`/data/shijinsheng/open_duck/projects/Open_Duck_Playground/.venv/bin/python -u -m diagnostics.train_getup_foot_task_r177`

Seed277, two generations, eight proposals/generation, sixCPU, two nonnegative task amplitudes0–.05; initialmean0/std.001, elite2, .6/.4 update, std bounds .0001–.005. Fixed task point, damping.01m, sign and timing were not searched. Clipped/cached duplicate proposals leave **10 actual programs, nine nonzero**, 250 original full-path training attempts, 50 final development and six formal startup consistency attempts:306formal attempts. Six independent smoke attempts already saved bring total312, not312independent initial states. Original maximum2279controls45.58s, entry≤12s, strict continuous30s and500Hzphysical audits apply; physical-invalid attempts stop early, so not all250walk the complete45.58s.

The new kernel projects causally anchored current body-relative foot-point error through each leg's3×5Jacobian and fixed damped least squares, then bounded target correction. Only actual current/causal-initial leg positions and same-phase nominal positions enter the new kernel. A private canonical-root MjData holds head joints at home; kinematics/composite positions/Jacobians only, no actualroot input, environment construction, forward/dynamics, contact/distance queries or integration. Task points are foot body origins in trunk frame, not floor height or predicted contact safety. R157 initialselector, R122snapshot nodes/feedback and originalIMUfeedback stay frozen. Original14joint total±.18rad, joint/torque/slew/home, mesh/flags, control/physics/reward/acceptance unchanged; no settling or mid-episode root reset.

The twelve corrected regressions, independent six full smoke attempts and formal six bitwise startup consistency checks are preserved. First pure mathematical head-dependence assertion failure and original three sources/log remain separately preserved: internal COM roundoff changed a leg Jacobian by2.77555756e-17; fixing private head positions removed it without relaxing limits or comparisons. Details remain in `GETUP_FOOT_TASK_R177_20261009.md`; do not overwrite that initial failure or repeat the fixedPROBE.

Output `/data/shijinsheng/open_duck/outputs/getup_foot_task_r177_left_20261009` contains all actual programs/failures, both distributions/checkpoints/history/learnerRNG, frozen model/source hashes, `training_closed.json`, `results.json` and corresponding natural-exit `.log`.

## R178 independent and formal read-only verification

`.../.venv/bin/python -u -m diagnostics.audit_getup_foot_task_terminal_r178 --smoke`

`.../.venv/bin/python -u -m diagnostics.audit_getup_foot_task_terminal_r178`

Independent audit smoke reconstructs six existing startup trajectories and25terminal pairs; it naturally exits before formal audit. Formal audit reconstructs all10programs/250attempts and25terminal pairs. White-listed actual sensors reproduce canonical foot error,3×5Jacobian,damped direction,extra and frozen right-hipfeedback **bitwise**, including control0/standard/home-zero and no head correction. It checks double saved original/adjusted pre-slew target differences exactly. Separate whole-field candidate/baseline comparison includes recorded root arrays only as equality evidence, never scalar or controller inputs. Every saved field/hash/original peak matches; baseline original fields reproduce R157. Actual pre-integration planned targets equal recorded applied targets. Tracked execution/frozen/source/training hashes remain unchanged. All original success/physical labels and500Hzpeaks remain untouched.

This audit creates a private canonical kinematic MjData; it does **not** claim no MjData construction. It creates no dynamic environment, calls no mj_forward/step/contact queries, performs no replay, force inference or relabeling. It does not independently recompute standalone joint/slew planning decisions. It saves25rank-selected representative signal arrays and threePROBE arrays. During terminal archiving the three overlapping PROBE arrays compare bitwise against independent audit smoke. The `best_nonzero` field uses the original standard/all-valid/success/return priority, not raw success-count ranking.

## Rescue and regression results

Only **one of nine nonzero programs** is physically valid across all training starts. It is generation2/candidate03, amplitudes `[.0002092158492387041,0]`, complete9/24. It rescues769002/769004/773004 but regresses769001/769005/769006/769007/773000/773002/773003/773009/773011/773012/773013/773014. Its new-target peak across25episodes is at most.0000472574755288086rad. The coordinate task has not solved the overall sensitivity/retention problem; this does not establish that all model-based feedback is ineffective.

Raw highest nonzero success count is generation2/candidate07, `[0,.0007044061843962011]`, complete11/24 with **two physically invalid episodes**. Rescue769002/769004/773001 accompanies ten regressions769001/769006/769007/773000/773002/773003/773007/773010/773012/773014. Invalid773002 and773015 each save304controls; original self peaks.048627708346635254m/.048630999699830826m, extra peaks.000058140550271703856rad/.00004480882037208348rad, total correction maxima.09437380539089477rad/.07436699640421057rad. Other original limits do not exceed. These endpoints/counts are not first500Hzcrossing times. Small corrections and being below±.18cap do not imply contact safety or justify relabeling.

Do not add the same two foot-position amplitudes, damping/task-point/sign/time-window search budget or rerunPROBE. No case-wise oracle selection, exact context lookup or success-union score is authorized. A future method must differ structurally, retain causal inputs/original physical limits, pass actual scalar/nominalzero/initialization/full-original parity regressions and independent full smoke, and use a finite declared budget.

## Publication dependencies and remaining gates

Unique `archive_getup_foot_task_terminal_r177_r178.py` is prepared but must not execute while the prior R175/R177 publication continuation is active. It requires its independently verified result file and exact newly published startup HEAD as base. Only then append immutable R177/R178 formal/smoke terminal snapshots, all closed generations/failures/RNG/logs/hashes and this new record. No old startup/terminal/source/failure/doc/bundle is overwritten. Server direct normal fast-forward uses only the repo deployment key and strict GitHub host verification; remote success requires waiting and independent read-only verification. No credential export.

At the time of preparation, R175terminal commit`dcf18833724b8ddbd52194d81b42c328a52e86fe` has not been independently confirmed published. Original helper reached its900s upload deadline while child transport continued; the finite continuation waits for that transport, verifies remote, then makes at most one3600s upload-only retry if needed, then publishes R177closed startup. That is not a training failure, permission to kill processes, or completed GitHub upload. Actual `tmp/getup_publication_r175_r177_20261009_result.json` and log govern next steps. Never run a competing push or mutate publish HEAD during that continuation.

Original24development≥22/24 and better than18baseline/allvalid/standardretained precede frozen316≥36/40 and better thanfixedzero/R102/allvalid, then unseen32040paired hashes/two20groups18/20/≤12sentry/strict30s/500Hzaudits. 316/318are developed;3200000–3200039remain unread/unexecuted. Larger independent perturbations/delays and right/prone/supine20each18/20still required. Simulator-only, no robot connection/enable/deployment. Do not mark full task complete.
