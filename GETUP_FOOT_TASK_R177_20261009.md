# R177 body-relative foot task-space feedback — 2026-10-09

This is a new finite simulation hypothesis, not a reported improvement or qualification. R175/R176 closed with selected zero extra and complete 18/24 against 18/24. Their highest nonzero/all-valid program achieved 10/24, rescuing four and regressing twelve. This motivates testing a fixed physical kinematic projection instead of adding budget to the arbitrary recurrent readout. It does not prove a unique failure cause or that all memory/feedback is ineffective.

## Distinction, causal inputs and fixed geometry

Freeze the R157 actual initial native50 selector, R122snapshot program/nodes, original right-hip feedback and IMU feedback. Only the new module uses current/causal-initial actual home-relative leg joint positions and same-phase frozen nominal leg positions. Other native channels, labels, root truth, case/seed/directory, future states and context/nearest lookup are not new-module inputs.

A private `MjData` holds a fixed canonical root `[0,0,0,1,0,0,0]`, leg positions reconstructed from actual float32 sensors and model home offsets, and head joints fixed at home. It never copies the simulator's actual root or changes the live episode/model. Only `mj_kinematics`, `mj_comPos` and `mj_jacBody` are used. There is no `mj_forward`, dynamics/integration, contacts, mesh-distance query or future-action model. The task is each foot body origin relative to the trunk in the trunk frame, **not floor height, root world position or a contact-safety predictor**.

For each leg, let `e=(FK(q)-FK(q_nominal))-(FK(q_initial)-FK(q_initial_nominal))`, in meters. The 3×5 current leg Jacobian maps the five leg actuator coordinates to that foot point. Fixed damped least squares gives `dq=-J.T*solve(J*J.T+(.01m)^2*I,e)`. Added leg targets are `.18*tanh(c_leg*dq/.18)`. The ankle column for a foot origin located on its rotation axis may be zero; the controller does not claim every leg joint always receives a nonzero correction. The four head joints receive none. Current error and the current Jacobian are recomputed throughout recovery, not merely at three nodes. Only two nonnegative response amplitudes `c_left,c_right∈[0,.05]` are searched; damping, task, point, signs and timing are fixed, not searched. This is nonlinear coordinated task-space correction, not a repeated joint-gain, IMU gate, two tracking-memory amplitudes or geometry-threshold/window budget.

The task gradient and local damped direction do not prove total mechanical-energy dissipation, safe contact, stable recovery or experts' recoverability on new states. No teacher/action imitation loss is used.

MuJoCo's Jacobian pipeline requires current kinematics and composite positions; the corresponding functions and interpretation are documented in the [official API reference](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html). Actual runtime version is recorded in `contract.json`.

## Invariants and regression evidence

At control zero the causally anchored foot error, direction and extra are exactly zero, leaving the first R157 target bitwise unchanged. Standard actual scalar errors/directions/extras are exactly zero throughout. Zero amplitude preserves original target object identity. At control 529 and later home remains original with new signals zero. All 14 joints' original/new total stays within ±.18rad around original reference before original joint/slew limits. Torque, mesh/flags, 50Hz control/500Hz physics, preparation, reward and acceptance remain unchanged. No extra settling/integration or episode root reset/teleport is added.

Twelve regressions cover causal signature, finite-difference leg Jacobians over ten postures/all ten joints, leg mapping/cross-leg independence, actual scalar nominal-zero sequence, nonzero initial-context zero, zero merge identity, unused native/head inputs, local task direction, private canonical root/model immutability, output/total target boundaries and invalid inputs.

First pure algebra test failed: changing head posture altered internal subtree-COM arithmetic and thereby leg Jacobians by at most 2.77555756e-17. No dynamic replay preceded that failure. Original three sources and failed log are preserved in unique `tmp/getup_foot_task_r177_regression_failure_01`. The corrected private leg configuration fixes head joints at home; all twelve tests pass without relaxing comparisons, output caps or physics. Initial failure is not overwritten. Corrected train source SHA256 is `7dcfe340d1d6cd314c47a50cb082a8f201863a664922c8580540dfcd3a3fdc1d`.

## Independent full smoke and formal budget

Independent command, project `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`:

`.venv/bin/python -u -m diagnostics.train_getup_foot_task_r177 --smoke`

Six complete smoke attempts: zero standard/769000/773004 reproduce all original R157 saved fields, preparation/history, initial hashes and original peaks bitwise. Fixed PROBE `[.002,.002]` keeps the standard full trajectory bitwise, entry 11.08s/strict tail 34.70s/new zero. 769002 and 773004 both fail full recovery but remain physically valid; respective added-action maxima .0004497230123450922 and .00043978207843815706rad, anchored task-error maxima .033026125082803046 and .032552414381287476m. These are two already-used development cases, not a unified result or independent qualification. Do not repeat or scale-search this PROBE.

Only after independent smoke natural exit, the safe launcher reruns twelve regressions, checks no competing getup diagnostic, output uniqueness and space. Formal command:

`.venv/bin/python -u -m diagnostics.train_getup_foot_task_r177`

Environment OMP/OPENBLAS/MKL each1, CUDA_VISIBLE_DEVICES empty, JAX_PLATFORMS cpu. Seed277, **two generations, eight proposals/generation, six CPU workers**, zero/best retained, duplicates cached. CEM mean0/std.001, elite2, .6/.4 update, std bounds .0001–.005, amplitude bounds 0–.05. At most 13 actual programs / 325 full-path training attempts; duplicate zero proposals may reduce actual counts. Every actual program uses standard plus the original24 real complete-fall cases and original maximum2279 controls/45.58s, ≤12s entry, continuous strict30s and original500Hz physical audits. Physical-invalid attempts can stop early. Ranking standard retention/all-valid/full successes/original return; no reward/acceptance change.

Formal first repeats independent six attempts and compares every old/new array, hash and peaks. All actual proposals, failures, source/model/physics hashes, parameters/distribution/history/complete RNG and closed checkpoint per generation are saved. End automatically freezes selected candidate and zero-extra baseline, each25 original full development attempts, initial-hash paired. Baseline must bitwise reproduce R15718/24, not R13417/R12215/oldzero12/R10213. No automatic expanded development or qualification.

## Immutable startup and remaining gates

Unique `archive_getup_foot_task_startup_r177.py` copies only independent smoke terminal and formal `startup_closed_01` six consistency attempts/frozen sources/models/contracts/regressions, plus the first regression failure sources/log. It excludes open training candidates. Formal startup has `terminal_result_saved=false`, `training_generations_saved=0`; it is not a training terminal or acceptance result. Never reuse its fixed output/archive/doc names. Future closed generations or final terminal require unique new archives based on actual latest main and source hashes.

After standard retained, original24 ≥22/24, strictly better than actual18 baseline and all physically valid, separately freeze expanded316≥36/40 and better than fixedzero/R102/all valid; only then unseen qualification. 3200000–3200039 remain unread/unexecuted,316/318 already developed. Paired candidate/zero new40 hashes, two disjoint20 groups each≥18/20, original entry/strict30s/500Hz validity, then larger perturbation/delay and right/prone/supine20 each18/20 remain required. Best proven unified policy still R15718/24. Keep continuation ACTIVE, no full-task/hardware-readiness claim, no real robot connection/enable/deployment.
