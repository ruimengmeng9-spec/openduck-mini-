# R181 full episode policy gradient terminal and R182 learner audit

R181 completed all four authorized updates and final development, then naturally exited. R182 independent read-only smoke and formal audit also naturally exited. The selected actor retains its initial zero output head: candidate and frozen R157 baseline both recover **18/24**, retain the standard recovery and have zero physical-invalid episodes. The original development gate is false. There is no improvement, expanded development, unseen qualification or hardware readiness. Continue ACTIVE with the original gates unchanged.

## Commands and complete attempts

Project `/data/shijinsheng/open_duck/projects/Open_Duck_Playground`, existing `.venv`. OMP/OPENBLAS/MKL each1, CUDA_VISIBLE_DEVICES empty, JAX_PLATFORMS cpu.

```text
/data/shijinsheng/open_duck/projects/Open_Duck_Playground/.venv/bin/python -u -m diagnostics.train_getup_full_episode_pg_r181
.../.venv/bin/python -u -m diagnostics.launch_getup_full_episode_pg_audit_r182
.../.venv/bin/python -u -m diagnostics.audit_getup_full_episode_pg_terminal_r182 --smoke
.../.venv/bin/python -u -m diagnostics.audit_getup_full_episode_pg_terminal_r182
```

R181 output is `/data/shijinsheng/open_duck/outputs/getup_full_episode_pg_r181_left_20261010`, with corresponding `.log`, all checkpoints update0000 through0004, `training_closed.json`, `selected_actor.npz` and `results.json`. Actual training command has no extra arguments. Original startup and executed source are immutable; SHA256 `359cf8572675ff5dba32633f58ed71833c250fea5a7d707ef131597fd5068115`.

Seed281, four updates, sixCPU, two independent exploratory attempts for each of the standard plus24 original complete-fall starts:200training attempts. Eight formal startup checks,25baseline,200training,100deterministic checkpoint development and25final candidate give **358formal dynamic attempts**. Eight previously saved independent smoke attempts bring total366, not366independent starting states. Each attempt uses original maximum2279controls/45.58s, entry≤12s, continuous strict30s and500Hzphysical audits. Physical-invalid attempts stop early; not all200training attempts walk all45.58s.

All1376 input/recurrent/output parameters are learnable. Fixed8hidden,150causal features, .0001latent innovation scale, one Adam.001/globalnorm1 update per closed50attempt batch, no critic or sample reuse, remain as in `GETUP_FULL_EPISODE_PG_R181_20261010.md`. The companion baseline is the other independent same-case complete original return, not an antithetic noise rollout, teacher action or online success oracle. The saved score loss omits parameter-independent variance and tanh-Jacobian terms for the actual sampled history and is not a full log-likelihood value. Original whole-episode return is not equivalent to strict30s recovery success.

## Deterministic checkpoint results

| Update | Original24 successes | Physical invalid | Rescued from R157 | Regressed from R157 |
| --- | ---: | ---: | --- | ---: |
| 1 | 5 | 1 | 773001,773015 | 15 |
| 2 | 9 | 2 | 773001 | 10 |
| 3 | 6 | 1 | 773001 | 13 |
| 4 | 6 | 1 | none | 12 |

All four retain standard. No nonzero checkpoint is all physically valid. Highest nonzero full-development success count is update2,9/24 with invalid2. It regresses769000/769001/769003/769006/769007/773000/773003/773007/773009/773013. The selected zero-output initial actor and final candidate preserve all25baseline fields, initial hashes and original peaks; baseline original fields reproduce R157 bitwise. There is no reason to proceed to expanded316development or unread320qualification.

First update input and recurrent gradient norms and parameter changes are exactly zero because initial output head is zero; output gradient norm96066.65370671597 and maximum change.0009999999670569073. Updates2–4 have nonzero input, recurrent and output gradients and changes. For update2 these norms are598.590997449068/157.95089230690644/146918.2123624186, with changes.0007441264846893558/.0007441170164069894/.001001352890769389 respectively. Learning all parameter blocks did not produce overall improvement; neither this result nor zero first-update hidden gradients proves a unique failure cause or that all nonlinear feedback is useless.

## R182 exact reconstruction

R182 outputs `/data/shijinsheng/open_duck/outputs/getup_full_episode_pg_terminal_audit_r182_smoke_20261010` and `getup_full_episode_pg_terminal_audit_r182_20261010`. Safe launcher reruns the original14regressions, checks unique outputs/no competing getup task/space, completes independent read-only smoke and waits for natural exit before formal audit. No new regression failure or modified R181 source is introduced.

Independent smoke reconstructs eight existing formal startup attempts and25terminal pairs. Formal audit reconstructs all358formal attempts. White-listed actual observations and causal initial observations reproduce150features, noise amplitude,8hidden,14mean/latent/extra and frozen right-hip feedback **bitwise**. Independent per-episode innovation RNG reproduces saved innovations exactly. Selector gains/choice/scores reproduce from actual preparation sensors. Control0, standard and home added signals are zero; recorded pre-integration planned target equals applied target. Original per-step reward sums exactly reproduce the separately saved full-episode sum and agree with original return_sum within the original1e-10accumulation check.

All four updates independently rebuild the actual50attempt feature/latent/amplitude batch in original order, use the original padded529recovery horizon for early-invalid trajectories, reconstruct both complete returns and companion advantages, and run the same float64 JAX score gradient/Optax update. Saved score loss, gradient norms and maximum changes match exactly. **All five complete actor plus optimizer msgpack checkpoints match byte for byte**, including Adam first/second moments and step counts. Initial actor, each checkpoint RNG, cumulative history and final closed RNG match. R182 saves full per-block gradient arrays and return/advantage/seed arrays for each update. NumPy runtime means versus JAX training means retain the original1e-12tolerance; cross-implementation bitwise equivalence is not claimed.

All25candidate/baseline saved fields and original peaks/hash match; separate baseline comparison reproduces every original R157 field. Root qpos/qvel are decoded only for this isolated whole-field equality comparison, never scalar/learner reconstruction or runtime policy input. Scalar arrays use a whitelist. SHA256 hashing reads complete compressed file bytes, including files containing recorded root arrays, without decoding those arrays for feedback. Execution, frozen/source/physical files and all tracked input evidence hashes remain unchanged.

Formal first-eight statistics and25pairing match independent audit smoke. Three PROBE signal arrays are compared bitwise during terminal archiving;25update2representative arrays are separately saved. These28formal signal arrays are not50representatives or new dynamic attempts. The audit constructs no environment/MjData, calls no forward/integration/contact or force computation and does not reclassify original success or physical validity. It does not independently rerun standalone joint/slew planning. Saved same-state double pre-slew differences and post-slew direct differences are distinct from subsequent executed differences between different visited trajectories; none is a dynamic counterfactual or immediate transfer gain.

Audit source SHA256 `e63a534bd2b22999adc32f6cfaccdd652a9574ca453a95953f948a99e128c214`.

## Original invalid evidence

Update2 case773003 saves304controls, self peak.048730133245795414m, maximum extra.000009031374480201359rad, total correction.06847238919441165rad. Case773009 saves45controls, self peak.030023386850996318m, extra.0000027915172853861757rad, total.03079189836988247rad. Other original physical categories do not exceed. Updates1/3/4 also invalidate773009 at45saved controls, with self peaks.030023388965802998/.030023395232344156/.030023405556852907m. Complete original peaks and every failed trajectory remain saved.

These control endpoints/counts are not first500Hzcrossing times. Very small actions and staying below±.18cap do not imply contact safety or justify relabeling. Do not repeat the same fixed PROBE or increase this four-update/noise/network budget. Different visited states and their subsequent IMU/target divergence are not same-state immediate-action comparisons.

## Immutable publication and remaining work

New `archive_getup_full_episode_pg_terminal_r181_r182.py` appends R181358formal trajectories/all failures/five checkpoints/four learners/RNG and R182formal/smoke terminal snapshots. Independent R181eight full smoke trajectories stay in their already published immutable terminal snapshot and are rechecked against formal startup. New record/scripts/logs/manifests use new paths; no old startup, executable, failed record or bundle is overwritten. Exact independently verified clean base is `810b93430a8d00f1269d4a99a2702705c56c5c1a`; terminal commit and publication success depend on completed upload and independent remote verification, not merely local archiving.

Publish directly from the authorized clean server repository using repo-only deployment key, proxy HTTP CONNECT and strict official GitHub host verification. No private credential is printed, exported or stored in evidence. No robot connection/enable/deployment or walker modification occurs; protected live_server is left running. /data available21GB before archiving; do not delete others' or unconfirmed data.

Next continuation first checks actual R183or higher tasks and immutable records. A new finite method must differ structurally from exhausted full-episode REINFORCE, fixed recurrent readout, foot-position/orientation amplitudes, gains/residuals/expert mix/velocity/tracking-memory/IMU gates/static exact interpolation/oldBC/geometric-threshold or time-window budgets. It must state its hypothesis, causal inputs and finite budget, pass actual scalar nominal-zero/initialization/boundary/full-original parity regressions and independent full smoke before training. No method implementation or loss alone is improvement evidence.

Original14joint old/new total reference±.18rad, original joint/torque/slew/home/mesh/flags/50Hz/500Hz/physics/reward/acceptance remain fixed. No extra preparation integration, waiting/settling or mid-episode root reset. Runtime only causal actual sensors/history/internal state/phase; case/seed/labels/root truth/teacher and future states are never inference inputs.

Standard retained, original24≥22/24 and better than18baseline/all valid must precede separately frozen316≥36/40 and better thanfixedzero/R102/all valid, then unseen32040paired hashes/two20groups each≥18/20, entry≤12s/continuous strict30s/original500Hzaudits. 316/318are developed;3200000–3200039remain unread/unexecuted. Larger independent perturbations/delays and right/prone/supine20each18/20remain mandatory. Continue ACTIVE, full task and hardware readiness false.
