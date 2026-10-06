# R159 bilateral coupling audit and R160 causal bilateral feedback

## Status and evidence boundary

R159 is a completed read-only diagnostic. R160 is a new bounded simulation feedback-parameter search, not an end-to-end neural-network retraining. Its independent smoke has naturally exited and formal startup is closed; formal training is still running. This archive does not contain closed training generations or terminal development results. The best previously verified unified controller remains R157, 18/24 original development cases, standard retained and no physical invalidity. No qualification or hardware deployment is authorized by these results.

## R159: complete paired evidence

The six remaining R157 failures are 769002, 769004, 773001, 773004, 773005 and 773015. Eight R133 complete physically valid alternative-program trajectories were paired with their R157 counterparts. Initial hashes and initial observations match. No dynamic integration or acceptance relabeling was performed; source hashes are unchanged.

Every alternative changes the right-side executed target at control 0. Original left-side targets subsequently differ at control 1 or 2, and left-side joint sensors at control 1 or 2. The 1e-8 rad thresholds are diagnostic markers, not failure thresholds or proofs of irrecoverability. The eight comparisons are only six known starting states, not independent qualification trials. This supports testing bilateral response, but does not prove a unique failure cause or that such response will improve get-up.

Actual native joint indices are left hip yaw/roll/pitch 0/1/2 and right hip yaw/roll/pitch 9/10/11; actuator IDs are resolved separately from the unchanged model. Paired arrays retain current native55 sensors, position/velocity errors, common/differential error modes, causal initial-subtracted modes, executed target differences and original strict-standing markers. Root truth is not a controller input.

## R160: bounded new hypothesis and scalar execution

The R157 initial native50 selector, R122 snapshot program/knots and original state feedback remain frozen. Twelve new coefficients describe actuator-coordinate common and differential position/velocity error-change modes of the six hip joints. These coordinates are not a claim of world-axis symmetry or angular-momentum conservation.

For each side, actual current joint position/velocity minus the saved same-phase standard sensor value is scaled by .05 rad / the native velocity divisor .05. Common modes are (left+right)/2 and differential modes (right-left)/2. The corresponding causally sampled initial modes are subtracted, then independently passed through tanh. Six common and six differential coefficients produce bounded direct residuals; left receives common minus differential, right common plus differential. Unlike R155's right-only six coefficients, this couples both measured sides to both hip targets. It does not multiply the new activation by the current position error or change selector weights.

Only actual current and causally retained initial sensors, nominal sensor references and phase are available. No root position/linear velocity, case/seed/directory/success label, future substep state, nearest-neighbor lookup or exact context matching enters inference. The first target is exactly R157's. Standard scalar activations and new extra actions are exactly zero. Zero coefficients preserve original target objects and all saved original trajectory fields. Home adds no new feedback. Original-plus-new corrections remain within .18 rad around the reference for each joint, followed by unchanged joint and slew limits; collision, torque, physics, reward and acceptance are unchanged. No extra preparation integration or mid-episode root edits occur.

## Regression and independent complete smoke

Twelve regressions passed, covering causal interfaces, exact initial/nominal zero, zero coefficients, independent common/differential behavior, bilateral sensor influence, native velocity units, unused-sensor invariance, odd error-change response, finite/bounded inputs and total target bounds.

Independent full-path smoke contains six trajectories. Zero coefficients for standard/769000/773004 reproduce all R157 saved original fields, initial hashes and physical peaks exactly. The pre-fixed nonzero coefficients are [.005,.004,.003,.002,.003,.004,.003,.004,.005,.004,.003,.002]. Standard retains the complete original trajectory with all new actions zero: entry 11.08 s and strict tail 34.70 s. Nonzero 769002 succeeds physically valid, entry 11.04 s/tail 34.74 s; 773004 fails but is physically valid. Both perturbations execute genuinely nonzero left and right actions. These are known-development interface/smoke results, not unified-policy performance or independent qualification.

Formal startup repeats the same six trajectories; the archive asserts every saved array, initial hash and original physical peak equal to independent smoke. The formal startup snapshot is explicitly terminal_result_saved=false and training_generations_saved=0.

## Fixed training budget and gates

Seed 260; six generations, twelve candidates per generation, six CPU workers. Twelve coefficients are bounded by +/-1. CEM starts at zero mean/std .01, preserves zero/best, caches duplicate proposals, uses three elites with .6/.4 updates, and bounds std to .002–.05. All actual candidate trajectories, failures, parameters, distributions, history and RNG are saved in closed generation checkpoints. Training uses standard plus the original 24 actual complete left-side fallen starting states. The 629-control-step/12.58 s path with a one-second short tail is only the unchanged training label/reward for ranking, never acceptance.

After six generations, the parent automatically performs candidate and zero-new-feedback baseline on standard plus 24 cases with all 2279 controls/45.58 s. Baseline must exactly reproduce R157's 18/24; this is not old zero12/R10213 or R13417. Initial hashes must pair. Original deadline 12 s, continuous strict standing 30 s, and every 500 Hz physical substep audit remain required. Standard retained, >=22/24, better than this baseline, and all physically valid are prerequisites for a separately frozen expanded 316-development test >=36/40 and better than fixed zero/R102, all valid. The parent does not automatically expand or qualify.

3160000–3160039 and 3180000–3180039 are development. 3200000–3200039 remain reserved, unread and unexecuted. Only a frozen development-gated controller may test new qualification with paired zero, two disjoint groups each >=18/20, original timing/standing/physics conditions. Broader unseen/delay tests and right/prone/supine acceptance remain outstanding. No perfect-policy or hardware-readiness claim is made.

## Append-only preservation

Parent baseline fd4c1edeec3617e91d064aa8a37dfe5ce5640120. New unique R159 terminal_snapshot, independent R160 smoke terminal_snapshot and formal R160 startup_closed_01 preserve source/model/physics hashes, six plus six dynamic startup/smoke trajectories and all read-only diagnostic arrays. No existing terminal, startup, failed-interface or checkpoint evidence is replaced. Later closed generations and terminal results require a new unique snapshot and the actual latest remote baseline.
