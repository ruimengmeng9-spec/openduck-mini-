# R35 full-fall-only stage 3 and R38 actor-action diagnosis

This server-side experiment is simulation only. No real robot was connected
or commanded, and the existing movement controller was not replaced.

Stage 3 continued the stage-2 actor but sampled **only actual full-fallen**
prone, supine, left-side and right-side starts. It completed 128 PPO iterations,
262,144 policy decisions and 1,310,413 50 Hz motor controls. Across 4,437
completed full-fall training episodes, there were **zero** one-second strict
loaded-standing discoveries. The final batch's return was not better than
the first batch's (-0.528 versus -0.419). This isolates the failure from
stage 2's 80% partial-tilt sample mixture; simply increasing the proportion
of full falls did not produce a recovery trajectory.

Independent native-MuJoCo six-second validation used three held-out seeds
per condition. Normal standing remained 3/3. Difficult partial-tilt boundary
results were prone 0/3, supine 1/3, left side 3/3 and right side 3/3.
**Actual full-fall results were 0/3 in each of the four orientations.** No
30-second acceptance claim follows from this failed six-second screening.

R38 then replayed the stage-2, stage-3 midpoint and stage-3 final ONNX
actors deterministically from the same fully fallen seed, recording actions,
physical state and failed trajectories without modifying physics. At the
stage-3 final actor, the largest target deviation from home during six seconds
was approximately 0.357 rad prone, 0.101 rad supine, 0.037 rad left-side and
0.022 rad right-side. Maximum achieved body up-vector Z was approximately
0.241, 0.000, -0.232 and -0.231 respectively. Supine and both side poses
never acquired foot support in this replay. These figures are single-seed
diagnostics, **not** broader success or impossibility estimates.

The trained deterministic controller is close to a HOME hold for supine and
side falls, despite training on thousands of actual falls. Along with earlier
R34 physical-valid single/pair joint pulses that produced some angular change
but no strict standing, this points to a missing multi-stage contact-transfer
teacher or search trajectory rather than merely a low standing-retention
score. The next experiment should seek a physically valid sequence that first
transfers body/neck contact into foot support, and only then distill a
feedback policy. It must keep the original MuJoCo geometry, motor limits,
50 Hz slew, and strict loaded-standing gate.

Executed source snapshots, ONNX/msgpack checkpoints, all episode records,
independent evaluation results, launcher log and R38 action/trajectory traces
are archived in the matching `results/` directories. Final completion remains
at least 18/20 independent recoveries per actual fallen orientation followed
by 30 seconds continuous strict loaded standing; it is **not achieved**.
