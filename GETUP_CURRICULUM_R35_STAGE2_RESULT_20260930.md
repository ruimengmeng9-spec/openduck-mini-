# R35 stage 2: completed partial-to-full curriculum, no actual-fall recovery

After fixing the prior-episode audit contamination, stage 2 completed all
128 PPO iterations from the stage-1 actor: 262,144 policy decisions and
1,309,375 actual 50 Hz motor controls. The training distribution contained
20% actual full-fall resets alongside harder physical partial tilts. Of 992
completed episodes with actual fully fallen starts, **zero** met the
one-second strict loaded-standing discovery label. This was not another
reset crash: the run finished and exported its actor. Average batch return
did not show a meaningful rising trend (about 1.49 in iterations 1–16 and
1.61 in iterations 65–128).

Independent deterministic native-MuJoCo tests used three held-out seeds per
condition, six seconds each, original collision model and motor limits, and
the unchanged strict loaded-standing criterion. The final actor retained
standing in 3/3. The first difficult partial-tilt boundaries were prone
0/3, supine 1/3, left-side 3/3, right-side 3/3. **Actual full-fall starts
were 0/3 in prone, supine, left-side, and right-side.** The independent
validator also clears prior audit peaks before each full-fall preparation,
while still auditing the newly settled state.

Across full-fall training episodes, the maximum final up-vector Z remained
below 0.34 in every orientation; no completed full-fall episode acquired
even a nonzero strict-standing tail. This is stronger evidence of an
exploration/contact-mode bottleneck than of a merely insufficient 30-second
hold. Another identical PPO extension is not justified. The next iteration
needs to find a physically valid staged transition from body/neck contact
to foot support before distilling or learning a full-fall controller.

The complete training directory, launcher log, ONNX and msgpack checkpoints,
all episode records, held-out failure trajectories and executed-source
snapshots are archived under
`results/getup_curriculum_r35_stage2_retry_20260930`.

This experiment is **simulation only**. It does not prove hardware readiness,
does not replace the existing motion controller, and does not meet the final
gate of at least 18/20 independent actual-fall recoveries per orientation
followed by 30 seconds of continuous strict loaded standing.
