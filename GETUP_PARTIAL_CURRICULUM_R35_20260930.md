# R35: stand retention and partial-tilt curriculum probe (simulation only)

The current full-fall task is **not solved**. R33's initial exported actor held a
loaded strict stand for 6 s in both tested standing starts; its final actor
held neither. A fixed home-joint target held 6 s in four independent standing
starts. This isolates a training-induced loss of the existing standing skill,
not a failure of the standing gate itself.

An AR(1) latent perturbation of the initial actor's action was tested for 6 s
from three independent standing seeds at each fixed noise scale. At standard
deviation 0.01 and 0.02, 3/3 reached the strict 1 s tail; at 0.05, 1/3 did;
at 0.1, 0.25 and 0.5, 0/3 did. These are small diagnostic samples, not
reliability estimates. The R33 training scale of 0.5 is much too disruptive
for standing retention on these starts.

`diagnostics/probe_getup_partial_curriculum_r35.py` sets the root orientation
once at reset, aligns it with the floor, settles it for 5 or 10 original motor
controls, and then holds the original home-joint target for 3 s. There are no
root edits during recovery, no changes to the MuJoCo model or motor limits,
and the original strict loaded-standing gate remains in use. Two perturbed
seeds were tested per condition:

| Tilt as fraction of 90° | Prone | Supine | Left side | Right side |
| --- | ---: | ---: | ---: | ---: |
| 0.05, 5-control settle | 2/2 | 2/2 | 2/2 | 2/2 |
| 0.10, 5-control settle | 2/2 | 1/2 | 2/2 | 2/2 |
| 0.15, 5-control settle | 2/2 | 0/2 | 2/2 | 2/2 |
| 0.20, 5-control settle | 0/2 | 0/2 | 2/2 | 2/2 |
| 0.35, 5-control settle | 0/2 | 0/2 | 0/2 | 0/2 |

The 10-control settle at 0.20 and 0.35 gave the same 0/2, 0/2, 2/2, 2/2
and 0/2, 0/2, 0/2, 0/2 rows respectively. Successful conditions were small
dynamic tilts, **not actually settled prone/supine/side-down poses**; they do
not count toward full-fall recovery. The negative conditions supply a graded
boundary for the next curriculum experiment.

Next: retain a verified deterministic home-standing branch while training
recovery from low, physically valid, increasingly difficult starts. Begin with
exploration no larger than the measured safe range, audit the exact PPO action
distribution and validate strict standing after every checkpoint. Do not claim
the skill until all four fully fallen orientations pass 18/20 independent
starts and a subsequent 30 s continuous loaded stand.
