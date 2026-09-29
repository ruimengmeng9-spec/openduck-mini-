# R35 curriculum stage 0: completed, but no full-fall recovery

This is a **simulation-only training experiment**, not a hardware-ready model.
The original MuJoCo physics, motor force/position/slew limits and strict
loaded-home-standing gate were kept. The reset curricula use physical settling
after an initial tilt; the root is never edited during an episode. Only a
90° canonical pose settled into actual torso contact counts as a full fall.

The R33 final actor had forgotten its original standing behavior. R35 therefore
started from the R33 **initial** actor, not the failed final actor. R35 uses an
exact state-dependent AR(1) PPO action density: latent standard deviation is
0.02 near upright and grows only after the trunk is down. A one-iteration
smoke test at learning rate 1e-4 gave approximate KL 0.129; reducing the
curriculum-mode learning rate to 5e-6 gave 0.0066. Existing R33 settings were
unchanged when the curriculum flag is omitted. Twelve related unit tests passed.

Stage 0 trained for 64 iterations with 16 environments, 131,072 policy
decisions and 655,360 actual 50 Hz motor controls. It ended normally and
exported `final.onnx`. The final batch contained only partial tilts and had
32/32 one-second strict standing discoveries. This is expected because the
fixed home target already recovered these easy starts.

The deterministic final model was then tested for 6 s on three held-out seeds
per condition. It held a normal standing start in 3/3. At the first difficult
partial-tilt boundary, it achieved prone 0/3, supine 1/3, left side 3/3 and
right side 3/3. From **actual fully fallen** starts, it achieved 0/3 in each
of prone, supine, left side and right side. The initial actor and checkpoints
16 and 32 produced essentially the same boundary results on their shared
held-out seeds. Stage 0 therefore preserved standing but did not demonstrate
meaningful new recovery skill.

Do not promote this model to real hardware. Stage 1 trains on a mixture that
includes the measured failing partial-tilt boundary. Stage 2 introduces some
actual full falls; stage 3 is entirely actual full falls. Each stage still
requires independent evaluation, and only the original full-task gate of at
least 18/20 for every fully fallen orientation plus 30 s continuous loaded
standing can establish completion.
