# R37 head/neck pulse grid: no transfer to deeper supine tilt

R37 searched 48 neck/head pitch pulse combinations under the unchanged
MuJoCo, actuator and 50 Hz slew limits. Each candidate was replayed from the
same three development starts at partial-supine tilt fractions 0.10 and 0.15,
for 288 physical episodes in total. This is a simulation-only curriculum
diagnostic, not a fully fallen recovery test.

The best development setting was neck pitch +0.35 rad, head pitch +0.25 rad
for 0.3 s. It yielded 3/3 short strict standing recoveries at fraction 0.10
but **0/3 at 0.15**. No candidate in this grid recovered the deeper 0.15
fraction. The earlier R36 pair (+0.3/+0.3 rad for 0.5 s) likewise succeeded
on some 0.10 starts but failed at 0.15, 0.20 and actual fully fallen supine
starts. The grid did not solve the transfer bottleneck; further repetition of
this limited two-joint pulse family is not justified by the evidence.

The original target remains actual prone, supine, left and right fallen starts
at >=18/20 independent recoveries each, followed by 30 s continuous strict
loaded standing. None of those pose-level completion gates has passed, and
no model or pulse from this work should be deployed to the real duck.
