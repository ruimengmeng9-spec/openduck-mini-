# R39–R41: left-side fall, transient torso-to-feet support bridge

All experiments are server simulation only. The original R4 MuJoCo scene,
3.23 Nm motor-force cap, 5.24 rad/s target slew, 50 Hz controller, substep
physical audit and strict loaded-home-standing gate were unchanged. No real
duck was operated and the existing movement controller was not replaced.

The R27 left-side reference already replayed from an actual fallen start to
a physically valid, static but unsuccessful state: body up-vector Z about
0.486, both feet touching, but torso still on the floor and the feet carrying
only about 10% of the normal load. R39 branched 168 one-joint pulses from a
snapshot of that *replayed* state, not from an artificial root teleport.
All 168 passed the physical audit. A right-hip-pitch target offset of
-0.6 rad for 0.4 s briefly produced up-vector Z about 0.508, both feet
carrying approximately 25.4 and 23.5 N, and no torso contact. After return
to HOME it reverted to torso support. That transient does **not** meet the
strict standing gate: orientation, height and foot alignment remain wrong.

R40 reproduced the brief unloaded state at simulation time about 38.58 s
in the complete replay and tried 336 additional one-joint targets from it.
All candidates passed the same physical audit. Adding a +0.6 rad target on
left hip pitch gave a momentary up-vector Z about 0.581, but returning to
HOME brought the torso back to the floor. The largest reported foot forces
were brief contact impacts, not evidence of stable loaded support. No
candidate achieved strict standing or maintained torso unloading.

R41 held that second hip target for 0.8, 1.5, 3 or 5 s at offsets +0.3 and
+0.6 rad, then returned to HOME for 3 s. All eight tests stayed within the
physical audit, but none held strict standing: final up-vector Z was around
0.48 with torso contact in seven cases, and near zero with torso contact in
the longest +0.6 rad case. A fleeting support transfer is therefore not a
sustained recovery method.

These development searches use seed 0 and have no independent 20-start/30-s
acceptance. The next search should target sustained torso unloading and
rotation toward upright, not merely maximize a single contact-force sample.
The code, exact input-reference hash, results and failed time-series measurements are archived
under corresponding `results/getup_*_bridge_r39/r40/r41` directories. The
full get-up objective remains unachieved.
