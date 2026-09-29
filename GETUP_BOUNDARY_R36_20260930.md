# R36: partial-tilt recovery bridge, not full-fall get-up

After R35 stages 0 and 1 failed to improve independent prone/supine boundary
tests, R36 tested whether the unchanged action space and motor dynamics could
produce a corrective trajectory. All experiments are simulation only. They
retain the original MuJoCo model, actuator force/position limits, 50 Hz slew,
substep physical audit and strict loaded-home-standing criterion. The root
is positioned only at episode reset.

First, 48 temporally correlated random rollouts around the trained standing
actor were tried at the first failing partial-tilt boundaries (prone 0.20,
supine 0.10 of the 90° canonical fall angle). Noise scales 0.05, 0.1 and 0.2
gave **zero** one-second standing recoveries. Prone trajectories often crossed
the original 1 cm floor-penetration audit around 0.6 s; supine trajectories
remained physically valid but typically tipped over.

Next, 114 single-joint pulses per direction were tested. Neither direction
produced a full one-second strict stand. The best supine pulse, +0.3 rad at
head pitch for 0.5 s, produced 0.48 s of strict standing. A coordinated pair
search (86 prone and 82 supine candidates) found no prone recovery, but one
supine candidate: +0.3 rad at both neck pitch and head pitch for 0.5 s,
followed by the original home-joint target. It was physically valid and
finished with 3.18 s of continuous strict, fully foot-loaded standing on its
development start.

Independent 6-second replay of this pulse achieved 3/3 short standing
recoveries from supine **partial tilt 0.10**, but 0/3 at partial tilts 0.15
and 0.20, and 0/3 from **actual fully fallen supine** starts. In a separate
33-second partial-tilt test, 4/5 starts finished with over 32 s of continuous
strict loaded standing; one never acquired a one-second stand. These results
are a limited, imperfect curriculum bridge, **not** a successful get-up model.

R37 tunes the pulse parameters and will require new held-out seeds. The
original full-task requirement is unchanged: actual prone, supine, left and
right fallen starts, at least 18/20 independent successes for each, followed
by 30 s continuous strict loaded standing. No result here authorizes real
robot deployment.
