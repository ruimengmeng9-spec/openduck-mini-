"""Compare the backward reward for 'stationary' vs 'tracking' before and after the
overspeed/imitation fix.

The environment applies  reward = clip(sum(term * scale) * dt, floor, 10000),
so this reproduces that sum for two policies: one that stands still, and one that
walks backwards at the commanded speed while the gait oscillates.
"""

DT = 0.02
V_CMD = -0.04  # the middle of the backward curriculum
SIGMA = 0.0025

# Terms that do not depend on the velocity choice are omitted (they cancel).
def net(imitation_scale, overspeed_scale, allowed_fn, v_actual, oscillation):
    # tracking_xy: exp(-||cmd - v||^2 / sigma)
    xy_err = (V_CMD - v_actual) ** 2
    tracking = 60.0 * jp_exp(-xy_err / SIGMA)
    # lin_vel_xy_error
    lin_err = -10.0 * xy_err
    # normalized_progress / progress_shortfall (command direction is -x)
    speed = abs(V_CMD)
    direction = -1.0
    progress = direction * v_actual
    ratio = progress / max(speed, 0.02)
    norm_prog = 80.0 * min(max(ratio, -1.0), 1.0)
    shortfall = -80.0 * max(0.80 - ratio, 0.0)
    wrong_way = -80.0 * max(-ratio, 0.0)
    # overspeed uses the worst instantaneous speed seen during the gait
    peak = abs(v_actual) + oscillation
    overspeed = -overspeed_scale * max(peak - allowed_fn(speed), 0.0)
    # upright is roughly equal in both cases on flat ground
    upright = 20.0
    total = tracking + lin_err + norm_prog + shortfall + wrong_way + overspeed + upright
    return total * DT, dict(tracking=tracking*DT, progress=(norm_prog+shortfall+wrong_way)*DT,
                            overspeed=overspeed*DT, tracking_xy=tracking*DT, upright=upright*DT)


def jp_exp(x):
    import math
    return math.exp(x)


OLD_ALLOWED = lambda s: 1.10 * s + 0.005
NEW_ALLOWED = lambda s: max(1.5 * s, 0.15)

GAIT_OSC = 0.10  # peak-to-mean velocity fluctuation of a walking gait (m/s)

print("target command: %.2f m/s, gait oscillation: +-%.2f m/s" % (V_CMD, GAIT_OSC))
print()
print("%-38s %-12s %-12s %s" % ("scenario", "per-step", "per-episode", "components"))
for label, imitation, overspeed, allowed, v in [
    ("BEFORE: stationary", 0.25, 4000.0, OLD_ALLOWED, 0.0),
    ("BEFORE: tracking", 0.25, 4000.0, OLD_ALLOWED, V_CMD),
    ("AFTER : stationary", 0.0, 600.0, NEW_ALLOWED, 0.0),
    ("AFTER : tracking", 0.0, 600.0, NEW_ALLOWED, V_CMD),
]:
    per, comp = net(imitation, overspeed, allowed, v, GAIT_OSC)
    print("%-38s %-12.3f %-12.0f ovs=%+.2f prog=%+.2f trk=%+.2f" % (
        label, per, per * 1000, comp["overspeed"], comp["progress"], comp["tracking_xy"]))

print()
for tag, imitation, overspeed, allowed in [
    ("BEFORE", 0.25, 4000.0, OLD_ALLOWED),
    ("AFTER ", 0.0, 600.0, NEW_ALLOWED),
]:
    stat, _ = net(imitation, overspeed, allowed, 0.0, GAIT_OSC)
    track, _ = net(imitation, overspeed, allowed, V_CMD, GAIT_OSC)
    verdict = "MOVING is better" if track > stat else "STANDING STILL is better  <-- degrades to standing"
    print("%s: moving %+.3f vs still %+.3f per step  ->  %s" % (tag, track, stat, verdict))
print()
print("note: imitation contributes 0 on flat ground in this arithmetic (its effect")
print("      shows up through the reference-joint tracking term, measured separately")
print("      by check_ref_motion.py: reference for -0.03 == reference for +0.03).")
