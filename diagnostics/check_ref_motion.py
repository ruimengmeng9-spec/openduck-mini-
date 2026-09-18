"""Check whether the imitation reference motion is valid for backward commands."""
import numpy as np
from playground.common.poly_reference_motion import PolyReferenceMotion

prm = PolyReferenceMotion("playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
print("nb_steps_in_period:", prm.nb_steps_in_period)
print()
print("%-10s %-8s %-10s %-10s %s" % ("cmd_vx", "phase", "|ref|", "max|ref|", "finite"))
for vx in (0.0, 0.03, 0.10, -0.03, -0.05, -0.10):
    for phase in (0, prm.nb_steps_in_period // 2):
        ref = np.asarray(prm.get_reference_motion(vx, 0.0, 0.0, phase), dtype=float)
        print("%-10.3f %-8d %-10.4f %-10.4f %s" % (
            vx, phase, float(np.linalg.norm(ref)), float(np.abs(ref).max()),
            bool(np.all(np.isfinite(ref))),
        ))
    print()
