"""Are the grid entries at -0.074 and +0.074 genuinely different gaits?

My earlier claim was 'the reference for a reverse command equals the forward one'.
That was based on vx=+-0.05, which both round to the +-0.074 grid entries and so
happen to have similar vector norms by gait symmetry.  Compare the raw entries
directly, and compare the standing (dx=0) entry against the reverse one.
"""
import numpy as np
from playground.common.poly_reference_motion import PolyReferenceMotion

prm = PolyReferenceMotion("/data/shijinsheng/open_duck/projects/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
nb = prm.nb_steps_in_period


def entry(dx):
    ix, iy, it = prm.vel_to_index(dx, 0.0, 0.0)
    return np.asarray(prm.data_array[int(ix)][int(iy)][int(it)], dtype=float), prm.dxs[int(ix)]


rev, rev_grid = entry(-0.074)
fwd, fwd_grid = entry(0.074)
stand, stand_grid = entry(0.0)
low_cmd, low_grid = entry(-0.03)

print("grid dx used:  reverse=%.3f  forward=%.3f  standing=%.3f  tgt(-0.03)=%.3f"
      % (rev_grid, fwd_grid, stand_grid, low_grid))
print()
print("Do reverse and forward references differ?      max|rev-fwd| = %.4f" % np.abs(rev - fwd).max())
print("Does a -0.03 command get the standing reference? max|low-stand| = %.4f" % np.abs(low_cmd - stand).max())
print("Does a -0.03 command get the reverse reference?  max|low-rev|  = %.4f" % np.abs(low_cmd - rev).max())
print()
# The base forward-velocity channel (index 34) shows the commanded direction.
print("base forward-vel channel (idx 34) at phase 0 and half-period:")
print("   standing : %+.4f  %+.4f" % (stand[0, 34], stand[nb // 2, 34]))
print("   forward  : %+.4f  %+.4f" % (fwd[0, 34], fwd[nb // 2, 34]))
print("   reverse  : %+.4f  %+.4f" % (rev[0, 34], rev[nb // 2, 34]))
print()
print("mean base forward-vel over the cycle (idx 34):")
print("   standing=%.4f  forward=%+.4f  reverse=%+.4f" % (
    stand[:, 34].mean(), fwd[:, 34].mean(), rev[:, 34].mean()))
print()
print("conclusion:")
if np.abs(low_cmd - stand).max() < 1e-6:
    print("  * a -0.03 command DOES resolve to the standing reference (dx=0.000).")
if np.abs(rev - fwd).max() > 0.01:
    print("  * reverse and forward references are genuinely DIFFERENT gaits.")
