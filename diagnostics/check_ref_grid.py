"""What velocity grid does the reference motion actually contain?"""
from playground.common.poly_reference_motion import PolyReferenceMotion
import jax.numpy as jp

prm = PolyReferenceMotion("/data/shijinsheng/open_duck/projects/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
print("dx range    :", prm.dx_range)
print("dy range    :", prm.dy_range)
print("dtheta range:", prm.dtheta_range)
print()
print("dxs   (", len(prm.dxs), "):", prm.dxs)
print("dys   (", len(prm.dys), "):", prm.dys)
print("dthetas(", len(prm.dthetas), "):", prm.dthetas)
print()
print("nb_steps_in_period:", prm.nb_steps_in_period)
print()
print("Which grid entry does each command map to?")
for v in (-0.10, -0.05, -0.03, 0.0, 0.03, 0.05, 0.10, 0.30):
    ix, iy, it = prm.vel_to_index(v, 0.0, 0.0)
    print("  cmd %+0.3f -> ix=%s (dx=%.3f)  iy=%s  itheta=%s" % (
        v, int(ix), prm.dxs[int(ix)], int(iy), int(it)))
