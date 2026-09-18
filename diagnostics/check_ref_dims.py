"""What is the actual dimensionality of the reference vector?

reward_imitation slices reference_frame at [0:16] joints, [16:32] joint vels,
[32:34] contacts, [34:37] base lin vel, [37:40] base ang vel.  But a diagnostic
suggested get_reference_motion returns only 16 values.  If so, every sub-term
beyond the joint positions is operating on empty slices -- a silent defect that
would explain why the imitation reward cannot stabilise a gait (no velocity or
contact target is ever provided).
"""

import pickle
import numpy as np

from playground.common.poly_reference_motion import PolyReferenceMotion

P = "/data/shijinsheng/open_duck/projects/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
raw = pickle.load(open(P, "rb"))
name0 = next(iter(raw))
print("pkl top-level keys:", len(raw))
print("sample key:", name0)
print("fields in one entry:", list(raw[name0].keys()))
print()
for k in raw[name0]:
    v = raw[name0][k]
    if isinstance(v, dict):
        print("  %-40s dict with %d keys -> %s" % (k, len(v), list(v.keys())[:6]))
    else:
        print("  %-40s %r" % (k, v))

coeffs = raw[name0]["coefficients"]
print()
print("coefficients: %d entries" % len(coeffs))
for i, (ck, cv) in enumerate(list(coeffs.items())[:20]):
    print("   [%2d] %-24s len=%d" % (i, ck, len(cv)))
print()
print("=> get_reference_motion therefore returns a %d-dim vector." % len(coeffs))

prm = PolyReferenceMotion(P)
ref = np.asarray(prm.get_reference_motion(-0.074, 0.0, 0.0, 0), dtype=float)
print("observed get_reference_motion shape:", ref.shape)
print()
print("reward_imitation expects these slices:")
for lo, hi, label in [(0,16,"joint pos"),(16,32,"joint vel"),(32,34,"contacts"),
                      (34,37,"base linvel"),(37,40,"base angvel")]:
    if hi <= ref.shape[0]:
        print("   [%2d:%2d] %-12s -> %d values  OK" % (lo, hi, label, hi-lo))
    else:
        print("   [%2d:%2d] %-12s -> EMPTY (vector only has %d)  <-- dead term"
              % (lo, hi, label, ref.shape[0]))
