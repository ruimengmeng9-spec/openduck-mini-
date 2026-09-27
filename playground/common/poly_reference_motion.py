import jax.numpy as jp
from jax import vmap
import pickle
import os


# dimensions_names = [
#     0  "pos left_hip_yaw",
#     1  "pos left_hip_roll",
#     2  "pos left_hip_pitch",
#     3  "pos left_knee",
#     4  "pos left_ankle",
#     5  "pos neck_pitch",
#     6  "pos head_pitch",
#     7  "pos head_yaw",
#     8  "pos head_roll",
#     9  "pos left_antenna",
#     10 "pos right_antenna",
#     11 "pos right_hip_yaw",
#     12 "pos right_hip_roll",
#     13 "pos right_hip_pitch",
#     14 "pos right_knee",
#     15 "pos right_ankle",

#     16 "vel left_hip_yaw",
#     17 "vel left_hip_roll",
#     18 "vel left_hip_pitch",
#     19 "vel left_knee",
#     20 "vel left_ankle",
#     21 "vel neck_pitch",
#     22 "vel head_pitch",
#     23 "vel head_yaw",
#     24 "vel head_roll",
#     25 "vel left_antenna",
#     26 "vel right_antenna",
#     27 "vel right_hip_yaw",
#     28 "vel right_hip_roll",
#     29 "vel right_hip_pitch",
#     30 "vel right_knee",
#     31 "vel right_ankle",

#     32 "foot_contacts left",
#     33 "foot_contacts right",

#     34 "base_linear_vel x",
#     35 "base_linear_vel y",
#     36 "base_linear_vel z",

#     37 "base_angular_vel x",
#     38 "base_angular_vel y",
#     39 "base_angular_vel z",
# ]


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        data = pickle.load(open(polynomial_coefficients, "rb"))
        # data = json.load(open(polynomial_coefficients))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            _coeffs = data[name]["coefficients"]

            coeffs = []
            for k, v in _coeffs.items():
                coeffs.append(jp.flip(jp.array(v)))
            _data[dx][dy][dtheta] = coeffs

        # print(self.dtheta_range)
        # exit()

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        self.data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][th] = jp.array(_data[dx][dy][dtheta])

        self.data_array = jp.array(self.data_array)

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):

        dx = jp.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = jp.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = jp.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        ix = jp.argmin(jp.abs(jp.array(self.dxs) - dx))
        iy = jp.argmin(jp.abs(jp.array(self.dys) - dy))
        itheta = jp.argmin(jp.abs(jp.array(self.dthetas) - dtheta))

        return ix, iy, itheta

    def sample_polynomial(self, t, coeffs):
        return vmap(lambda c: jp.polyval(c, t))(coeffs)

    def get_reference_motion(self, dx, dy, dtheta, i):
        # Optional continuous backward-reference speed.  The default remains
        # nearest-neighbor quantization for every skill and command.
        reference_dx_env = os.environ.get("REFERENCE_DX")
        # Keep a traced command value inside JAX; only convert the optional
        # environment override to a Python scalar during configuration.
        reference_dx = (
            jp.asarray(float(reference_dx_env), dtype=jp.float32)
            if reference_dx_env is not None
            else dx
        )
        use_dx_interp = os.environ.get("REFERENCE_DX_INTERPOLATION", "0") == "1"
        ix, iy, itheta = self.vel_to_index(reference_dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = jp.clip(t, 0.0, 1.0)  # safeguard

        use_dtheta_interp = os.environ.get(
            "REFERENCE_DTHETA_INTERPOLATION", "0"
        ) == "1"

        if use_dx_interp:
            grid = jp.asarray(self.dxs)
            clipped = jp.clip(reference_dx, self.dx_range[0], self.dx_range[1])
            upper = jp.searchsorted(grid, clipped, side="right")
            upper = jp.clip(upper, 1, len(self.dxs) - 1)
            lower = upper - 1
            low_dx = grid[lower]
            high_dx = grid[upper]
            alpha = (clipped - low_dx) / jp.maximum(high_dx - low_dx, 1.0e-8)
            alpha = jp.clip(alpha, 0.0, 1.0)
            low = self.sample_polynomial(t, self.data_array[lower][iy][itheta])
            high = self.sample_polynomial(t, self.data_array[upper][iy][itheta])
            # Continuous channels are interpolated; contacts retain the lower
            # gait's discrete schedule instead of creating fractional labels.
            out = (1.0 - alpha) * low + alpha * high
            return out.at[32:34].set(low[32:34])

        if not use_dtheta_interp:
            return self.sample_polynomial(
                t, self.data_array[ix][iy][itheta]
            )

        # Clip to the supported reference-motion range.
        dtheta_clipped = jp.clip(
            dtheta, self.dtheta_range[0], self.dtheta_range[1]
        )
        grid = jp.asarray(self.dthetas)

        # Find the two neighbouring dtheta reference gaits.
        upper = jp.searchsorted(grid, dtheta_clipped, side="right")
        upper = jp.clip(upper, 1, len(self.dthetas) - 1)
        lower = upper - 1

        dtheta_low = grid[lower]
        dtheta_high = grid[upper]

        alpha = (dtheta_clipped - dtheta_low) / jp.maximum(
            dtheta_high - dtheta_low, 1.0e-8
        )
        alpha = jp.clip(alpha, 0.0, 1.0)

        ref_low = self.sample_polynomial(
            t, self.data_array[ix][iy][lower]
        )
        ref_high = self.sample_polynomial(
            t, self.data_array[ix][iy][upper]
        )

        return (1.0 - alpha) * ref_low + alpha * ref_high


if __name__ == "__main__":

    PRM = PolyReferenceMotion(
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    )
    vals = []
    select_dim = -1
    for i in range(PRM.nb_steps_in_period):
        vals.append(PRM.get_reference_motion(0.0, -0.05, -0.1, i)[select_dim])

    # plot
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.arange(0, PRM.nb_steps_in_period)
    plt.plot(ts, vals)
    plt.show()
