"""Check that backward support shaping uses actual active foot contacts."""

import unittest
from types import SimpleNamespace

import jax.numpy as jp

from playground.open_duck_mini_v2.focused_skill import rear_contact_margin


class RearContactMarginTest(unittest.TestCase):
    def test_uses_rearmost_active_foot_floor_point(self):
        contact = SimpleNamespace(
            geom=jp.array([[46, 18], [43, 46], [46, 7], [46, 18]]),
            dist=jp.array([-0.001, -0.002, -0.01, 0.1]),
            pos=jp.array([
                [-0.02, 0.0, 0.0], [0.01, 0.0, 0.0],
                [-0.50, 0.0, 0.0], [-0.80, 0.0, 0.0],
            ]),
        )
        data = SimpleNamespace(contact=contact)
        margin = rear_contact_margin(
            data, 46, [18, 43], jp.array([0.0, 0.0]), jp.array([1.0, 0.0])
        )
        self.assertAlmostEqual(float(margin), 0.02, places=6)

    def test_no_active_contact_is_unsafe(self):
        contact = SimpleNamespace(
            geom=jp.array([[46, 18]]), dist=jp.array([0.1]),
            pos=jp.array([[-0.02, 0.0, 0.0]]),
        )
        data = SimpleNamespace(contact=contact)
        margin = rear_contact_margin(
            data, 46, [18, 43], jp.array([0.0, 0.0]), jp.array([1.0, 0.0])
        )
        self.assertEqual(float(margin), -1.0)


if __name__ == "__main__":
    unittest.main()
