"""Domain randomization used by the Open Duck Mini environments."""

from __future__ import annotations

import os
from collections.abc import Sequence

import jax
import jax.numpy as jp
from mujoco import mjx


FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    floor_geom_id: int = FLOOR_GEOM_ID,
    foot_geom_ids: Sequence[int] = (),
):
    """Randomize dynamics, including the actual foot-floor contact geoms.

    The upstream implementation hard-coded geom 0 as the floor.  In the Open
    Duck flat-terrain model the floor is geom 46 and the two sole collision
    geoms are 18 and 43.  Randomizing only geom 0 therefore did not vary the
    contact friction used by the policy.  The runner now supplies the resolved
    IDs and one sampled coefficient is applied to the floor and both soles.
    """

    dof_id = jp.array(
        [index for index, value in enumerate(model.dof_hasfrictionloss) if value]
    )
    jnt_id = model.dof_jntid[dof_id]
    dof_addr = jp.array([address for address in model.jnt_dofadr if address in dof_id])
    joint_addr = model.jnt_qposadr[jnt_id]
    contact_geom_ids = jp.asarray(
        [int(floor_geom_id), *(int(value) for value in foot_geom_ids)],
        dtype=jp.int32,
    )
    friction_min = float(os.environ.get("CONTACT_FRICTION_MIN", "0.3"))
    friction_max = float(os.environ.get("CONTACT_FRICTION_MAX", "1.4"))
    randomize_contact_friction = os.environ.get(
        "RANDOMIZE_CONTACT_FRICTION", "1"
    ) != "0"
    if not 0.0 < friction_min <= friction_max:
        raise ValueError("CONTACT_FRICTION_MIN/MAX must satisfy 0 < min <= max")

    @jax.vmap
    def rand_dynamics(one_rng):
        one_rng, key = jax.random.split(one_rng)
        contact_friction = jax.random.uniform(
            key, minval=friction_min, maxval=friction_max
        )
        geom_friction = (
            model.geom_friction.at[contact_geom_ids, 0].set(contact_friction)
            if randomize_contact_friction
            else model.geom_friction
        )

        one_rng, key = jax.random.split(one_rng)
        frictionloss = model.dof_frictionloss[dof_addr] * jax.random.uniform(
            key, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[dof_addr].set(frictionloss)

        one_rng, key = jax.random.split(one_rng)
        armature = model.dof_armature[dof_addr] * jax.random.uniform(
            key, shape=(model.nu,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[dof_addr].set(armature)

        one_rng, key = jax.random.split(one_rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
            model.body_ipos[TORSO_BODY_ID] + dpos
        )

        one_rng, key = jax.random.split(one_rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        one_rng, key = jax.random.split(one_rng)
        torso_mass = jax.random.uniform(key, minval=-0.1, maxval=0.1)
        body_mass = body_mass.at[TORSO_BODY_ID].set(
            body_mass[TORSO_BODY_ID] + torso_mass
        )

        one_rng, key = jax.random.split(one_rng)
        qpos0 = model.qpos0.at[joint_addr].set(
            model.qpos0[joint_addr]
            + jax.random.uniform(
                key, shape=(model.nu,), minval=-0.03, maxval=0.03
            )
        )

        one_rng, key = jax.random.split(one_rng)
        factor = jax.random.uniform(
            key, shape=(model.nu,), minval=0.9, maxval=1.1
        )
        current_kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(current_kp * factor)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-current_kp * factor)

        return (
            geom_friction,
            body_ipos,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm,
        )

    (
        friction,
        body_ipos,
        frictionloss,
        armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_ipos": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )
    model = model.tree_replace(
        {
            "geom_friction": friction,
            "body_ipos": body_ipos,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )
    return model, in_axes
