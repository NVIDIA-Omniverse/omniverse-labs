# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed SI-unit mechanical-energy arithmetic; no solver correctness verdict."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MechanicalEnergy:
    components: tuple
    absolute_potential: float
    reference_potential: float

    @property
    def total(self):
        return sum(self.components)

    @property
    def roundoff(self):
        return 64 * 2**-23 * (abs(self.components[0]) + abs(self.components[1])
                              + abs(self.absolute_potential) + abs(self.reference_potential))


def _quaternion(value):
    if len(value) != 4 or not all(math.isfinite(v) for v in value):
        raise ValueError("Invalid captured quaternion")
    length = math.hypot(*value)
    if not math.isclose(length, 1, rel_tol=0, abs_tol=1e-4):
        raise ValueError("Captured quaternion is not normalized")
    return tuple(v / length for v in value)


def _rotate(q, v):
    x, y, z, w = q
    tx, ty, tz = 2*(y*v[2]-z*v[1]), 2*(z*v[0]-x*v[2]), 2*(x*v[1]-y*v[0])
    return (v[0] + w*tx + y*tz-z*ty, v[1] + w*ty + z*tx-x*tz, v[2] + w*tz + x*ty-y*tx)


def _multiply(a, b):
    x, y, z, w = a;u, v, t, s = b
    return (w*u+x*s+y*t-z*v, w*v-x*t+y*s+z*u, w*t+x*v-y*u+z*s, w*s-x*u-y*v-z*t)


def mechanical_energy(mass, inertia, com_pose, global_pose, velocity, angular, gravity, reference):
    if (not math.isfinite(mass) or mass <= 0 or len(inertia) != 3 or any(v < 0 for v in inertia)
            or len(com_pose) != 7 or len(global_pose) != 7
            or any(len(v) != 3 for v in (velocity, angular, gravity, reference))
            or not all(math.isfinite(v) for vector in (inertia, com_pose, global_pose, velocity, angular, gravity, reference) for v in vector)):
        raise ValueError("Missing or invalid captured physical setup")
    actor_q, local_q = _quaternion(global_pose[:4]), _quaternion(com_pose[:4])
    offset = _rotate(actor_q, com_pose[4:])
    world_com = tuple(global_pose[4+i] + offset[i] for i in range(3))
    mass_q = _multiply(actor_q, local_q)
    local_omega = _rotate((-mass_q[0], -mass_q[1], -mass_q[2], mass_q[3]), angular)
    translation = .5 * mass * sum(v*v for v in velocity)
    rotation = .5 * sum(i*w*w for i, w in zip(inertia, local_omega))
    absolute = -mass * sum(g*x for g, x in zip(gravity, world_com))
    ref_potential = -mass * sum(g*x for g, x in zip(gravity, reference))
    components = (translation, rotation, absolute-ref_potential)
    if not all(math.isfinite(v) for v in (*components, absolute, ref_potential)):
        raise ValueError("Mechanical energy is nonfinite")
    return MechanicalEnergy(components, absolute, ref_potential)
