# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Headless regression test for the PD controller FMU simulation logic.

Tests the PD controller equations in pure Python (no ovphysx/ovrtx required):
  force = kp * (target - pos) + kd * (0 - vel)
  force_y += mass * (-gravity)   # gravity compensation

Simulates a rigid body under PD control with Euler integration to verify
convergence to the target position.
"""

import math

import pytest


def simulate_pd_controller(
    pos: list[float] | None = None,
    vel: list[float] | None = None,
    target: list[float] | None = None,
    kp: float = 50.0,
    kd: float = 10.0,
    mass: float = 1.0,
    gravity: float = -9.81,
    dt: float = 1e-3,
    duration: float = 5.0,
) -> list[dict]:
    """Simulate a body under PD control with gravity.

    Returns list of dicts with keys: time, pos, vel, force at each time step.
    Uses semi-implicit Euler integration.
    """
    if pos is None:
        pos = [0.0, 5.0, 0.0]
    if vel is None:
        vel = [0.0, 0.0, 0.0]
    if target is None:
        target = [0.0, 2.0, 0.0]

    pos = list(pos)
    vel = list(vel)
    results = []
    t = 0.0

    while t <= duration + 1e-12:
        # PD control force
        force = [0.0, 0.0, 0.0]
        for i in range(3):
            error = target[i] - pos[i]
            force[i] = kp * error + kd * (0.0 - vel[i])

        # Gravity compensation on Y axis
        force[1] += mass * (-gravity)

        # Record state
        results.append({
            "time": t,
            "pos": list(pos),
            "vel": list(vel),
            "force": list(force),
        })

        # Gravity force on body
        gravity_force = [0.0, gravity * mass, 0.0]

        # Net force = PD control + gravity
        net_force = [force[i] + gravity_force[i] for i in range(3)]

        # Semi-implicit Euler: update velocity first, then position
        accel = [net_force[i] / mass for i in range(3)]
        for i in range(3):
            vel[i] += accel[i] * dt
            pos[i] += vel[i] * dt

        t += dt

    return results


class TestPDControllerSimulation:
    """Test the PD controller logic in pure Python."""

    def test_initial_force_direction(self, pd_controller_defaults):
        """Starting at y=5.0 with target y=2.0, initial force_y should push down."""
        results = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            duration=0.01,
            **pd_controller_defaults,
        )
        # At t=0: error_y = 2.0 - 5.0 = -3.0
        # force_y = 50*(-3) + 10*(0-0) + 1*9.81 = -150 + 9.81 = -140.19
        first = results[0]
        assert first["force"][1] < 0, (
            f"Expected downward force at y=5.0, got force_y={first['force'][1]}"
        )
        # X and Z forces should be zero (pos == target)
        assert first["force"][0] == pytest.approx(0.0, abs=1e-10)
        assert first["force"][2] == pytest.approx(0.0, abs=1e-10)

    def test_gravity_compensation_at_target(self, pd_controller_defaults):
        """At the target position with zero velocity, force should exactly cancel gravity."""
        results = simulate_pd_controller(
            pos=[0.0, 2.0, 0.0],
            vel=[0.0, 0.0, 0.0],
            duration=0.001,
            **pd_controller_defaults,
        )
        first = results[0]
        # error = 0, vel = 0, so PD term = 0
        # force_y = 0 + mass*(-gravity) = 1*9.81 = 9.81
        expected_fy = pd_controller_defaults["mass"] * (-pd_controller_defaults["gravity"])
        assert first["force"][1] == pytest.approx(expected_fy, abs=1e-10)

    def test_convergence_to_target(self, pd_controller_defaults):
        """Body should converge to target_y=2.0 within 5 seconds."""
        results = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            duration=5.0,
            **pd_controller_defaults,
        )
        final = results[-1]
        assert final["pos"][1] == pytest.approx(2.0, abs=0.05), (
            f"Final pos_y={final['pos'][1]:.4f}, expected ~2.0"
        )
        # Velocity should be nearly zero at convergence
        assert abs(final["vel"][1]) < 0.1, (
            f"Final vel_y={final['vel'][1]:.4f}, expected ~0"
        )

    def test_convergence_from_below(self, pd_controller_defaults):
        """Body starting below target should rise to target."""
        results = simulate_pd_controller(
            pos=[0.0, 0.5, 0.0],
            duration=5.0,
            **pd_controller_defaults,
        )
        final = results[-1]
        assert final["pos"][1] == pytest.approx(2.0, abs=0.05), (
            f"Final pos_y={final['pos'][1]:.4f}, expected ~2.0"
        )

    def test_3d_convergence(self):
        """Body offset in all axes should converge to target."""
        results = simulate_pd_controller(
            pos=[3.0, 5.0, -2.0],
            target=[0.0, 2.0, 0.0],
            duration=5.0,
        )
        final = results[-1]
        assert final["pos"][0] == pytest.approx(0.0, abs=0.05)
        assert final["pos"][1] == pytest.approx(2.0, abs=0.05)
        assert final["pos"][2] == pytest.approx(0.0, abs=0.05)

    def test_force_bounded(self, pd_controller_defaults):
        """Forces should be bounded (not infinite or NaN)."""
        results = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            duration=2.0,
            **pd_controller_defaults,
        )
        for r in results:
            for i in range(3):
                assert math.isfinite(r["force"][i]), (
                    f"Non-finite force at t={r['time']}: {r['force']}"
                )
                # With default params and starting pos, forces should stay reasonable
                assert abs(r["force"][i]) < 1000.0, (
                    f"Excessive force at t={r['time']}: {r['force']}"
                )

    def test_different_kp_kd(self):
        """Higher Kp/Kd should converge faster."""
        # Low gains
        results_low = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            kp=20.0, kd=5.0,
            duration=5.0,
        )
        # High gains
        results_high = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            kp=100.0, kd=20.0,
            duration=5.0,
        )

        # Both should converge eventually
        assert results_low[-1]["pos"][1] == pytest.approx(2.0, abs=0.1)
        assert results_high[-1]["pos"][1] == pytest.approx(2.0, abs=0.05)

        # High gains should have less error at t=1.0 s
        at_1s_low = next(r for r in results_low if r["time"] >= 1.0)
        at_1s_high = next(r for r in results_high if r["time"] >= 1.0)
        err_low = abs(at_1s_low["pos"][1] - 2.0)
        err_high = abs(at_1s_high["pos"][1] - 2.0)
        assert err_high < err_low, (
            f"High-gain error {err_high:.4f} should be < low-gain error {err_low:.4f} at t=1s"
        )

    def test_underdamped_oscillation(self):
        """With low damping, system should oscillate around target."""
        results = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            kp=100.0, kd=2.0,  # very low damping
            duration=3.0,
        )
        # Find sign changes in (pos_y - target_y)
        crossings = 0
        prev_err = results[0]["pos"][1] - 2.0
        for r in results[1:]:
            err = r["pos"][1] - 2.0
            if prev_err * err < 0:
                crossings += 1
            prev_err = err

        assert crossings >= 2, (
            f"Expected oscillation (≥2 crossings), got {crossings}"
        )

    def test_heavy_mass(self):
        """Heavier mass should converge with same gains (gravity compensation)."""
        results = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            mass=10.0,
            kp=500.0,  # scale gains with mass for similar dynamics
            kd=100.0,
            duration=5.0,
        )
        final = results[-1]
        assert final["pos"][1] == pytest.approx(2.0, abs=0.1), (
            f"Heavy body: final pos_y={final['pos'][1]:.4f}"
        )

    def test_zero_gravity(self):
        """With zero gravity, no gravity compensation needed."""
        results = simulate_pd_controller(
            pos=[0.0, 5.0, 0.0],
            gravity=0.0,
            duration=3.0,
        )
        final = results[-1]
        # Should still converge to target
        assert final["pos"][1] == pytest.approx(2.0, abs=0.05)

    def test_stationary_at_target(self, pd_controller_defaults):
        """Body at rest at target should remain at target."""
        results = simulate_pd_controller(
            pos=[0.0, 2.0, 0.0],
            vel=[0.0, 0.0, 0.0],
            duration=2.0,
            **pd_controller_defaults,
        )
        for r in results:
            assert r["pos"][1] == pytest.approx(2.0, abs=0.001), (
                f"Drifted from target at t={r['time']}: pos_y={r['pos'][1]}"
            )

    def test_deterministic(self, pd_controller_defaults):
        """Two runs must produce identical results."""
        r1 = simulate_pd_controller(duration=1.0, **pd_controller_defaults)
        r2 = simulate_pd_controller(duration=1.0, **pd_controller_defaults)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["time"] == b["time"]
            for i in range(3):
                assert a["pos"][i] == b["pos"][i]
                assert a["vel"][i] == b["vel"][i]
                assert a["force"][i] == b["force"][i]
