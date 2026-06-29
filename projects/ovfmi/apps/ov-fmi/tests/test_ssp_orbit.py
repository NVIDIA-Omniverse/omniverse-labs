"""Regression tests for the SSP orbit demo (trajectory generator + PD controller)."""

import math
import os
import sys

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory generator FMU — pure Python reference
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryGeneratorRef:
    """Pure Python reference for the trajectory generator FMU."""
    def __init__(self, radius=1.5, height=2.0, omega=1.0, phase=0.0,
                 center_x=0.0, center_z=0.0):
        self.radius = radius
        self.height = height
        self.omega = omega
        self.phase = phase
        self.center_x = center_x
        self.center_z = center_z

    def compute(self, t):
        angle = self.omega * t + self.phase
        return (
            self.center_x + self.radius * math.cos(angle),
            self.height,
            self.center_z + self.radius * math.sin(angle),
        )


class PDControllerRef:
    """Pure Python reference for the PD controller FMU."""
    def __init__(self, kp=50.0, kd=10.0, mass=1.0, gravity=-9.81):
        self.kp = kp
        self.kd = kd
        self.mass = mass
        self.gravity = gravity

    def compute(self, pos, vel, target):
        error = np.array(target) - np.array(pos)
        force = self.kp * error + self.kd * (np.zeros(3) - np.array(vel))
        force[1] += self.mass * (-self.gravity)
        return force


# ──────────────────────────────────────────────────────────────────────────────
# Tests: trajectory generator
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectoryGenerator:
    def test_initial_position(self):
        """At t=0 with default params, target should be (radius, height, 0)."""
        tg = TrajectoryGeneratorRef()
        x, y, z = tg.compute(0.0)
        assert abs(x - 1.5) < 1e-10
        assert abs(y - 2.0) < 1e-10
        assert abs(z - 0.0) < 1e-10

    def test_quarter_orbit(self):
        """At t=π/2, target should be (0, height, radius)."""
        tg = TrajectoryGeneratorRef()
        x, y, z = tg.compute(math.pi / 2)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 2.0) < 1e-10
        assert abs(z - 1.5) < 1e-10

    def test_half_orbit(self):
        """At t=π, target should be (-radius, height, 0)."""
        tg = TrajectoryGeneratorRef()
        x, y, z = tg.compute(math.pi)
        assert abs(x - (-1.5)) < 1e-10
        assert abs(y - 2.0) < 1e-10
        assert abs(z - 0.0) < 1e-6

    def test_full_orbit(self):
        """At t=2π, target should be back at start."""
        tg = TrajectoryGeneratorRef()
        x, y, z = tg.compute(2 * math.pi)
        assert abs(x - 1.5) < 1e-10
        assert abs(y - 2.0) < 1e-10
        assert abs(z - 0.0) < 1e-6

    def test_radius_on_circle(self):
        """Distance from orbit center to target should always equal radius."""
        tg = TrajectoryGeneratorRef()
        for t in np.linspace(0, 10, 100):
            x, y, z = tg.compute(t)
            dist = math.sqrt((x - tg.center_x)**2 + (z - tg.center_z)**2)
            assert abs(dist - tg.radius) < 1e-10

    def test_custom_params(self):
        """Custom radius, height, omega, phase, center."""
        tg = TrajectoryGeneratorRef(radius=3.0, height=5.0, omega=2.0,
                                    phase=math.pi/4, center_x=1.0, center_z=-1.0)
        x, y, z = tg.compute(0.0)
        expected_x = 1.0 + 3.0 * math.cos(math.pi / 4)
        expected_z = -1.0 + 3.0 * math.sin(math.pi / 4)
        assert abs(x - expected_x) < 1e-10
        assert abs(y - 5.0) < 1e-10
        assert abs(z - expected_z) < 1e-10

    def test_height_constant(self):
        """Y output should always equal height regardless of time."""
        tg = TrajectoryGeneratorRef(height=7.5)
        for t in np.linspace(0, 20, 200):
            _, y, _ = tg.compute(t)
            assert abs(y - 7.5) < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Tests: PD controller
# ──────────────────────────────────────────────────────────────────────────────

class TestPDController:
    def test_zero_error_gravity_comp(self):
        """At target with zero velocity, force should be pure gravity compensation."""
        pd = PDControllerRef()
        force = pd.compute([0, 2, 0], [0, 0, 0], [0, 2, 0])
        assert abs(force[0]) < 1e-10
        assert abs(force[1] - 9.81) < 1e-10  # gravity compensation = mass * |g|
        assert abs(force[2]) < 1e-10

    def test_positive_x_error(self):
        """Target to the right of body → positive x force."""
        pd = PDControllerRef()
        force = pd.compute([0, 2, 0], [0, 0, 0], [1, 2, 0])
        assert force[0] > 0
        assert abs(force[0] - 50.0) < 1e-10  # kp * 1.0

    def test_damping(self):
        """Positive velocity → negative damping force."""
        pd = PDControllerRef()
        force = pd.compute([0, 2, 0], [1, 0, 0], [0, 2, 0])
        assert force[0] < 0
        assert abs(force[0] - (-10.0)) < 1e-10  # kd * (-1.0)

    def test_combined(self):
        """Combined proportional + derivative + gravity."""
        pd = PDControllerRef(kp=100, kd=20, mass=2.0, gravity=-9.81)
        force = pd.compute([1, 0, 0], [0.5, 0, 0], [3, 4, 1])
        # x: kp*(3-1) + kd*(0-0.5) = 200 - 10 = 190
        assert abs(force[0] - 190.0) < 1e-10
        # y: kp*(4-0) + kd*(0-0) + mass*|g| = 400 + 0 + 19.62 = 419.62
        assert abs(force[1] - 419.62) < 1e-10
        # z: kp*(1-0) + kd*(0-0) = 100
        assert abs(force[2] - 100.0) < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Tests: integrated orbit tracking
# ──────────────────────────────────────────────────────────────────────────────

class TestOrbitTracking:
    def _simulate(self, duration=10.0, dt=1.0/60.0, **tg_kwargs):
        """Run trajectory + PD controller + physics for given duration."""
        tg = TrajectoryGeneratorRef(**tg_kwargs)
        pd = PDControllerRef()

        # Start at trajectory initial position
        x0, y0, z0 = tg.compute(0)
        pos = np.array([x0, y0, z0])
        vel = np.array([0.0, 0.0, 0.0])

        steps = int(duration / dt)
        positions = []
        targets = []

        for step in range(steps):
            t = step * dt
            target = np.array(tg.compute(t))
            force = pd.compute(pos, vel, target)

            acc = force / pd.mass + np.array([0.0, pd.gravity, 0.0])
            vel += acc * dt
            pos = pos + vel * dt

            positions.append(pos.copy())
            targets.append(target.copy())

        return np.array(positions), np.array(targets)

    def test_converges_to_orbit(self):
        """After transient, tracking error should settle below 0.5m."""
        positions, targets = self._simulate(duration=10.0)
        # Skip first 2 seconds of transient
        steady_state = positions[120:]
        steady_targets = targets[120:]
        errors = np.linalg.norm(steady_state - steady_targets, axis=1)
        assert errors.max() < 0.5, f"Max tracking error {errors.max():.3f}m"

    def test_steady_state_error_bounded(self):
        """Steady-state error should be approximately constant (stable)."""
        positions, targets = self._simulate(duration=15.0)
        # Use last 10 seconds
        late_errors = np.linalg.norm(positions[300:] - targets[300:], axis=1)
        # Error should be roughly constant (std < 0.05m)
        assert late_errors.std() < 0.05, f"Error std {late_errors.std():.3f}m"

    def test_height_maintained(self):
        """Ball should maintain orbit height after settling."""
        positions, _ = self._simulate(duration=10.0)
        # After 2s transient
        heights = positions[120:, 1]
        assert abs(heights.mean() - 2.0) < 0.05
        assert heights.std() < 0.02

    def test_orbit_radius_tracked(self):
        """Ball XZ distance from center should approximate orbit radius."""
        positions, _ = self._simulate(duration=10.0)
        # After 2s transient
        xz_dist = np.sqrt(positions[120:, 0]**2 + positions[120:, 2]**2)
        assert abs(xz_dist.mean() - 1.5) < 0.3

    def test_different_omega(self):
        """Faster angular velocity still tracks (with more lag)."""
        positions, targets = self._simulate(duration=10.0, omega=2.0)
        late_errors = np.linalg.norm(positions[300:] - targets[300:], axis=1)
        # Higher omega → more phase lag but still bounded
        assert late_errors.max() < 1.0

    def test_different_radius(self):
        """Larger orbit radius still tracks."""
        positions, targets = self._simulate(duration=10.0, radius=3.0)
        late_errors = np.linalg.norm(positions[300:] - targets[300:], axis=1)
        assert late_errors.max() < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: FMU binary validation (fmpy)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectoryFMUBinary:
    """Tests that the compiled trajectory generator FMU works via fmpy."""

    @pytest.fixture
    def fmu_path(self):
        return os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "usd", "ov-fmi", "TrajectoryGenerator.fmu3"
        )

    def test_model_description_valid(self, fmu_path):
        import fmpy
        md = fmpy.read_model_description(fmu_path)
        assert md.modelName == "TrajectoryGenerator"
        assert md.coSimulation is not None
        assert md.coSimulation.modelIdentifier == "TrajectoryGenerator"

    def test_simulate_one_orbit(self, fmu_path):
        import fmpy
        result = fmpy.simulate_fmu(
            fmu_path,
            start_time=0.0,
            stop_time=2 * math.pi,  # one full orbit at omega=1
            step_size=0.01,
            output=["target_x", "target_y", "target_z"],
            fmi_type="CoSimulation",
        )
        # First point: (1.5, 2.0, 0.0)
        assert abs(result[0]["target_x"] - 1.5) < 0.01
        assert abs(result[0]["target_y"] - 2.0) < 0.01
        assert abs(result[0]["target_z"] - 0.0) < 0.01
        # All points should be at radius distance from center in XZ
        for row in result:
            dist = math.sqrt(row["target_x"]**2 + row["target_z"]**2)
            assert abs(dist - 1.5) < 0.01

    def test_outputs_count(self, fmu_path):
        import fmpy
        md = fmpy.read_model_description(fmu_path)
        outputs = [v for v in md.modelVariables if v.causality == "output"]
        assert len(outputs) == 3
        output_names = {v.name for v in outputs}
        assert output_names == {"target_x", "target_y", "target_z"}

    def test_parameters_count(self, fmu_path):
        import fmpy
        md = fmpy.read_model_description(fmu_path)
        params = [v for v in md.modelVariables if v.causality == "parameter"]
        assert len(params) == 6
        param_names = {v.name for v in params}
        assert param_names == {"radius", "height", "omega", "phase", "center_x", "center_z"}
