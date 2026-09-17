"""Checks for units, the spin axis convention and integrator accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from inrange.physics import (
    DEFAULT_DT,
    TEXTBOOK,
    Aero,
    rpm_to_rad_s,
    simulate,
    simulate_reference,
    spin_axis,
)

# Five representative launches (m/s, shot frame d, l, h) and spins (rpm).
VEL0 = np.array([
    [60.0, 0.0, 14.0],
    [45.0, -3.0, 12.0],
    [72.0, 2.0, 10.0],
    [35.0, 1.0, 16.0],
    [55.0, -6.0, 8.0],
])
SPIN = np.array([4000.0, 7000.0, 2500.0, 9500.0, 5500.0])
TILT = np.radians([0.0, -12.0, 8.0, 25.0, -30.0])
QUERY = np.array([[0.3, 0.7, 1.1, 1.6]] * len(VEL0))


def test_rpm_conversion() -> None:
    assert rpm_to_rad_s(60.0) == pytest.approx(2 * np.pi)
    assert rpm_to_rad_s(3000.0) == pytest.approx(314.159, rel=1e-5)


def test_zero_tilt_is_pure_backspin() -> None:
    axis = spin_axis(np.array([0.0]))[0]
    lift_direction = np.cross(axis, [1.0, 0.0, 0.0])
    assert lift_direction == pytest.approx([0.0, 0.0, 1.0])


def test_positive_tilt_curves_left() -> None:
    vel0 = np.array([[55.0, 0.0, 12.0]] * 3)
    spin = np.full(3, 6000.0)
    result = simulate(vel0, spin, np.radians([-15.0, 0.0, 15.0]), TEXTBOOK)
    lateral = result.landing_pos[:, 1]
    assert lateral[0] < -1.0
    assert abs(lateral[1]) < 1e-9
    assert lateral[2] > 1.0
    assert lateral[2] == pytest.approx(-lateral[0])


def test_backspin_increases_carry_and_height() -> None:
    vel0 = np.array([[55.0, 0.0, 12.0]] * 2)
    result = simulate(vel0, np.array([0.0, 6000.0]), np.zeros(2), TEXTBOOK)
    assert result.apex_pos[1, 2] > result.apex_pos[0, 2]
    assert result.landing_pos[1, 0] > result.landing_pos[0, 0]


@pytest.mark.parametrize("aero", [TEXTBOOK, Aero(0.35, 0.8, 1.2, 0.2, 3.0)])
def test_rk4_agrees_with_solve_ivp(aero: Aero) -> None:
    batch = simulate(VEL0, SPIN, TILT, aero, QUERY, dt=DEFAULT_DT)
    for i in range(len(VEL0)):
        ref = simulate_reference(VEL0[i], SPIN[i], TILT[i], aero, QUERY[i])
        assert batch.apex_t[i] == pytest.approx(ref["apex_t"], abs=1e-3)
        assert batch.landing_t[i] == pytest.approx(ref["landing_t"], abs=1e-3)
        assert np.abs(batch.apex_pos[i] - ref["apex_pos"]).max() < 0.01
        assert np.abs(batch.landing_pos[i] - ref["landing_pos"]).max() < 0.01
        assert np.abs(batch.query_pos[i] - ref["query_pos"]).max() < 0.01


def test_landing_is_at_launch_height_and_after_apex() -> None:
    result = simulate(VEL0, SPIN, TILT, TEXTBOOK)
    assert np.abs(result.landing_pos[:, 2]).max() < 1e-6
    assert np.all(result.landing_t > result.apex_t)
    assert np.all(result.landing_vel[:, 2] < 0)
