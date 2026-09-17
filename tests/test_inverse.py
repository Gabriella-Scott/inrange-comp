"""Checks for the checkpoint-only inverse solve and the variant selection rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inrange.calibration import ShotBatch
from inrange.hybrid import COMPONENT_TARGETS, combine, select_per_component
from inrange.inverse import solve_states
from inrange.models import SHOT_FRAME_TARGETS
from inrange.physics import Aero, simulate

AERO = Aero(0.19, 0.33, 0.44, 0.39, 25.0)
SCALES = {"checkpoints": 0.4, "apex_pos": 1.0, "apex_t": 0.1, "landing_pos": 2.5, "landing_t": 0.1}


def _synthetic_batch(true_spin, true_tilt, true_speed) -> ShotBatch:
    vel0 = np.array([[55.0, -2.0, 13.0], [45.0, 1.0, 15.0], [68.0, 0.0, 10.0]])
    cp_t = np.array([[0.3, 0.65, 1.0, 1.4], [0.35, 0.75, 1.2, 1.7], [0.25, 0.5, 0.8, 1.1]])
    truth = simulate(vel0 * true_speed[:, None], true_spin, true_tilt, AERO, cp_t)
    n = len(vel0)
    nan = np.full(n, np.nan)
    return ShotBatch(pd.RangeIndex(n), vel0, np.full(n, 5000.0), cp_t, truth.query_pos,
                     nan, np.full((n, 3), np.nan), nan, np.full((n, 3), np.nan))


def test_inverse_recovers_noise_free_states() -> None:
    spin = np.array([4000.0, 8500.0, 2800.0])
    tilt = np.radians([-8.0, 12.0, 3.0])
    speed = np.array([0.96, 0.99, 1.03])
    result = solve_states(_synthetic_batch(spin, tilt, speed), AERO, SCALES, max_nfev=200)
    states = result.states
    assert states["spin"].to_numpy() == pytest.approx(spin, rel=0.02)
    assert np.degrees(states["tilt"].to_numpy()) == pytest.approx(np.degrees(tilt), abs=0.5)
    assert states["speed"].to_numpy() == pytest.approx(speed, abs=0.002)
    assert states["cp_rms"].max() < 0.01


def test_spin_prior_pulls_towards_prior() -> None:
    spin = np.array([4000.0, 8500.0, 2800.0])
    batch = _synthetic_batch(spin, np.zeros(3), np.ones(3))
    prior = spin + 3000.0
    weak = solve_states(batch, AERO, SCALES, prior, prior_sd=1e6).states["spin"]
    strong = solve_states(batch, AERO, SCALES, prior, prior_sd=10.0).states["spin"]
    assert np.abs(weak.to_numpy() - spin).max() < np.abs(strong.to_numpy() - spin).min()
    assert strong.to_numpy() == pytest.approx(prior, abs=100.0)


def _table(entries: dict) -> pd.DataFrame:
    rows = []
    for (component, variant, strategy), (mean, se) in entries.items():
        rows.append({"column": component, "variant": variant, "strategy": strategy, "mean_diff": mean, "se_diff": se})
    return pd.DataFrame(rows).set_index(["column", "variant", "strategy"])


def test_selection_requires_improvement_beyond_se_in_both_strategies() -> None:
    entries = {}
    for component in COMPONENT_TARGETS:
        entries[(component, "a", "leave_one_session_out")] = (0.01, 0.01)
        entries[(component, "a", "within_session")] = (0.01, 0.01)
        entries[(component, "c", "leave_one_session_out")] = (0.5, 0.01)
        entries[(component, "c", "within_session")] = (0.01, 0.01)
    entries[("landing_pos", "a", "leave_one_session_out")] = (-0.5, 0.1)
    entries[("landing_pos", "a", "within_session")] = (-0.3, 0.1)
    entries[("apex_t", "a", "leave_one_session_out")] = (-0.5, 0.1)
    entries[("apex_t", "a", "within_session")] = (-0.05, 0.1)   # within SE: not enough
    entries[("spin", "c", "within_session")] = (-0.3, 0.1)      # c judged within-session only
    choice, _ = select_per_component(_table(entries))
    assert choice == {"landing_pos": "a", "apex_pos": "lgbm", "apex_t": "lgbm", "landing_t": "lgbm", "spin": "c"}


def test_combine_takes_each_component_from_its_variant() -> None:
    index = pd.RangeIndex(2)
    preds = {name: pd.DataFrame(value, index=index, columns=SHOT_FRAME_TARGETS)
             for name, value in [("lgbm", 1.0), ("a", 2.0)]}
    choice = {c: "lgbm" for c in COMPONENT_TARGETS} | {"apex_pos": "a"}
    combined = combine(preds, choice)
    assert list(combined.columns) == SHOT_FRAME_TARGETS
    assert (combined[["apex_d", "apex_l", "apex_h"]] == 2.0).all().all()
    assert (combined.drop(columns=["apex_d", "apex_l", "apex_h"]) == 1.0).all().all()
