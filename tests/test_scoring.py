"""Checks for the composite metric and CV splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inrange.io import RAW_DIR, TARGET_COLS, load_train
from inrange.scoring import (
    COMPONENTS,
    REFERENCE_SCALES,
    WEIGHTS,
    compute_reference_scales,
    cross_validate,
    per_shot_errors,
    score,
    session_grouped_splits,
    within_session_splits,
)

needs_data = pytest.mark.skipif(not (RAW_DIR / "train.csv").exists(), reason="competition data not present")


def _targets(**overrides: float) -> pd.DataFrame:
    row = {col: 0.0 for col in TARGET_COLS}
    row.update(overrides)
    return pd.DataFrame([row])


def test_perfect_prediction_scores_zero() -> None:
    truth = _targets(landing_x=150.0, apex_z=25.0, launch_spin_rate=5000.0)
    result = score(truth, truth.copy())
    assert result["composite"] == 0.0
    for component in COMPONENTS:
        assert result[component] == 0.0


def test_component_errors_use_correct_geometry_and_units() -> None:
    truth = _targets()
    pred = _targets(landing_x=3.0, landing_y=4.0, apex_z=-2.0, apex_t=0.5,
                    landing_t=-0.25, launch_spin_rate=100.0)
    errors = per_shot_errors(truth, pred).iloc[0]
    assert errors["landing_pos"] == pytest.approx(5.0)
    assert errors["apex_pos"] == pytest.approx(2.0)
    assert errors["apex_t"] == pytest.approx(0.5)
    assert errors["landing_t"] == pytest.approx(0.25)
    assert errors["spin"] == pytest.approx(100.0)


def test_composite_is_weighted_sum_of_scaled_components() -> None:
    truth = _targets()
    pred = _targets(landing_x=10.0, apex_y=5.0, apex_t=0.1, landing_t=0.2, launch_spin_rate=300.0)
    result = score(truth, pred)
    expected = sum(WEIGHTS[c] * result[c] / REFERENCE_SCALES[c] for c in COMPONENTS)
    assert result["composite"] == pytest.approx(expected)


def test_weights_follow_stated_priority() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert WEIGHTS["landing_pos"] > WEIGHTS["apex_pos"] > WEIGHTS["apex_t"] > WEIGHTS["spin"]
    assert WEIGHTS["apex_t"] == WEIGHTS["landing_t"]


def test_misaligned_index_is_rejected() -> None:
    truth = _targets()
    pred = _targets()
    pred.index = [5]
    with pytest.raises(ValueError):
        per_shot_errors(truth, pred)


@needs_data
def test_reference_scales_match_training_data() -> None:
    computed = compute_reference_scales(load_train())
    for component in COMPONENTS:
        assert computed[component] == pytest.approx(REFERENCE_SCALES[component], rel=1e-12)


@needs_data
def test_in_sample_mean_prediction_scores_one_on_every_component() -> None:
    train = load_train()
    mean_prediction = pd.DataFrame([train[TARGET_COLS].mean()] * len(train), index=train.index)
    result = score(train[TARGET_COLS], mean_prediction)
    for component in COMPONENTS:
        assert result[f"{component}_scaled"] == pytest.approx(1.0)
    assert result["composite"] == pytest.approx(1.0)


def test_grouped_splits_never_share_a_session() -> None:
    sessions = pd.Series(np.repeat(np.arange(11), 7))
    splits = session_grouped_splits(sessions, n_splits=5)
    validated = np.concatenate([val for _, val in splits])
    assert np.array_equal(np.sort(validated), np.arange(len(sessions)))
    for train_idx, val_idx in splits:
        assert set(sessions.iloc[train_idx]).isdisjoint(sessions.iloc[val_idx])


def test_within_session_splits_match_holdout_fractions() -> None:
    sessions = pd.Series(np.repeat([0, 1, 2], [12, 40, 60]))
    fractions = {0: 0.62, 1: 0.5, 2: 0.38}
    for train_idx, val_idx in within_session_splits(sessions, fractions, n_repeats=3, seed=1):
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(sessions)
        val_counts = sessions.iloc[val_idx].value_counts()
        assert val_counts[0] == 7 and val_counts[1] == 20 and val_counts[2] == 23


@needs_data
def test_cross_validate_hides_validation_targets() -> None:
    train = load_train()
    sessions = pd.Series(np.arange(len(train)) % 3, index=train.index)
    splits = session_grouped_splits(sessions, n_splits=3)

    def fit_predict(train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> pd.DataFrame:
        assert not set(TARGET_COLS) & set(val_rows.columns)
        return pd.DataFrame([train_rows[TARGET_COLS].mean()] * len(val_rows), index=val_rows.index)

    result = cross_validate(fit_predict, train, splits, sessions, {"<50": 0.3, "50-70": 0.5, ">=70": 0.2}, "t")
    assert len(result.errors) == len(train)
    assert result.overall()["n"] == len(train)
    assert result.by_fold()["n"].sum() == len(train)
