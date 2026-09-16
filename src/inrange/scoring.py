"""Composite metric and cross-validation harness.

The competition score is a hidden weighted composite. This module is our own
approximation of it, not the real thing. Every report returns the composite
together with each component, so conclusions never rest on our weights alone.

Components (per shot):
    landing_pos  Euclidean distance between true and predicted landing (x, y, z), m
    apex_pos     Euclidean distance between true and predicted apex (x, y, z), m
    apex_t       |true - predicted| apex time, s
    landing_t    |true - predicted| landing time, s
    spin         |true - predicted| launch spin rate, rpm

Scaling. Each component is divided by REFERENCE_SCALES[component]: the mean
error of predicting the training-set mean for every shot, computed in-sample on
all 491 training rows (``compute_reference_scales``). For the positions that is
the mean Euclidean distance of each shot from the training centroid; for the
scalars it is the mean absolute deviation about the training mean. A scaled
component of 1.0 therefore means "no better than the training mean", and the
sample_submission.csv fallback scores exactly 1.0 on every component in-sample.
Mean-based scales are used rather than standard deviations because the
components are mean absolute errors, so like is compared with like.

Weights. The competition states the priority order landing position > apex
position > apex and landing times > spin. We choose

    landing_pos 0.40, apex_pos 0.25, apex_t 0.125, landing_t 0.125, spin 0.10

which sum to 1, respect that order strictly, give landing position the largest
single share, and split the time weight evenly because the two times share a
rank. The composite is the weighted sum of scaled mean errors (equivalently the
mean over shots of the weighted per-shot scaled error). These numbers are a
judgement call; the hidden metric may use different weights, scales, or even
squared errors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from inrange.features import SPEED_BAND_LABELS, session_labels, speed_band
from inrange.io import TARGET_COLS

COMPONENTS: list[str] = ["landing_pos", "apex_pos", "apex_t", "landing_t", "spin"]

COMPONENT_UNITS: dict[str, str] = {
    "landing_pos": "m", "apex_pos": "m", "apex_t": "s", "landing_t": "s", "spin": "rpm",
}

WEIGHTS: dict[str, float] = {
    "landing_pos": 0.40,
    "apex_pos": 0.25,
    "apex_t": 0.125,
    "landing_t": 0.125,
    "spin": 0.10,
}

# Output of compute_reference_scales(load_train()); tests/test_scoring.py checks
# the two agree. Units follow COMPONENT_UNITS.
REFERENCE_SCALES: dict[str, float] = {
    "landing_pos": 42.093899169512206,
    "apex_pos": 31.37859936008478,
    "apex_t": 0.4650738963253015,
    "landing_t": 0.8568472836930326,
    "spin": 2036.4877331989273,
}

_POSITION_COLS: dict[str, list[str]] = {
    "landing_pos": ["landing_x", "landing_y", "landing_z"],
    "apex_pos": ["apex_x", "apex_y", "apex_z"],
}
_SCALAR_COLS: dict[str, str] = {
    "apex_t": "apex_t", "landing_t": "landing_t", "spin": "launch_spin_rate",
}


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def per_shot_errors(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Unscaled error of every component for every shot.

    Both frames must contain TARGET_COLS and share the same index. Returns one
    row per shot and one column per component, in the units of COMPONENT_UNITS.
    """
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must share the same index")
    missing = [c for c in TARGET_COLS if c not in y_pred.columns]
    if missing:
        raise ValueError(f"y_pred is missing target columns: {missing}")

    errors = pd.DataFrame(index=y_true.index)
    for component, cols in _POSITION_COLS.items():
        diff = y_true[cols].to_numpy(dtype=float) - y_pred[cols].to_numpy(dtype=float)
        errors[component] = np.sqrt((diff ** 2).sum(axis=1))
    for component, col in _SCALAR_COLS.items():
        errors[component] = (y_true[col] - y_pred[col]).abs()
    return errors[COMPONENTS]


def summarise_errors(
    errors: pd.DataFrame,
    scales: Mapping[str, float] = REFERENCE_SCALES,
    weights: Mapping[str, float] = WEIGHTS,
) -> pd.Series:
    """Composite plus every component from a per-shot error table.

    Returns a Series with ``composite``, ``n`` (shots), then for each component
    ``{c}`` (mean error in its own units) and ``{c}_scaled`` (mean error divided
    by its reference scale). composite = sum over c of weight[c] * {c}_scaled.
    """
    out: dict[str, float] = {"composite": 0.0, "n": float(len(errors))}
    for component in COMPONENTS:
        mean_error = float(errors[component].mean())
        scaled = mean_error / scales[component]
        out[component] = mean_error
        out[f"{component}_scaled"] = scaled
        out["composite"] += weights[component] * scaled
    return pd.Series(out)


def score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    scales: Mapping[str, float] = REFERENCE_SCALES,
    weights: Mapping[str, float] = WEIGHTS,
) -> pd.Series:
    """Score predictions: composite plus every component (see summarise_errors)."""
    return summarise_errors(per_shot_errors(y_true, y_pred), scales, weights)


def compute_reference_scales(train: pd.DataFrame) -> dict[str, float]:
    """Mean error (component units) of predicting the training mean, in-sample."""
    means = train[TARGET_COLS].mean()
    mean_prediction = pd.DataFrame([means] * len(train), index=train.index)
    errors = per_shot_errors(train[TARGET_COLS], mean_prediction)
    return {component: float(errors[component].mean()) for component in COMPONENTS}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

Split = tuple[np.ndarray, np.ndarray]  # positional (train_idx, val_idx)


def session_grouped_splits(sessions: pd.Series, n_splits: int = 5) -> list[Split]:
    """GroupKFold on session id: every session is validated once, never trained on in that fold.

    Harder than the real test, where every test shot has same-session shots in train.
    """
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(sessions))
    return list(splitter.split(placeholder, groups=sessions.to_numpy()))


def session_holdout_fractions(train_sessions: pd.Series, test_sessions: pd.Series) -> dict[int, float]:
    """Per-session share of shots that the real split put in test: n_test / (n_train + n_test)."""
    n_train = train_sessions.value_counts()
    n_test = test_sessions.value_counts().reindex(n_train.index, fill_value=0)
    return (n_test / (n_train + n_test)).to_dict()


def within_session_splits(
    sessions: pd.Series,
    holdout_fractions: Mapping[int, float],
    n_repeats: int = 10,
    seed: int = 0,
) -> list[Split]:
    """Repeated random holdout inside each session, mimicking the real train/test split.

    In each repeat, every session holds out round(fraction * n_rows) of its
    training rows (at least one, and at least one kept for training), where
    fraction is that session's real test share. One split per repeat; a shot
    can be validated in several repeats.
    """
    rng = np.random.default_rng(seed)
    positions = pd.Series(np.arange(len(sessions)), index=sessions.index)
    splits: list[Split] = []
    for _ in range(n_repeats):
        val_parts = []
        for session, rows in positions.groupby(sessions.to_numpy()):
            n_rows = len(rows)
            n_val = int(round(holdout_fractions[session] * n_rows))
            n_val = min(max(n_val, 1), n_rows - 1)
            val_parts.append(rng.choice(rows.to_numpy(), size=n_val, replace=False))
        val_idx = np.sort(np.concatenate(val_parts))
        train_idx = np.setdiff1d(np.arange(len(sessions)), val_idx)
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

# fit_predict(train_rows, val_rows) -> predictions for val_rows with TARGET_COLS,
# indexed like val_rows. Must only use targets from train_rows.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


@dataclass
class CVResult:
    """Per-shot validation errors from one splitting strategy, with summaries.

    ``errors`` has one row per (fold, validated shot): columns fold, session,
    speed_band, then every component in its own units.
    """

    strategy: str
    errors: pd.DataFrame
    test_band_shares: dict[str, float]

    def overall(self) -> pd.Series:
        """Composite and components pooled over every validated shot."""
        return summarise_errors(self.errors)

    def by_fold(self) -> pd.DataFrame:
        """One summary row per fold."""
        return self.errors.groupby("fold").apply(summarise_errors, include_groups=False)

    def by_speed_band(self) -> pd.DataFrame:
        """One summary row per ball speed band, in SPEED_BAND_LABELS order."""
        table = self.errors.groupby("speed_band").apply(summarise_errors, include_groups=False)
        return table.reindex([b for b in SPEED_BAND_LABELS if b in table.index])

    def test_mix(self) -> pd.Series:
        """Band summaries reweighted to the test set's share of shots per speed band.

        Corrects for train having fewer >=70 m/s shots than test. Raw mean errors
        and scaled components are reweighted the same way; n is the pooled count.
        """
        bands = self.by_speed_band()
        shares = pd.Series(self.test_band_shares).reindex(bands.index)
        shares = shares / shares.sum()
        mixed = bands.drop(columns="n").mul(shares, axis=0).sum()
        mixed["n"] = bands["n"].sum()
        return mixed[bands.columns]

    def report(self) -> pd.DataFrame:
        """Overall, test-mix and per-band rows in one table."""
        rows = {"overall": self.overall(), "test speed mix": self.test_mix()}
        for band, row in self.by_speed_band().iterrows():
            rows[f"band {band} m/s"] = row
        table = pd.DataFrame(rows).T
        table.insert(0, "strategy", self.strategy)
        return table


def cross_validate(
    fit_predict: FitPredict,
    train: pd.DataFrame,
    splits: list[Split],
    sessions: pd.Series,
    test_band_shares: Mapping[str, float],
    strategy: str,
) -> CVResult:
    """Run fit_predict on every split and collect per-shot validation errors."""
    bands = speed_band(train)
    parts = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_rows = train.iloc[train_idx]
        val_rows = train.iloc[val_idx]
        val_inputs = val_rows.drop(columns=TARGET_COLS)
        predictions = fit_predict(train_rows, val_inputs)
        errors = per_shot_errors(val_rows[TARGET_COLS], predictions.loc[val_rows.index])
        errors.insert(0, "speed_band", bands.iloc[val_idx].to_numpy())
        errors.insert(0, "session", sessions.iloc[val_idx].to_numpy())
        errors.insert(0, "fold", fold)
        parts.append(errors)
    return CVResult(strategy, pd.concat(parts), dict(test_band_shares))


def evaluate(
    fit_predict: FitPredict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_group_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, CVResult]:
    """Cross-validate under both strategies. Always report both.

    Returns {"session_grouped": ..., "within_session": ...}. ``test`` supplies
    only inputs: session ids, per-session holdout fractions and speed band shares.
    """
    train_sessions, test_sessions = session_labels(train, test)
    test_band_shares = speed_band(test).value_counts(normalize=True).to_dict()

    grouped = session_grouped_splits(train_sessions, n_group_splits)
    fractions = session_holdout_fractions(train_sessions, test_sessions)
    within = within_session_splits(train_sessions, fractions, n_repeats, seed)

    return {
        "session_grouped": cross_validate(fit_predict, train, grouped, train_sessions,
                                          test_band_shares, "session_grouped"),
        "within_session": cross_validate(fit_predict, train, within, train_sessions,
                                         test_band_shares, "within_session"),
    }


def report_table(results: Mapping[str, CVResult]) -> pd.DataFrame:
    """Stack the report() of each strategy into one table."""
    return pd.concat([result.report() for result in results.values()])

