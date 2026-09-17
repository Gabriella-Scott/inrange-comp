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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

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

# Composite without the spin term (not renormalised, so it equals the full
# composite minus its spin contribution). Used when spin is supplied as an
# input, e.g. the oracle-spin physics model.
WEIGHTS_WITHOUT_SPIN: dict[str, float] = {**WEIGHTS, "spin": 0.0}

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


def leave_one_session_out_splits(sessions: pd.Series) -> list[Split]:
    """One fold per session: that session is validated, every other session trains.

    Harder than the real test, where every test shot has same-session shots in
    train. Folds are ordered by session id.
    """
    placeholder = np.zeros(len(sessions))
    return list(LeaveOneGroupOut().split(placeholder, groups=sessions.to_numpy()))


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

    ``errors`` has one row per (fold, validated shot): columns fold, shot
    (index label in the training frame), session, speed_band, then every
    component in its own units.
    """

    strategy: str
    errors: pd.DataFrame
    test_band_shares: dict[str, float]
    weights: dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))

    def overall(self) -> pd.Series:
        """Composite and components pooled over every validated shot."""
        return summarise_errors(self.errors, weights=self.weights)

    def by_fold(self) -> pd.DataFrame:
        """One summary row per fold."""
        return self.errors.groupby("fold").apply(summarise_errors, weights=self.weights, include_groups=False)

    def by_session(self) -> pd.DataFrame:
        """One summary row per session (pooled over folds or repeats)."""
        return self.errors.groupby("session").apply(summarise_errors, weights=self.weights, include_groups=False)

    def by_speed_band(self) -> pd.DataFrame:
        """One summary row per ball speed band, in SPEED_BAND_LABELS order."""
        table = self.errors.groupby("speed_band").apply(summarise_errors, weights=self.weights, include_groups=False)
        return table.reindex([b for b in SPEED_BAND_LABELS if b in table.index])

    def test_mix(self) -> pd.Series:
        """Band summaries reweighted to the test set's share of shots per speed band.

        Corrects for train having fewer >=70 m/s shots than test. Raw mean errors
        and scaled components are reweighted the same way; n is the pooled count.
        """
        return _mix_bands(self.errors, self.test_band_shares, self.weights)

    def test_mix_spread(self, n_boot: int = 2000, seed: int = 0) -> pd.Series:
        """Uncertainty of the test-mix composite, estimated two ways.

        fold_sd: standard deviation of the test-mix composite computed per fold
            (or repeat), using only folds that contain every speed band.
            Leave-one-session-out folds often lack a band, so few may qualify.
        boot_sd, boot_p05, boot_p95: bootstrap over shots. Each shot's errors
            are first averaged over the folds it was validated in, then shots
            are resampled with replacement within their speed band.
        """
        per_fold = []
        for _, fold_errors in self.errors.groupby("fold"):
            if set(fold_errors["speed_band"]) >= set(SPEED_BAND_LABELS):
                per_fold.append(_mix_bands(fold_errors, self.test_band_shares, self.weights)["composite"])

        per_shot = self.errors.groupby("shot").agg(
            {"speed_band": "first", **{c: "mean" for c in COMPONENTS}})
        rng = np.random.default_rng(seed)
        weights = np.array([self.weights[c] / REFERENCE_SCALES[c] for c in COMPONENTS])
        shares = _band_shares(self.test_band_shares, per_shot["speed_band"])
        boot = np.zeros(n_boot)
        for band, share in shares.items():
            values = per_shot.loc[per_shot["speed_band"] == band, COMPONENTS].to_numpy()
            picks = rng.integers(0, len(values), size=(n_boot, len(values)))
            band_means = values[picks].mean(axis=1)  # (n_boot, n_components)
            boot += share * band_means @ weights

        return pd.Series({
            "composite": self.test_mix()["composite"],
            "fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else np.nan,
            "folds_used": float(len(per_fold)),
            "folds_total": float(self.errors["fold"].nunique()),
            "boot_sd": float(boot.std(ddof=1)),
            "boot_p05": float(np.percentile(boot, 5)),
            "boot_p95": float(np.percentile(boot, 95)),
        })

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
    weights: Mapping[str, float] = WEIGHTS,
) -> CVResult:
    """Run fit_predict on every split and collect per-shot validation errors.

    ``weights`` sets the composite used in every summary of the result, e.g.
    WEIGHTS_WITHOUT_SPIN when spin is an input rather than a prediction.
    """
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
        errors.insert(0, "shot", val_rows.index.to_numpy())
        errors.insert(0, "fold", fold)
        parts.append(errors)
    return CVResult(strategy, pd.concat(parts, ignore_index=True), dict(test_band_shares), dict(weights))


def evaluate(
    fit_predict: FitPredict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, CVResult]:
    """Cross-validate under both strategies. Always report both.

    Returns {"leave_one_session_out": ..., "within_session": ...}. ``test`` supplies
    only inputs: session ids, per-session holdout fractions and speed band shares.
    """
    train_sessions, test_sessions = session_labels(train, test)
    test_band_shares = speed_band(test).value_counts(normalize=True).to_dict()

    grouped = leave_one_session_out_splits(train_sessions)
    fractions = session_holdout_fractions(train_sessions, test_sessions)
    within = within_session_splits(train_sessions, fractions, n_repeats, seed)

    return {
        "leave_one_session_out": cross_validate(fit_predict, train, grouped, train_sessions,
                                                test_band_shares, "leave_one_session_out"),
        "within_session": cross_validate(fit_predict, train, within, train_sessions,
                                         test_band_shares, "within_session"),
    }


def report_table(results: Mapping[str, CVResult]) -> pd.DataFrame:
    """Stack the report() of each strategy into one table."""
    return pd.concat([result.report() for result in results.values()])


def test_mix_table(results: Mapping[str, CVResult]) -> pd.DataFrame:
    """Test-mix composite and its spread (CVResult.test_mix_spread) per strategy."""
    return pd.DataFrame({name: result.test_mix_spread() for name, result in results.items()}).T


def _band_shares(test_band_shares: Mapping[str, float], present: pd.Series) -> pd.Series:
    """Test shares for the bands present in ``present``, renormalised to sum to 1."""
    bands = [b for b in SPEED_BAND_LABELS if b in set(present)]
    shares = pd.Series(test_band_shares).reindex(bands)
    return shares / shares.sum()


def _mix_bands(
    errors: pd.DataFrame,
    test_band_shares: Mapping[str, float],
    weights: Mapping[str, float] = WEIGHTS,
) -> pd.Series:
    """Per-band summaries of ``errors`` reweighted to the test band shares."""
    bands = errors.groupby("speed_band").apply(summarise_errors, weights=weights, include_groups=False)
    shares = _band_shares(test_band_shares, errors["speed_band"])
    bands = bands.reindex(shares.index)
    mixed = bands.drop(columns="n").mul(shares, axis=0).sum()
    mixed["n"] = bands["n"].sum()
    return mixed[bands.columns]


def paired_comparison(scores_a: pd.Series, scores_b: pd.Series) -> pd.Series:
    """Compare two models fold by fold on the same folds (difference = a - b).

    Fold-to-fold spread is dominated by how hard each held-out session is,
    which both models share, so the spread of the paired difference is the
    right noise level for a model comparison, not the spread of either score.

    Returns mean_diff, sd_diff (across folds), se_diff (sd / sqrt(n)),
    t (mean / se), n_folds, and a_better (number of folds where a < b; lower
    scores are better).
    """
    if not scores_a.index.equals(scores_b.index):
        raise ValueError("scores_a and scores_b must cover the same folds in the same order")
    diff = (scores_a - scores_b).to_numpy(dtype=float)
    n = len(diff)
    sd = float(diff.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    mean = float(diff.mean())
    return pd.Series({
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "t": mean / se if n > 1 and se > 0 else np.nan,
        "n_folds": float(n),
        "a_better": float((diff < 0).sum()),
    })


def compare_results(result_a: CVResult, result_b: CVResult, column: str = "composite",
                    weights: Mapping[str, float] | None = None) -> pd.Series:
    """paired_comparison of one per-fold summary column for two CV results.

    Both results must come from the same splits. ``weights`` (optional)
    recomputes both composites with the same weights, e.g. WEIGHTS_WITHOUT_SPIN.
    """
    def per_fold(result: CVResult) -> pd.Series:
        w = result.weights if weights is None else weights
        table = result.errors.groupby("fold").apply(summarise_errors, weights=w, include_groups=False)
        return table[column]

    folds_a = result_a.errors.groupby("fold")["shot"].apply(tuple)
    folds_b = result_b.errors.groupby("fold")["shot"].apply(tuple)
    if not folds_a.equals(folds_b):
        raise ValueError("results were not produced on the same folds")
    return paired_comparison(per_fold(result_a), per_fold(result_b))
