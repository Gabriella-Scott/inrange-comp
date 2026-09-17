"""Predictive models. Each exposes fit(train) and predict(rows) -> TARGET_COLS frame."""

from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from inrange.calibration import GlobalFit, fit_globals, fit_tilts, predict_shot_frame, shot_batch
from inrange.features import build_features
from inrange.frame import add_shot_frame, from_shot_frame
from inrange.io import TARGET_COLS

# Targets modelled in the tee-centred shot frame (m, s, rpm). landing_h is not
# modelled: level landing is at launch height by definition, so landing_z is
# predicted as launch_z.
SHOT_FRAME_TARGETS: list[str] = [
    "launch_spin_rate",
    "apex_t", "apex_d", "apex_l", "apex_h",
    "landing_t", "landing_d", "landing_l",
]

# Conservative settings for 491 rows: shallow trees, a slow learning rate, row
# and column subsampling, and at least 15 shots per leaf. Not tuned.
LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 0,
    "verbose": -1,
}


def shot_frame_targets(train: pd.DataFrame) -> pd.DataFrame:
    """SHOT_FRAME_TARGETS for training rows (positions relative to each tee)."""
    return add_shot_frame(train)[SHOT_FRAME_TARGETS]


def targets_from_shot_frame(rows: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Rotate shot-frame predictions back to TARGET_COLS in the raw x, y, z frame."""
    origin = (rows["launch_x"], rows["launch_y"], rows["launch_z"])
    out = pd.DataFrame(index=rows.index)
    out["launch_spin_rate"] = predicted["launch_spin_rate"]
    out["apex_t"] = predicted["apex_t"]
    out["apex_x"], out["apex_y"], out["apex_z"] = from_shot_frame(
        predicted["apex_d"], predicted["apex_l"], predicted["apex_h"], *origin)
    out["landing_t"] = predicted["landing_t"]
    out["landing_x"], out["landing_y"], out["landing_z"] = from_shot_frame(
        predicted["landing_d"], predicted["landing_l"], np.zeros(len(rows)), *origin)
    return out[TARGET_COLS]


class LGBMBaseline:
    """One LightGBM regressor per shot-frame target on build_features inputs."""

    def __init__(self, params: dict[str, object] | None = None, use_session: bool = True) -> None:
        self.params = dict(LGBM_PARAMS if params is None else params)
        self.use_session = use_session
        self.models: dict[str, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []

    def fit(self, train: pd.DataFrame) -> "LGBMBaseline":
        """Fit on training rows (inputs, targets, and ``session`` if use_session)."""
        features = build_features(train, self.use_session)
        targets = shot_frame_targets(train)
        self.feature_names = list(features.columns)
        self.models = {}
        for target in SHOT_FRAME_TARGETS:
            model = lgb.LGBMRegressor(**self.params)
            model.fit(features, targets[target])
            self.models[target] = model
        return self

    def predict_shot_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predictions for SHOT_FRAME_TARGETS (m, s, rpm), indexed like rows."""
        features = build_features(rows, self.use_session)[self.feature_names]
        return pd.DataFrame(
            {target: model.predict(features) for target, model in self.models.items()},
            index=rows.index,
        )

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predictions for TARGET_COLS in the raw frame, indexed like rows."""
        return targets_from_shot_frame(rows, self.predict_shot_frame(rows))

    def feature_importance(self) -> pd.DataFrame:
        """Share of total split gain per feature (rows) for each target (columns)."""
        table = pd.DataFrame({
            target: pd.Series(model.booster_.feature_importance("gain"), index=self.feature_names)
            for target, model in self.models.items()
        })
        return table / table.sum()


def lgbm_fit_predict(use_session: bool = True):
    """A scoring.FitPredict that trains a fresh LGBMBaseline on each fold."""

    def fit_predict(train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> pd.DataFrame:
        return LGBMBaseline(use_session=use_session).fit(train_rows).predict(val_rows)

    return fit_predict


class OracleSpinPhysics:
    """Physics simulator with fitted global coefficients and the TRUE launch spin.

    Not a submittable model: spin magnitude is taken from the targets, so this
    measures the ceiling of the physics approach when spin is known. Training
    fits the globals and per-shot tilts (fit_globals). Prediction fits each
    shot's tilt from its checkpoints only, then simulates apex and landing.
    """

    def __init__(self, fit_tau: bool = False) -> None:
        self.fit_tau = fit_tau
        self.fit_result: GlobalFit | None = None

    def fit(self, train: pd.DataFrame) -> "OracleSpinPhysics":
        self.fit_result = fit_globals(shot_batch(train), fit_tau=self.fit_tau)
        return self

    def predict_with_tilt(self, rows: pd.DataFrame, spin_rpm: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """TARGET_COLS predictions and the fitted tilts (rad) for rows, given spin (rpm)."""
        fit = self.fit_result
        batch = shot_batch(rows, spin_rpm=spin_rpm.loc[rows.index])
        tilt = fit_tilts(batch, fit.aero, fit.scales)
        shot_frame = predict_shot_frame(batch, fit.aero, tilt.to_numpy())
        return targets_from_shot_frame(rows, shot_frame), tilt

    def predict(self, rows: pd.DataFrame, spin_rpm: pd.Series) -> pd.DataFrame:
        return self.predict_with_tilt(rows, spin_rpm)[0]


def _fit_physics_fold(train_rows: pd.DataFrame, val_rows: pd.DataFrame, spin_rpm: pd.Series) -> dict:
    model = OracleSpinPhysics().fit(train_rows)
    predictions, tilt = model.predict_with_tilt(val_rows, spin_rpm)
    return {"fit": model.fit_result, "predictions": predictions, "val_tilt": tilt}


def physics_fold_results(train: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]],
                         n_jobs: int = -1) -> list[dict]:
    """Fit OracleSpinPhysics on every split in parallel.

    Returns one dict per split with the GlobalFit ("fit"), predictions for the
    validation rows ("predictions", TARGET_COLS) and their checkpoint-only
    tilts ("val_tilt", rad).
    """
    spin = train["launch_spin_rate"]
    return Parallel(n_jobs=n_jobs)(
        delayed(_fit_physics_fold)(train.iloc[tr], train.iloc[va].drop(columns=TARGET_COLS), spin)
        for tr, va in splits
    )


def replay_fit_predict(fold_results: list[dict]) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """A scoring.FitPredict that returns stored predictions for each validation set."""
    stored = {frozenset(result["predictions"].index): result["predictions"] for result in fold_results}

    def fit_predict(train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> pd.DataFrame:
        return stored[frozenset(val_rows.index)].loc[val_rows.index]

    return fit_predict
