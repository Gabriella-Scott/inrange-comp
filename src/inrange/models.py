"""Predictive models. Each exposes fit(train) and predict(rows) -> TARGET_COLS frame."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

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
