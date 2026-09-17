"""Physics from inputs only, and LightGBM variants built on it.

For one split (training rows with targets, validation rows without):

1. physics_artefacts: fit the globals with a per-shot speed factor on the
   training rows (true spin), then solve spin, tilt and speed factor from the
   checkpoints for training and validation rows alike. The spin prior is the
   LightGBM spin prediction: out-of-fold (inner 5-fold) for training rows, and
   from a model trained on all training rows for validation rows. Its strength
   (prior_sd, rpm) is the standard deviation of the inner out-of-fold spin
   error. Physics predictions come from simulating the fitted states.
2. variant_predictions: shot-frame predictions of
     physics   the simulation alone (spin = fitted spin)
     lgbm      the step 4 baseline (no physics, no session)
     a         LightGBM on step 4 features plus physics features
     b         LightGBM predicting truth minus physics, added to physics
     c         b with the session feature
   All use the step 4 hyperparameters.

Training rows get physics features from the same checkpoint-only solve as
validation rows, so features mean the same thing at fit and predict time. The
globals are fitted on the training rows' targets (four numbers), which is the
only place training targets enter their own features.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from inrange.calibration import GlobalFit, fit_globals, predict_shot_frame, shot_batch
from inrange.features import build_features
from inrange.inverse import InverseResult, solve_states
from inrange.io import CHECKPOINT_NAMES, TARGET_COLS
from inrange.models import LGBM_PARAMS, SHOT_FRAME_TARGETS, LGBMBaseline, shot_frame_targets

PHYSICS_STATE_COLS: list[str] = ["spin", "tilt", "speed", "cp_rms"]
PHYSICS_TARGET_COLS: list[str] = [t for t in SHOT_FRAME_TARGETS if t != "launch_spin_rate"]
VARIANTS: list[str] = ["physics", "lgbm", "a", "b", "c"]

# Which shot-frame targets make up each scored component.
COMPONENT_TARGETS: dict[str, list[str]] = {
    "landing_pos": ["landing_d", "landing_l"],
    "apex_pos": ["apex_d", "apex_l", "apex_h"],
    "apex_t": ["apex_t"],
    "landing_t": ["landing_t"],
    "spin": ["launch_spin_rate"],
}


def _spin_model(train_rows: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(build_features(train_rows, use_session=False), train_rows["launch_spin_rate"])
    return model


def spin_prior(train_rows: pd.DataFrame, val_rows: pd.DataFrame, n_splits: int = 5,
               seed: int = 0) -> tuple[pd.Series, pd.Series, float]:
    """LightGBM spin (rpm): inner out-of-fold for train_rows, full-fit for val_rows,
    and the standard deviation of the inner out-of-fold error (rpm)."""
    oof = pd.Series(np.nan, index=train_rows.index)
    for inner_train, inner_val in KFold(n_splits, shuffle=True, random_state=seed).split(train_rows):
        model = _spin_model(train_rows.iloc[inner_train])
        rows = train_rows.iloc[inner_val]
        oof.loc[rows.index] = model.predict(build_features(rows, use_session=False))
    val = pd.Series(_spin_model(train_rows).predict(build_features(val_rows, use_session=False)),
                    index=val_rows.index)
    sd = float((oof - train_rows["launch_spin_rate"]).std())
    return oof, val, sd


@dataclass
class PhysicsArtefacts:
    """Everything physics-derived for one split. Frames are indexed like the rows.

    states: spin (rpm), tilt (rad), speed, cp_rms (m), corr_spin_speed,
    sd_spin, plus prior (rpm). physics: SHOT_FRAME_TARGETS simulated from the
    states (launch_spin_rate = fitted spin).
    """

    fit: GlobalFit
    prior_sd: float
    train_states: pd.DataFrame
    val_states: pd.DataFrame
    train_physics: pd.DataFrame
    val_physics: pd.DataFrame
    seconds: float
    solve_info: dict


def _solve_and_simulate(rows: pd.DataFrame, fit: GlobalFit, prior: pd.Series,
                        prior_sd: float) -> tuple[pd.DataFrame, pd.DataFrame, InverseResult]:
    batch = shot_batch(rows, spin_rpm=prior.loc[rows.index])
    result = solve_states(batch, fit.aero, fit.scales, prior.loc[rows.index].to_numpy(), prior_sd)
    states = result.states.assign(prior=prior.loc[rows.index])
    solved = shot_batch(rows, spin_rpm=states["spin"])
    physics = predict_shot_frame(solved, fit.aero, states["tilt"].to_numpy(), states["speed"].to_numpy())
    return states, physics, result


def physics_artefacts(train_rows: pd.DataFrame, val_rows: pd.DataFrame,
                      fit: GlobalFit | None = None) -> PhysicsArtefacts:
    """Globals (unless given), spin prior, inverse solves and simulations for one split."""
    started = time.perf_counter()
    if fit is None:
        fit = fit_globals(shot_batch(train_rows), fit_speed=True)
    val_inputs = val_rows.drop(columns=[c for c in TARGET_COLS if c in val_rows.columns])
    train_prior, val_prior, prior_sd = spin_prior(train_rows, val_inputs)
    train_states, train_physics, train_res = _solve_and_simulate(train_rows, fit, train_prior, prior_sd)
    val_states, val_physics, val_res = _solve_and_simulate(val_inputs, fit, val_prior, prior_sd)
    info = {"train_nfev": train_res.nfev, "train_status": train_res.status, "train_seconds": train_res.seconds,
            "val_nfev": val_res.nfev, "val_status": val_res.status, "val_seconds": val_res.seconds}
    return PhysicsArtefacts(fit, prior_sd, train_states, val_states, train_physics, val_physics,
                            time.perf_counter() - started, info)


def all_physics_artefacts(train: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]],
                          n_jobs: int = 8) -> list[PhysicsArtefacts]:
    """physics_artefacts for every split, in parallel."""
    return Parallel(n_jobs=n_jobs)(
        delayed(physics_artefacts)(train.iloc[tr], train.iloc[va]) for tr, va in splits
    )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def physics_features(states: pd.DataFrame, physics: pd.DataFrame) -> pd.DataFrame:
    """Fitted states and simulated targets as model features (prefix phys_)."""
    out = states[PHYSICS_STATE_COLS].add_prefix("phys_")
    return out.join(physics.add_prefix("phys_"))


def _fit_predict_lgbm(x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for target in y_train.columns:
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(x_train, y_train[target])
        out[target] = model.predict(x_val[x_train.columns])
    return pd.DataFrame(out, index=x_val.index)


def variant_predictions(train_rows: pd.DataFrame, val_rows: pd.DataFrame, art: PhysicsArtefacts,
                        variants: list[str] = VARIANTS) -> dict[str, pd.DataFrame]:
    """Shot-frame predictions (SHOT_FRAME_TARGETS) for val_rows from each variant."""
    targets = shot_frame_targets(train_rows)
    out: dict[str, pd.DataFrame] = {}
    if "physics" in variants:
        out["physics"] = art.val_physics[SHOT_FRAME_TARGETS]
    if "lgbm" in variants:
        out["lgbm"] = LGBMBaseline(use_session=False).fit(train_rows).predict_shot_frame(val_rows)

    for name, use_session, residual in [("a", False, False), ("b", False, True), ("c", True, True)]:
        if name not in variants:
            continue
        x_train = build_features(train_rows, use_session).join(physics_features(art.train_states, art.train_physics))
        x_val = build_features(val_rows, use_session).join(physics_features(art.val_states, art.val_physics))
        if residual:
            y = targets - art.train_physics[SHOT_FRAME_TARGETS]
            out[name] = art.val_physics[SHOT_FRAME_TARGETS] + _fit_predict_lgbm(x_train, y, x_val)
        else:
            out[name] = _fit_predict_lgbm(x_train, targets, x_val)
    return out


def combine(predictions: Mapping[str, pd.DataFrame], choice: Mapping[str, str]) -> pd.DataFrame:
    """Shot-frame predictions taking each component's targets from the chosen variant."""
    parts = [predictions[choice[component]][cols] for component, cols in COMPONENT_TARGETS.items()]
    return pd.concat(parts, axis=1)[SHOT_FRAME_TARGETS]


def split_plan(train: pd.DataFrame, test: pd.DataFrame, n_repeats: int = 10,
               seed: int = 0) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """The splits used in step 6: leave-one-session-out and within-session
    (identical to scoring.evaluate). Both frames need a session column."""
    from inrange.scoring import leave_one_session_out_splits, session_holdout_fractions, within_session_splits

    fractions = session_holdout_fractions(train["session"], test["session"])
    return {
        "leave_one_session_out": leave_one_session_out_splits(train["session"]),
        "within_session": within_session_splits(train["session"], fractions, n_repeats, seed),
    }


def run_all_artefacts(train: pd.DataFrame, test: pd.DataFrame, n_jobs: int = 8) -> dict:
    """Physics artefacts for every CV split and for all training rows -> test.

    Returns {"leave_one_session_out": [...], "within_session": [...],
    "full": PhysicsArtefacts (val = test), "seconds": wall time}.
    """
    started = time.perf_counter()
    plan = split_plan(train, test)
    jobs = [(name, i, train.iloc[tr], train.iloc[va]) for name, splits in plan.items()
            for i, (tr, va) in enumerate(splits)]
    jobs.append(("full", 0, train, test))
    results = Parallel(n_jobs=n_jobs)(delayed(physics_artefacts)(tr_rows, va_rows) for _, _, tr_rows, va_rows in jobs)
    out: dict = {name: [None] * len(splits) for name, splits in plan.items()}
    for (name, i, _, _), art in zip(jobs, results):
        if name == "full":
            out["full"] = art
        else:
            out[name][i] = art
    out["seconds"] = time.perf_counter() - started
    return out


# ---------------------------------------------------------------------------
# Evaluation and selection
# ---------------------------------------------------------------------------

def split_predictions(train: pd.DataFrame, plan: Mapping[str, list], artefacts: Mapping[str, list],
                      variants: list[str] = VARIANTS) -> dict[str, list[dict[str, pd.DataFrame]]]:
    """Shot-frame predictions per strategy, per split, per variant."""
    out = {}
    for strategy, splits in plan.items():
        out[strategy] = []
        for (tr, va), art in zip(splits, artefacts[strategy]):
            val_rows = train.iloc[va].drop(columns=TARGET_COLS)
            out[strategy].append(variant_predictions(train.iloc[tr], val_rows, art, variants))
    return out


def cv_results(train: pd.DataFrame, test: pd.DataFrame, plan: Mapping[str, list],
               shot_frame_preds: Mapping[str, list[pd.DataFrame]]) -> dict:
    """scoring.CVResult per strategy for one set of per-split shot-frame predictions."""
    from inrange.features import speed_band
    from inrange.models import targets_from_shot_frame
    from inrange.scoring import cross_validate

    shares = speed_band(test).value_counts(normalize=True).to_dict()
    results = {}
    for strategy, splits in plan.items():
        stored = {}
        for (tr, va), preds in zip(splits, shot_frame_preds[strategy]):
            rows = train.iloc[va]
            stored[frozenset(rows.index)] = targets_from_shot_frame(rows, preds.loc[rows.index])

        def replay(train_rows, val_rows, stored=stored):
            return stored[frozenset(val_rows.index)].loc[val_rows.index]

        results[strategy] = cross_validate(replay, train, splits, train["session"], shares, strategy)
    return results


def paired_table(results: Mapping[str, Mapping[str, object]], baseline: str = "lgbm",
                 columns: tuple[str, ...] = ("composite", "landing_pos", "apex_pos", "apex_t", "landing_t", "spin"),
                 ) -> pd.DataFrame:
    """Paired difference (variant minus baseline) per strategy, variant and column.

    results[variant][strategy] is a CVResult. Negative mean_diff = variant better.
    """
    from inrange.scoring import compare_results

    rows = []
    for variant, by_strategy in results.items():
        if variant == baseline:
            continue
        for strategy, result in by_strategy.items():
            for column in columns:
                stats = compare_results(result, results[baseline][strategy], column)
                rows.append({"variant": variant, "strategy": strategy, "column": column, **stats.to_dict()})
    return pd.DataFrame(rows).set_index(["column", "variant", "strategy"]).sort_index()


def select_per_component(table: pd.DataFrame, within_only: tuple[str, ...] = ("c",)) -> tuple[dict[str, str], pd.DataFrame]:
    """Choose a variant per component (else 'lgbm').

    A variant qualifies for a component when its paired mean difference is
    below minus one standard error in both strategies; variants in within_only
    are judged on within-session CV alone. Among qualifiers the one with the
    most negative within-session mean difference wins.
    """
    table = table.sort_index()
    choice, notes = {}, []
    for component in COMPONENT_TARGETS:
        best, best_diff = "lgbm", 0.0
        for variant in table.loc[component].index.get_level_values("variant").unique():
            strategies = ["within_session"] if variant in within_only else ["leave_one_session_out", "within_session"]
            rows = table.loc[(component, variant)]
            qualifies = all(rows.loc[s, "mean_diff"] < -rows.loc[s, "se_diff"] for s in strategies)
            within_diff = rows.loc["within_session", "mean_diff"]
            notes.append({"component": component, "variant": variant, "qualifies": qualifies,
                          "within_mean_diff": within_diff,
                          "loso_mean_diff": rows.loc["leave_one_session_out", "mean_diff"]})
            if qualifies and within_diff < best_diff:
                best, best_diff = variant, within_diff
        choice[component] = best
    return choice, pd.DataFrame(notes).set_index(["component", "variant"])


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

# Variant per scored component for the submitted model (step 7 carry-over):
# b wherever step 6 selection chose c, because session id adds nothing within
# sessions and hurts when a session is new (CLAUDE.md, "Model choice").
FINAL_CHOICE: dict[str, str] = {
    "landing_pos": "b",
    "apex_pos": "b",
    "apex_t": "b",
    "landing_t": "b",
    "spin": "a",
}


def enforce_apex_floor(targets: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Raise predicted apex height (m above the tee) to at least the highest
    measured checkpoint height. The true apex is never below a checkpoint, and
    in out-of-fold predictions this cuts apex height error on the affected
    shots by about 0.5 m (03_modelling.ipynb, section 4)."""
    from inrange.frame import add_shot_frame

    cp_max = add_shot_frame(rows)[[f"{cp}_h" for cp in CHECKPOINT_NAMES]].max(axis=1)
    out = targets.copy()
    out["apex_h"] = np.maximum(out["apex_h"], cp_max.loc[out.index])
    return out


@dataclass
class ShotPrediction:
    """HybridModel output for a set of rows, all indexed like the rows.

    targets: SHOT_FRAME_TARGETS of the final model (shot frame; m, s, rpm).
    states: fitted spin (rpm), tilt (rad), speed factor, cp_rms (m), prior (rpm).
    physics: SHOT_FRAME_TARGETS simulated from the states alone.
    """

    targets: pd.DataFrame
    states: pd.DataFrame
    physics: pd.DataFrame


class HybridModel:
    """The submitted model: physics inverse solve plus LightGBM, per FINAL_CHOICE.

    fit(train) fits the global coefficients (with per-shot speed factors), the
    LightGBM spin prior, solves every training shot from its checkpoints, and
    trains only the LightGBM models the chosen variants need. predict(rows)
    works on any input rows (no targets, no session), one or many. Predicted
    apex height is floored at the highest measured checkpoint height.
    """

    def __init__(self, choice: Mapping[str, str] = FINAL_CHOICE) -> None:
        if any(v not in ("a", "b") for v in choice.values()):
            raise ValueError("HybridModel supports variants a and b (no session feature)")
        self.choice = dict(choice)
        self.fit_result: GlobalFit | None = None
        self.prior_model: lgb.LGBMRegressor | None = None
        self.prior_sd: float = float("nan")
        self.models: dict[str, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []
        self.train_states: pd.DataFrame | None = None
        self.train_physics: pd.DataFrame | None = None

    def _target_variants(self) -> dict[str, str]:
        return {target: self.choice[component]
                for component, targets in COMPONENT_TARGETS.items() for target in targets}

    def fit(self, train: pd.DataFrame) -> "HybridModel":
        """Fit on training rows with targets. Mirrors physics_artefacts + variant_predictions."""
        self.fit_result = fit_globals(shot_batch(train), fit_speed=True)
        train_prior, _, self.prior_sd = spin_prior(train, train.head(1).drop(columns=TARGET_COLS))
        self.prior_model = _spin_model(train)
        states, physics, _ = _solve_and_simulate(train, self.fit_result, train_prior, self.prior_sd)
        self.train_states, self.train_physics = states, physics

        features = build_features(train, use_session=False).join(physics_features(states, physics))
        self.feature_names = list(features.columns)
        targets = shot_frame_targets(train)
        self.models = {}
        for target, variant in self._target_variants().items():
            y = targets[target] - physics[target] if variant == "b" else targets[target]
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(features, y)
            self.models[target] = model
        return self

    def predict_full(self, rows: pd.DataFrame) -> ShotPrediction:
        """Final targets, fitted states and physics-only targets for input rows."""
        inputs = rows.drop(columns=[c for c in TARGET_COLS if c in rows.columns])
        base = build_features(inputs, use_session=False)
        prior = pd.Series(self.prior_model.predict(base), index=inputs.index)
        states, physics, _ = _solve_and_simulate(inputs, self.fit_result, prior, self.prior_sd)
        features = base.join(physics_features(states, physics))[self.feature_names]
        out = {}
        for target, variant in self._target_variants().items():
            value = self.models[target].predict(features)
            out[target] = physics[target].to_numpy() + value if variant == "b" else value
        targets = pd.DataFrame(out, index=inputs.index)[SHOT_FRAME_TARGETS]
        targets = enforce_apex_floor(targets, inputs)
        return ShotPrediction(targets, states, physics)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """TARGET_COLS in the raw frame for input rows."""
        from inrange.models import targets_from_shot_frame

        return targets_from_shot_frame(rows, self.predict_full(rows).targets)

    def globals_record(self) -> dict:
        """Global coefficients, residual scales and prior strength, JSON-ready."""
        return {**vars(self.fit_result.aero), "prior_sd_rpm": self.prior_sd, "scales": self.fit_result.scales,
                "choice": self.choice}

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "HybridModel":
        import joblib

        return joblib.load(path)
