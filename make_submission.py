"""Train the final model on all training rows, predict test, and write every downstream artefact.

Outputs (all rebuilt from data/raw, nothing read from notebook caches):
    outputs/submissions/submission_hybrid_b_<YYYY-MM-DD>.csv  final model predictions
    outputs/submissions/submission_mean.csv                  training-mean fallback
    models/final_model.joblib                                trained HybridModel (gitignored)
    data/processed/physics_states.csv                        per-shot spin, tilt, speed factor
    data/processed/physics_globals.json                      global coefficients and scales

The final model is inrange.hybrid.HybridModel with FINAL_CHOICE: variant b
(LightGBM on the physics residual) for positions and times, variant a
(physics features) for spin. Takes about a minute.

Usage:
    .venv/bin/python make_submission.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from inrange.hybrid import HybridModel
from inrange.io import (
    REPO_ROOT,
    SUBMISSIONS_DIR,
    TARGET_COLS,
    load_sample_submission,
    load_test,
    load_train,
    write_submission,
)
from inrange.models import targets_from_shot_frame

MODEL_PATH: Path = REPO_ROOT / "models" / "final_model.joblib"
PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"


def main() -> None:
    train = load_train()
    test = load_test()
    today = dt.date.today().isoformat()

    model = HybridModel().fit(train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved model to {MODEL_PATH.relative_to(REPO_ROOT)}")

    prediction = model.predict_full(test)
    write_submission(targets_from_shot_frame(test, prediction.targets), test,
                     SUBMISSIONS_DIR / f"submission_hybrid_b_{today}.csv")

    state_cols = ["spin", "tilt", "speed", "cp_rms", "prior"]
    states = pd.concat([
        model.train_states[state_cols].assign(track_id=train["track_id"], split="train"),
        prediction.states[state_cols].assign(track_id=test["track_id"], split="test"),
    ])[["track_id", "split"] + state_cols]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    states.to_csv(PROCESSED_DIR / "physics_states.csv", index=False)
    (PROCESSED_DIR / "physics_globals.json").write_text(json.dumps(model.globals_record(), indent=2))
    print(f"saved {len(states)} per-shot states and the global coefficients to data/processed/")

    write_submission(load_sample_submission()[TARGET_COLS], test, SUBMISSIONS_DIR / "submission_mean.csv")


if __name__ == "__main__":
    main()
