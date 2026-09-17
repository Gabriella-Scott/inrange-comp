"""Train the current model on all training rows, predict test, write validated submissions.

Outputs:
    outputs/submissions/submission_hybrid_<YYYY-MM-DD>.csv  step 6 hybrid (physics + LightGBM)
    outputs/submissions/submission_mean.csv                training-mean fallback

The hybrid takes each scored component from the variant chosen in
notebooks/03_modelling.ipynb (section 3); see inrange.hybrid. Fitting the
physics globals and solving every shot takes a few minutes.

Usage:
    .venv/bin/python make_submission.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from inrange.features import add_sessions
from inrange.hybrid import combine, physics_artefacts, variant_predictions
from inrange.io import (
    SUBMISSIONS_DIR,
    TARGET_COLS,
    load_sample_submission,
    load_test,
    load_train,
    write_submission,
)
from inrange.models import targets_from_shot_frame

# Variant per component, from the step 6 selection rule (03_modelling.ipynb).
CHOICE: dict[str, str] = {
    "landing_pos": "c",
    "apex_pos": "b",
    "apex_t": "c",
    "landing_t": "c",
    "spin": "a",
}


def predict(train: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Predict TARGET_COLS for each input row (spin rpm, times s, positions m).

    Both frames need a session column (inrange.features.add_sessions).
    """
    artefacts = physics_artefacts(train, rows)
    variants = variant_predictions(train, rows, artefacts, sorted(set(CHOICE.values())))
    return targets_from_shot_frame(rows, combine(variants, CHOICE))


def main() -> None:
    train, test = add_sessions(load_train(), load_test())
    today = dt.date.today().isoformat()

    write_submission(predict(train, test), load_test(), SUBMISSIONS_DIR / f"submission_hybrid_{today}.csv")

    mean_targets = load_sample_submission()[TARGET_COLS]
    write_submission(mean_targets, load_test(), SUBMISSIONS_DIR / "submission_mean.csv")


if __name__ == "__main__":
    main()
