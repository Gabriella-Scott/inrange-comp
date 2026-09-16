"""Train the current model on all training rows, predict test, write validated submissions.

Outputs:
    outputs/submissions/submission_lgbm_baseline_<YYYY-MM-DD>.csv  LGBMBaseline predictions
    outputs/submissions/submission_mean.csv                        training-mean fallback

Usage:
    .venv/bin/python make_submission.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from inrange.io import (
    SUBMISSIONS_DIR,
    TARGET_COLS,
    load_sample_submission,
    load_test,
    load_train,
    write_submission,
)
from inrange.models import LGBMBaseline


def predict(train: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Predict TARGET_COLS for each input row (spin rpm, times s, positions m).

    Current model: LGBMBaseline without the session feature (see 03_modelling.ipynb).
    """
    return LGBMBaseline(use_session=False).fit(train).predict(rows)


def main() -> None:
    train = load_train()
    test = load_test()
    today = dt.date.today().isoformat()

    write_submission(predict(train, test), test, SUBMISSIONS_DIR / f"submission_lgbm_baseline_{today}.csv")

    mean_targets = load_sample_submission()[TARGET_COLS]
    write_submission(mean_targets, test, SUBMISSIONS_DIR / "submission_mean.csv")


if __name__ == "__main__":
    main()
