"""Write submission files from test.csv and validate them.

Outputs:
    outputs/submissions/submission_baseline_zero.csv  predict() applied to test
    outputs/submissions/submission_mean.csv           sample_submission passed through validation

The placeholder predict() returns zeros. Later steps replace it with a model.

Usage:
    .venv/bin/python make_submission.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from inrange.io import (
    ID_COL,
    SUBMISSION_COLS,
    SUBMISSIONS_DIR,
    TARGET_COLS,
    load_sample_submission,
    load_test,
    validate_submission,
)


def predict(rows: pd.DataFrame) -> pd.DataFrame:
    """Predict the nine targets for each input row.

    Returns a frame with one row per input row and columns TARGET_COLS
    (spin in rpm, times in s, positions in m). Placeholder: all zeros.
    """
    return pd.DataFrame(0.0, index=rows.index, columns=TARGET_COLS)


def write_validated(df: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    """Validate a submission, print the result, and write it only if valid."""
    problems = validate_submission(df, test)
    if problems:
        print(f"INVALID {path.name}:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"VALID   {path.name}: {len(df)} rows, {len(df.columns)} columns")


def main() -> None:
    test = load_test()

    predictions = predict(test)
    zero_sub = pd.concat([test[[ID_COL]], predictions], axis=1)[SUBMISSION_COLS]
    write_validated(zero_sub, test, SUBMISSIONS_DIR / "submission_baseline_zero.csv")

    mean_sub = load_sample_submission()
    write_validated(mean_sub, test, SUBMISSIONS_DIR / "submission_mean.csv")


if __name__ == "__main__":
    main()
