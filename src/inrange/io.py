"""Data loading, column groups and submission validation.

Units, as given by the competition data description:
    positions in metres (m), velocities in metres per second (m/s),
    spin in revolutions per minute (rpm), times in seconds (s).
    ``launch_time`` is an absolute Unix timestamp; every other ``*_t``
    column is a duration in seconds from that track's own launch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = REPO_ROOT / "data" / "raw"
SUBMISSIONS_DIR: Path = REPO_ROOT / "outputs" / "submissions"

ID_COL: str = "track_id"

LAUNCH_COLS: list[str] = [
    "launch_time",  # Unix timestamp, s
    "launch_x", "launch_y", "launch_z",  # m
    "launch_vx", "launch_vy", "launch_vz",  # m/s
]

CHECKPOINT_NAMES: list[str] = ["cp1", "cp2", "cp3", "cp4"]  # 15, 30, 45, 60 m downrange
CHECKPOINT_COLS: list[str] = [
    f"{cp}_{field}" for cp in CHECKPOINT_NAMES for field in ("t", "x", "y", "z")
]

INPUT_COLS: list[str] = [ID_COL] + LAUNCH_COLS + CHECKPOINT_COLS

TARGET_COLS: list[str] = [
    "launch_spin_rate",  # rpm
    "apex_t", "apex_x", "apex_y", "apex_z",  # s, m, m, m
    "landing_t", "landing_x", "landing_y", "landing_z",  # s, m, m, m
]

SUBMISSION_COLS: list[str] = [ID_COL] + TARGET_COLS


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a competition CSV, keeping track_id as a string and floats exact."""
    return pd.read_csv(path, dtype={ID_COL: str}, float_precision="round_trip")


def load_train(path: Path = RAW_DIR / "train.csv") -> pd.DataFrame:
    """Load the training set: input columns plus the nine target columns."""
    return _read_csv(path)


def load_test(path: Path = RAW_DIR / "test.csv") -> pd.DataFrame:
    """Load the test set: input columns only."""
    return _read_csv(path)


def load_sample_submission(path: Path = RAW_DIR / "sample_submission.csv") -> pd.DataFrame:
    """Load the sample submission (every target filled with the training-set mean)."""
    return _read_csv(path)


def validate_submission(df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """Check a submission against the test set.

    Checks row count, exact column names and order, that every test track_id
    appears exactly once and no unknown or altered track_id is present, and
    that every target value is a finite number.

    Returns a list of problems; an empty list means the submission is valid.
    """
    problems: list[str] = []

    if list(df.columns) != SUBMISSION_COLS:
        problems.append(f"columns are {list(df.columns)}, expected {SUBMISSION_COLS}")

    if len(df) != len(test_df):
        problems.append(f"row count is {len(df)}, expected {len(test_df)}")

    if ID_COL not in df.columns:
        problems.append(f"missing {ID_COL} column")
        return problems

    sub_ids = df[ID_COL]
    test_ids = set(test_df[ID_COL])

    duplicated = sub_ids[sub_ids.duplicated()].unique()
    if len(duplicated) > 0:
        problems.append(f"{len(duplicated)} track_id values appear more than once")

    unknown = set(sub_ids) - test_ids
    if unknown:
        problems.append(f"{len(unknown)} track_id values are not in test (altered or extra)")

    missing = test_ids - set(sub_ids)
    if missing:
        problems.append(f"{len(missing)} test track_id values are missing")

    present_targets = [c for c in TARGET_COLS if c in df.columns]
    values = df[present_targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    if n_nan:
        problems.append(f"{n_nan} target values are NaN or non-numeric")
    if n_inf:
        problems.append(f"{n_inf} target values are infinite")

    return problems


def write_submission(predictions: pd.DataFrame, test_df: pd.DataFrame, path: Path) -> None:
    """Validate and write a submission; print the result.

    ``predictions`` holds TARGET_COLS indexed like ``test_df``. Raises
    ValueError listing every problem, and writes nothing, if validation fails.
    """
    submission = pd.concat([test_df[[ID_COL]], predictions[TARGET_COLS]], axis=1)[SUBMISSION_COLS]
    problems = validate_submission(submission, test_df)
    if problems:
        raise ValueError(f"invalid submission {path.name}: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"VALID   {path.name}: {len(submission)} rows, {len(submission.columns)} columns, no problems found")
