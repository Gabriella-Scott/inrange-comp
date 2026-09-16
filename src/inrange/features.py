"""Derived columns shared by analysis and modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Gap between consecutive shots (s) that starts a new practice session.
# Within-day gaps are at most ~41 min and between-day gaps at least ~21.7 h,
# so any threshold between those gives the same split (see 01_eda.ipynb).
SESSION_GAP_S: float = 3600.0

ELEVATED_TEE_MIN_Z: float = 1.0  # m; the upper-tier bay launches at z = 4.125 m


def assign_sessions(launch_time: pd.Series, gap_s: float = SESSION_GAP_S) -> pd.Series:
    """Session index (0, 1, ...) in chronological order from Unix launch times (s).

    Pass train and test launch times together so both share session labels.
    """
    order = np.argsort(launch_time.to_numpy(), kind="stable")
    sorted_times = launch_time.to_numpy()[order]
    new_session = np.concatenate([[False], np.diff(sorted_times) > gap_s])
    session_sorted = np.cumsum(new_session)
    session = np.empty_like(session_sorted)
    session[order] = session_sorted
    return pd.Series(session, index=launch_time.index, name="session")


def assign_tees(df: pd.DataFrame) -> pd.Series:
    """Tee label per row, e.g. 'T1'..'T4', ordered by launch_x (m)."""
    keys = df["launch_x"].round(3).astype(str) + "," + df["launch_y"].round(3).astype(str)
    ordered = df.assign(_key=keys).sort_values("launch_x")["_key"].unique()
    labels = {key: f"T{i + 1}" for i, key in enumerate(ordered)}
    return keys.map(labels).rename("tee")


# Ball speed bands (m/s) used to break down CV errors. Test is denser than
# train above 70 m/s (01_eda.ipynb, section 6).
SPEED_BAND_EDGES: list[float] = [0.0, 50.0, 70.0, np.inf]
SPEED_BAND_LABELS: list[str] = ["<50", "50-70", ">=70"]


def ball_speed(df: pd.DataFrame) -> pd.Series:
    """Launch ball speed (m/s) from launch_vx, launch_vy, launch_vz (m/s)."""
    speed = np.sqrt(df["launch_vx"] ** 2 + df["launch_vy"] ** 2 + df["launch_vz"] ** 2)
    return speed.rename("ball_speed")


def speed_band(df: pd.DataFrame) -> pd.Series:
    """Ball speed band label per row: '<50', '50-70' or '>=70' (m/s)."""
    bands = pd.cut(ball_speed(df), SPEED_BAND_EDGES, labels=SPEED_BAND_LABELS, right=False)
    return bands.astype(str).rename("speed_band")


def session_labels(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Session ids for train and test rows, assigned on their combined launch times.

    Returned Series keep the original indexes of ``train`` and ``test``.
    """
    combined = pd.concat([train["launch_time"], test["launch_time"]], ignore_index=True)
    sessions = assign_sessions(combined)
    train_sessions = pd.Series(sessions.iloc[: len(train)].to_numpy(), index=train.index, name="session")
    test_sessions = pd.Series(sessions.iloc[len(train):].to_numpy(), index=test.index, name="session")
    return train_sessions, test_sessions
