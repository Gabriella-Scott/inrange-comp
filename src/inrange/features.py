"""Derived columns shared by analysis and modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from inrange.frame import add_shot_frame
from inrange.io import CHECKPOINT_NAMES

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


# Tee positions (launch_x, launch_y in m) found in 01_eda.ipynb, labelled by launch_x.
# T3 is the elevated bay (launch_z = 4.125 m). Fixed so labels never depend on
# which rows are passed in.
TEE_POSITIONS: dict[str, tuple[float, float]] = {
    "T1": (-24.900, 37.467),
    "T2": (-23.344, 34.627),
    "T3": (-22.744, 30.764),
    "T4": (-21.456, 31.779),
}
TEE_LABELS: list[str] = list(TEE_POSITIONS)

# Sessions found on train + test launch times combined (01_eda.ipynb).
N_SESSIONS: int = 11


def assign_tees(df: pd.DataFrame) -> pd.Series:
    """Tee label per row ('T1'..'T4'): the nearest entry in TEE_POSITIONS (m)."""
    labels = np.array(TEE_LABELS)
    positions = np.array(list(TEE_POSITIONS.values()))
    dx = df["launch_x"].to_numpy()[:, None] - positions[None, :, 0]
    dy = df["launch_y"].to_numpy()[:, None] - positions[None, :, 1]
    nearest = np.argmin(dx ** 2 + dy ** 2, axis=1)
    return pd.Series(labels[nearest], index=df.index, name="tee")


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


def add_sessions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of train and test with a ``session`` column (see session_labels)."""
    train_sessions, test_sessions = session_labels(train, test)
    return train.assign(session=train_sessions), test.assign(session=test_sessions)


# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------

POINT_NAMES: list[str] = ["launch"] + CHECKPOINT_NAMES  # P0 .. P4


def _quadratic_slope(t: list[np.ndarray], y: list[np.ndarray], at: np.ndarray) -> np.ndarray:
    """Derivative at ``at`` of the parabola through three (t, y) points (per row)."""
    (t0, t1, t2), (y0, y1, y2) = t, y
    return (y0 * (2 * at - t1 - t2) / ((t0 - t1) * (t0 - t2))
            + y1 * (2 * at - t0 - t2) / ((t1 - t0) * (t1 - t2))
            + y2 * (2 * at - t0 - t1) / ((t2 - t0) * (t2 - t1)))


def _quadratic_curvature(t: list[np.ndarray], y: list[np.ndarray]) -> np.ndarray:
    """Second derivative of the parabola through three (t, y) points (per row)."""
    (t0, t1, t2), (y0, y1, y2) = t, y
    return 2 * (y0 / ((t0 - t1) * (t0 - t2))
                + y1 / ((t1 - t0) * (t1 - t2))
                + y2 / ((t2 - t0) * (t2 - t1)))


def build_features(df: pd.DataFrame, use_session: bool = True) -> pd.DataFrame:
    """Model inputs in the tee-centred shot frame (see inrange.frame).

    Uses only input columns; needs a ``session`` column when use_session is True.
    launch_time is never used as a number.

    Units: distances m, times s, speeds m/s, accelerations m/s^2, angles degrees
    (launch_direction positive to the left of the range heading), curve rates m/m.

    Launch: ball_speed, launch_angle, launch_direction, launch_height (launch_z),
        tee_d (tee position relative to the checkpoint tee line).
    Per checkpoint k = 1..4: cpk_t, cpk_h, cpk_l; velocity at the checkpoint
        (cpk_vd, cpk_vl, cpk_vh) from the parabola through it and its
        neighbours (cp4 uses cp2, cp3, cp4).
    Per segment k = 1..4 (point k-1 to point k, point 0 = launch): segk_speed
        (3D distance / time), segk_decel (drop in speed from the previous
        segment, or launch speed for k = 1, per second between segment
        midpoints), segk_curve (change in lateral offset per metre downrange).
    Accelerations from parabolas through consecutive point triples j = 1..3
        (points j-1, j, j+1): accj_d, accj_l, accj_h. accj_h + 9.81 is the
        upward force per unit mass other than gravity, mostly backspin lift.
    Net summary (cp4_h and cp4_l are already the height and lateral offset at
        the net): net_curve (cp4 lateral offset relative to the launch direction
        line), curve_change (seg4_curve - seg1_curve).
    Categorical: tee, and session if use_session.
    """
    frame = add_shot_frame(df)
    out = pd.DataFrame(index=df.index)

    speed = np.sqrt(frame["launch_vd"] ** 2 + frame["launch_vl"] ** 2 + frame["launch_vh"] ** 2)
    horizontal = np.hypot(frame["launch_vd"], frame["launch_vl"])
    out["ball_speed"] = speed
    out["launch_angle"] = np.degrees(np.arctan2(frame["launch_vh"], horizontal))
    out["launch_direction"] = np.degrees(np.arctan2(frame["launch_vl"], frame["launch_vd"]))
    out["launch_height"] = df["launch_z"]
    out["tee_d"] = frame["tee_d"]

    # Points P0 (launch, at the tee origin) to P4 (net) in the shot frame.
    zeros = np.zeros(len(df))
    t = [zeros] + [frame[f"{cp}_t"].to_numpy() for cp in CHECKPOINT_NAMES]
    d = [zeros] + [frame[f"{cp}_d"].to_numpy() for cp in CHECKPOINT_NAMES]
    l = [zeros] + [frame[f"{cp}_l"].to_numpy() for cp in CHECKPOINT_NAMES]
    h = [zeros] + [frame[f"{cp}_h"].to_numpy() for cp in CHECKPOINT_NAMES]

    for k, cp in enumerate(CHECKPOINT_NAMES, start=1):
        out[f"{cp}_t"] = t[k]
        out[f"{cp}_h"] = h[k]
        out[f"{cp}_l"] = l[k]
        lo, hi = (k - 1, k + 1) if k < 4 else (2, 4)
        window = slice(lo, hi + 1)
        for axis, values in (("d", d), ("l", l), ("h", h)):
            out[f"{cp}_v{axis}"] = _quadratic_slope(t[window], values[window], t[k])

    previous_speed = speed.to_numpy()
    previous_mid = zeros
    for k in range(1, 5):
        dt = t[k] - t[k - 1]
        seg_speed = np.sqrt((d[k] - d[k - 1]) ** 2 + (l[k] - l[k - 1]) ** 2 + (h[k] - h[k - 1]) ** 2) / dt
        mid = (t[k] + t[k - 1]) / 2
        out[f"seg{k}_speed"] = seg_speed
        out[f"seg{k}_decel"] = (previous_speed - seg_speed) / (mid - previous_mid)
        out[f"seg{k}_curve"] = (l[k] - l[k - 1]) / (d[k] - d[k - 1])
        previous_speed, previous_mid = seg_speed, mid

    for j in range(1, 4):
        window = slice(j - 1, j + 2)
        out[f"acc{j}_d"] = _quadratic_curvature(t[window], d[window])
        out[f"acc{j}_l"] = _quadratic_curvature(t[window], l[window])
        out[f"acc{j}_h"] = _quadratic_curvature(t[window], h[window])

    out["net_curve"] = l[4] - d[4] * np.tan(np.radians(out["launch_direction"]))
    out["curve_change"] = out["seg4_curve"] - out["seg1_curve"]

    out["tee"] = pd.Categorical(assign_tees(df), categories=TEE_LABELS)
    if use_session:
        out["session"] = pd.Categorical(df["session"], categories=list(range(N_SESSIONS)))
    return out
