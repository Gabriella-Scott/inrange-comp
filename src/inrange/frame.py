"""Rotation between the raw range frame (x, y, z) and a shot-aligned frame.

The shot-aligned frame is right-handed, in metres:
    d  downrange distance along the fixed range heading
    l  lateral offset, positive to the LEFT of the target line (d x l = up)
    h  height, identical to z

Frame origin is each track's own tee (launch_x, launch_y, launch_z), so launch
is at (0, 0, 0) and level landing is at h = 0 by definition.

The heading and checkpoint lines were derived in notebooks/01_eda.ipynb: the
four checkpoint crossings are fixed lines perpendicular to HEADING_RAD, 15 m
apart, measured from a common tee line (not from each tee). ``estimate_heading``
reproduces that derivation from data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from inrange.io import CHECKPOINT_NAMES

# Range heading, radians anticlockwise from +x. Fitted on train + test inputs;
# checkpoint projections onto it are constant to ~1e-11 m.
HEADING_RAD: float = 0.495090112985
HEADING_DEG: float = float(np.degrees(HEADING_RAD))

# Projection onto the heading (m, measured from the raw x-y origin) of the
# common tee line from which the 15/30/45/60 m checkpoint lines are measured.
TEE_LINE_PROJECTION: float = -4.137921
CHECKPOINT_DISTANCES: dict[str, float] = {"cp1": 15.0, "cp2": 30.0, "cp3": 45.0, "cp4": 60.0}

DOWNRANGE_UNIT: np.ndarray = np.array([np.cos(HEADING_RAD), np.sin(HEADING_RAD)])
LATERAL_UNIT: np.ndarray = np.array([-np.sin(HEADING_RAD), np.cos(HEADING_RAD)])

POINT_PREFIXES: list[str] = CHECKPOINT_NAMES + ["apex", "landing"]


def estimate_heading(df: pd.DataFrame) -> float:
    """Heading (radians from +x) that makes every checkpoint a straight line.

    Minimises the within-checkpoint variance of the projection of (cpN_x, cpN_y)
    onto the heading, summed over the four checkpoints.
    """

    def spread(theta: float) -> float:
        unit = np.array([np.cos(theta), np.sin(theta)])
        total = 0.0
        for cp in CHECKPOINT_NAMES:
            projection = df[f"{cp}_x"] * unit[0] + df[f"{cp}_y"] * unit[1]
            total += float(((projection - projection.mean()) ** 2).sum())
        return total

    result = minimize_scalar(spread, bounds=(0.0, np.pi / 2), method="bounded",
                             options={"xatol": 1e-14})
    return float(result.x)


def to_shot_frame(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    origin_x: np.ndarray, origin_y: np.ndarray, origin_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw positions (m) to (downrange, lateral, height) relative to an origin (m)."""
    dx = np.asarray(x) - np.asarray(origin_x)
    dy = np.asarray(y) - np.asarray(origin_y)
    d = dx * DOWNRANGE_UNIT[0] + dy * DOWNRANGE_UNIT[1]
    l = dx * LATERAL_UNIT[0] + dy * LATERAL_UNIT[1]
    h = np.asarray(z) - np.asarray(origin_z)
    return d, l, h


def from_shot_frame(
    d: np.ndarray, l: np.ndarray, h: np.ndarray,
    origin_x: np.ndarray, origin_y: np.ndarray, origin_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(downrange, lateral, height) relative to an origin (m) back to raw x, y, z (m)."""
    d = np.asarray(d)
    l = np.asarray(l)
    x = np.asarray(origin_x) + d * DOWNRANGE_UNIT[0] + l * LATERAL_UNIT[0]
    y = np.asarray(origin_y) + d * DOWNRANGE_UNIT[1] + l * LATERAL_UNIT[1]
    z = np.asarray(origin_z) + np.asarray(h)
    return x, y, z


def velocity_to_shot_frame(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw velocity (m/s) to (downrange, lateral, vertical) components (m/s)."""
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vd = vx * DOWNRANGE_UNIT[0] + vy * DOWNRANGE_UNIT[1]
    vl = vx * LATERAL_UNIT[0] + vy * LATERAL_UNIT[1]
    return vd, vl, np.asarray(vz)


def add_shot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with tee-relative shot-frame columns added.

    For each point prefix present (cp1..cp4, apex, landing) adds ``{p}_d``,
    ``{p}_l``, ``{p}_h`` in metres. Adds ``launch_vd``, ``launch_vl``,
    ``launch_vh`` in m/s, and ``tee_d``: the tee's downrange position (m)
    relative to the common tee line the checkpoints are measured from.
    """
    out = df.copy()
    origin = (df["launch_x"], df["launch_y"], df["launch_z"])
    for prefix in POINT_PREFIXES:
        if f"{prefix}_x" not in df.columns:
            continue
        d, l, h = to_shot_frame(df[f"{prefix}_x"], df[f"{prefix}_y"], df[f"{prefix}_z"], *origin)
        out[f"{prefix}_d"], out[f"{prefix}_l"], out[f"{prefix}_h"] = d, l, h
    vd, vl, vh = velocity_to_shot_frame(df["launch_vx"], df["launch_vy"], df["launch_vz"])
    out["launch_vd"], out["launch_vl"], out["launch_vh"] = vd, vl, vh
    out["tee_d"] = (df["launch_x"] * DOWNRANGE_UNIT[0] + df["launch_y"] * DOWNRANGE_UNIT[1]
                    - TEE_LINE_PROJECTION)
    return out
