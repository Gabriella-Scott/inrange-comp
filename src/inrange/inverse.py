"""Per-shot state from the checkpoints only: spin, spin-axis tilt, speed factor.

With the global coefficients fixed, each shot has three unknowns:

    spin   launch spin magnitude (rpm)
    tilt   spin axis tilt (rad), convention as in inrange.physics
    speed  factor k on the given launch velocity (initial velocity = k * v0)

fitted to the 12 checkpoint residuals (simulated minus observed position at
each observed cp_t), scaled and robustified exactly as in inrange.calibration.
No apex, landing or spin targets are used, so this works on test shots.

Optional spin prior. Spin and k both change how fast the ball slows, so they
can trade off. A prior adds one residual per shot,

    (spin - prior_rpm) / prior_sd_rpm

pulling spin towards an independent estimate (the LightGBM out-of-fold spin
prediction) with strength set by prior_sd_rpm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from inrange.calibration import (
    ROBUST_THRESHOLD,
    ShotBatch,
    _column_divisors,
    _grouped_soft_l1,
    _residual_blocks,
    initial_tilt,
)
from inrange.physics import Aero

# Training spin range (rpm), from load_train()["launch_spin_rate"] min and max.
SPIN_BOUNDS: tuple[float, float] = (775.0, 12349.0)
# Speed factor bounds: the global fit with a per-shot factor (02_physics.ipynb)
# puts the 0.5th to 99.5th percentiles at about 0.925 to 1.085.
SPEED_BOUNDS: tuple[float, float] = (0.90, 1.10)
SPIN_START: float = 5660.0  # rpm, training median
SPEED_START: float = 0.973  # training median of the fitted factor


@dataclass
class InverseResult:
    """Per-shot fitted state, indexed like the batch.

    states: columns spin (rpm), tilt (rad), speed (factor), cp_rms (m, RMS of
    the 12 raw checkpoint residuals), corr_spin_speed and sd_spin (rpm), the
    correlation and standard deviation of spin from the local Jacobian (in
    units where each checkpoint residual has its typical size).
    """

    states: pd.DataFrame
    nfev: int
    seconds: float
    status: int


def solve_states(batch: ShotBatch, aero: Aero, scales: dict[str, float],
                 spin_prior: np.ndarray | None = None, prior_sd: float | None = None,
                 max_nfev: int = 60) -> InverseResult:
    """Fit spin, tilt and speed factor per shot from checkpoints only."""
    started = time.perf_counter()
    n = batch.n
    divisor, counts = _column_divisors(scales, with_targets=False)
    use_prior = spin_prior is not None
    width = 13 if use_prior else 12
    all_counts = np.append(counts, 1.0) if use_prior else counts

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return x[:n], x[n:2 * n], x[2 * n:]

    def residuals(x: np.ndarray) -> np.ndarray:
        spin, tilt, speed = unpack(x)
        cp = _residual_blocks(batch, aero, tilt, False, speed, spin) / divisor
        if use_prior:
            # Counted as a group of one: quadratic up to ROBUST_THRESHOLD
            # prior standard deviations, linear beyond.
            prior = ((spin - spin_prior) / prior_sd)[:, None]
            cp = np.hstack([cp, prior])
        return cp.ravel()

    sparsity = lil_matrix((n * width, 3 * n), dtype=int)
    for i in range(n):
        for j in range(3):
            sparsity[i * width:(i + 1) * width, j * n + i] = 1

    spin0 = np.full(n, SPIN_START) if spin_prior is None else np.clip(spin_prior, *SPIN_BOUNDS)
    x0 = np.concatenate([spin0, initial_tilt(batch), np.full(n, SPEED_START)])
    lower = np.concatenate([np.full(n, SPIN_BOUNDS[0]), np.full(n, -np.radians(80)), np.full(n, SPEED_BOUNDS[0])])
    upper = np.concatenate([np.full(n, SPIN_BOUNDS[1]), np.full(n, np.radians(80)), np.full(n, SPEED_BOUNDS[1])])
    x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)
    result = least_squares(residuals, x0, jac_sparsity=sparsity, bounds=(lower, upper),
                           loss=_grouped_soft_l1(np.tile(all_counts, n), ROBUST_THRESHOLD),
                           x_scale="jac", max_nfev=max_nfev, method="trf")
    spin, tilt, speed = unpack(result.x)

    raw = _residual_blocks(batch, aero, tilt, False, speed, spin)
    corr, sd_spin = _local_spin_uncertainty(result.jac, n, width)
    states = pd.DataFrame({
        "spin": spin, "tilt": tilt, "speed": speed,
        "cp_rms": np.sqrt((raw ** 2).mean(axis=1)),
        "corr_spin_speed": corr, "sd_spin": sd_spin,
    }, index=batch.index)
    return InverseResult(states, int(result.nfev), time.perf_counter() - started, int(result.status))


def _local_spin_uncertainty(jac, n: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-shot correlation(spin, speed) and sd(spin) from the 3x3 block of J^T J.

    Residuals are already divided by sigma_g * sqrt(12), so each shot's cost is
    about 1 at a typical fit; the covariance is scaled by 12 to express it per
    checkpoint residual of typical size. Shots with a singular block get NaN.
    """
    jac = jac.tocsr() if hasattr(jac, "tocsr") else np.asarray(jac)
    corr = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    for i in range(n):
        block = jac[i * width:(i + 1) * width][:, [i, n + i, 2 * n + i]]
        block = block.toarray() if hasattr(block, "toarray") else block
        info = block.T @ block
        try:
            cov = np.linalg.inv(info) / 12.0
        except np.linalg.LinAlgError:
            continue
        if cov[0, 0] > 0 and cov[2, 2] > 0:
            corr[i] = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2])
            sd[i] = np.sqrt(cov[0, 0])
    return corr, sd
