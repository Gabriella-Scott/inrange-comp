"""Fitting the physics model to tracked shots.

* ShotBatch: arrays for many shots in the tee-centred shot frame.
* launch_consistency: does the given launch state reach the first checkpoints
  the way physics allows?
* fit_globals: global aerodynamic coefficients plus one spin-axis tilt per
  shot, with spin magnitude given (oracle spin in step 5).
* fit_tilts: tilts only, from checkpoints only, with coefficients fixed.

Residual scaling (fit_globals, fit_tilts). Residuals come in five groups per
shot: checkpoint positions at the observed cp_t (12 numbers, m), apex position
(3, m), apex time (1, s), landing position (2, m: d and l only, because
simulated landing height is 0 by construction and the true value is only the
+-0.2 m sampling noise from 01_eda.ipynb), landing time (1, s). Each residual r in group g
becomes f = r / (sigma_g * sqrt(n_g)), so every group contributes about one
unit of squared error per shot whatever its units or size. sigma_g is the
typical residual of that group: 1.4826 * median absolute residual after a pilot
fit at fixed provisional scales (PILOT_SCALES).

Robust loss. A soft_l1 loss is applied per residual number, with the count
scaling undone first so the threshold is ROBUST_THRESHOLD * sigma_g for every
number in every group (see _grouped_soft_l1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from inrange.frame import add_shot_frame
from inrange.io import CHECKPOINT_NAMES
from inrange.physics import Aero, TEXTBOOK, DEFAULT_DT, integrate, summarise_flight, simulate

GROUPS: list[str] = ["checkpoints", "apex_pos", "apex_t", "landing_pos", "landing_t"]
GROUP_SIZES: dict[str, int] = {"checkpoints": 12, "apex_pos": 3, "apex_t": 1, "landing_pos": 2, "landing_t": 1}

# Provisional typical residuals (m or s) for the pilot fit only.
PILOT_SCALES: dict[str, float] = {
    "checkpoints": 0.3, "apex_pos": 2.0, "apex_t": 0.1, "landing_pos": 3.0, "landing_t": 0.1,
}
ROBUST_THRESHOLD: float = 2.0  # soft_l1 turns linear beyond this many sigma_g

# Bounds for the global parameters (cd0, cd1, cl0, cl1, tau).
GLOBAL_LOWER = np.array([0.01, -1.0, 0.01, 0.01, 0.5])
GLOBAL_UPPER = np.array([1.0, 3.0, 3.0, 3.0, 500.0])
TILT_LIMIT: float = np.radians(80.0)
# Spin decay time (s). Not identifiable from these flights: with the other
# globals refitted, the fit cost changes by under 1% for tau from 10 s to
# 200 s (02_physics.ipynb), so tau is fixed at a typical literature value.
FIXED_TAU: float = 25.0


@dataclass
class ShotBatch:
    """Inputs (and targets, when known) for n shots in the shot frame.

    vel0 (n, 3) m/s; spin_rpm (n,); cp_t (n, 4) s; cp_pos (n, 4, 3) m.
    Targets (NaN when unknown): apex_t (n,) s, apex_pos (n, 3) m,
    landing_t (n,) s, landing_pos (n, 3) m.
    """

    index: pd.Index
    vel0: np.ndarray
    spin_rpm: np.ndarray
    cp_t: np.ndarray
    cp_pos: np.ndarray
    apex_t: np.ndarray
    apex_pos: np.ndarray
    landing_t: np.ndarray
    landing_pos: np.ndarray

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def has_targets(self) -> bool:
        return not np.isnan(self.apex_t).any()

    def subset(self, rows: np.ndarray) -> "ShotBatch":
        return ShotBatch(self.index[rows], self.vel0[rows], self.spin_rpm[rows], self.cp_t[rows],
                         self.cp_pos[rows], self.apex_t[rows], self.apex_pos[rows],
                         self.landing_t[rows], self.landing_pos[rows])


def shot_batch(df: pd.DataFrame, spin_rpm: np.ndarray | pd.Series | None = None) -> ShotBatch:
    """Build a ShotBatch from raw rows. spin_rpm defaults to launch_spin_rate."""
    frame = add_shot_frame(df)
    n = len(df)
    if spin_rpm is None:
        spin_rpm = df["launch_spin_rate"]
    nan3 = np.full((n, 3), np.nan)
    has_targets = "apex_t" in df.columns
    return ShotBatch(
        index=df.index,
        vel0=frame[["launch_vd", "launch_vl", "launch_vh"]].to_numpy(),
        spin_rpm=np.asarray(spin_rpm, dtype=float),
        cp_t=frame[[f"{cp}_t" for cp in CHECKPOINT_NAMES]].to_numpy(),
        cp_pos=np.stack([frame[[f"{cp}_d", f"{cp}_l", f"{cp}_h"]].to_numpy() for cp in CHECKPOINT_NAMES], axis=1),
        apex_t=frame["apex_t"].to_numpy() if has_targets else np.full(n, np.nan),
        apex_pos=frame[["apex_d", "apex_l", "apex_h"]].to_numpy() if has_targets else nan3,
        landing_t=frame["landing_t"].to_numpy() if has_targets else np.full(n, np.nan),
        landing_pos=frame[["landing_d", "landing_l", "landing_h"]].to_numpy() if has_targets else nan3,
    )


# ---------------------------------------------------------------------------
# Launch consistency
# ---------------------------------------------------------------------------

def launch_consistency(batch: ShotBatch, checkpoints: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    """How far the launch state misses the early checkpoints (0 = cp1, 1 = cp2).

    Drag-only textbook physics (no lift, true spin in the drag term) gives, per
    checkpoint k, the miss in crossing time (dt_k, s: simulated minus observed
    time to reach the checkpoint line) and in lateral and height position at
    the observed crossing time (dl_k, dh_k, m: simulated minus observed).

    Envelope: the same quantities for every combination of constant drag
    coefficient {0.1, 0.5}, constant lift coefficient {0, 0.4} and spin tilt
    {-45, 0, +45} degrees, which spans far more than any plausible golf ball.
    excess_t_k (s) and excess_pos_k (m, largest of lateral and height) say how
    far the observation lies outside that envelope; zero means some physics
    connects the launch state to the checkpoint.
    """
    combos = [(cd, cl, tilt) for cd in (0.1, 0.5) for cl in (0.0, 0.4) for tilt in (-45.0, 0.0, 45.0)]
    n = batch.n
    out = pd.DataFrame(index=batch.index)

    drag_only = integrate(batch.vel0, batch.spin_rpm, np.zeros(n), TEXTBOOK, lift=False,
                          min_time=batch.cp_t[:, max(checkpoints)])
    envelopes = []
    for cd, cl, tilt in combos:
        aero = Aero(cd0=cd, cd1=0.0, cl0=cl, cl1=0.0, tau=1e6)
        envelopes.append(integrate(batch.vel0, np.full(n, 3000.0), np.full(n, np.radians(tilt)), aero,
                                   min_time=batch.cp_t[:, max(checkpoints)]))

    for k in checkpoints:
        line_d = batch.cp_pos[:, k, 0]
        observed_t = batch.cp_t[:, k]
        observed = batch.cp_pos[:, k]

        def misses(flight):
            crossing = flight.first_crossing(0, line_d, rising=True)
            pos, _ = flight.state_at(observed_t)
            return crossing - observed_t, pos[:, 1] - observed[:, 1], pos[:, 2] - observed[:, 2]

        dt_k, dl_k, dh_k = misses(drag_only)
        out[f"dt_{k + 1}"], out[f"dl_{k + 1}"], out[f"dh_{k + 1}"] = dt_k, dl_k, dh_k

        env = np.array([misses(flight) for flight in envelopes])  # (combos, 3, n)
        lo, hi = env.min(axis=0), env.max(axis=0)

        def outside(i):
            return np.maximum(lo[i], 0) + np.maximum(-hi[i], 0)  # distance of 0 from [lo, hi]

        out[f"excess_t_{k + 1}"] = outside(0)
        out[f"excess_pos_{k + 1}"] = np.maximum(outside(1), outside(2))
    return out


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def _residual_blocks(batch: ShotBatch, aero: Aero, tilt: np.ndarray, with_targets: bool) -> np.ndarray:
    """Raw residuals (simulated minus observed), shape (n, 19) or (n, 12).

    Column order: 12 checkpoint numbers (cp1 d, l, h, ..., cp4 d, l, h), then
    apex d, l, h, apex t, landing d, l, landing t. Failed events are given a
    large finite residual.
    """
    min_time = batch.cp_t[:, -1]
    flight = integrate(batch.vel0, batch.spin_rpm, tilt, aero, min_time=min_time)
    summary = summarise_flight(flight, batch.cp_t)
    cp = (summary.query_pos - batch.cp_pos).reshape(batch.n, 12)
    if not with_targets:
        return cp
    blocks = np.column_stack([
        cp,
        summary.apex_pos - batch.apex_pos,
        summary.apex_t - batch.apex_t,
        summary.landing_pos[:, :2] - batch.landing_pos[:, :2],
        summary.landing_t - batch.landing_t,
    ])
    return np.nan_to_num(blocks, nan=100.0)


def _group_columns(with_targets: bool) -> dict[str, slice]:
    cols = {"checkpoints": slice(0, 12)}
    if with_targets:
        cols.update({"apex_pos": slice(12, 15), "apex_t": slice(15, 16),
                     "landing_pos": slice(16, 18), "landing_t": slice(18, 19)})
    return cols


def _column_divisors(scales: dict[str, float], with_targets: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-column divisor sigma_g * sqrt(n_g), and n_g per column."""
    columns = _group_columns(with_targets)
    width = 19 if with_targets else 12
    divisor, count = np.empty(width), np.empty(width)
    for group, cols in columns.items():
        divisor[cols] = scales[group] * np.sqrt(GROUP_SIZES[group])
        count[cols] = GROUP_SIZES[group]
    return divisor, count


def robust_group_scales(raw: np.ndarray, with_targets: bool = True) -> dict[str, float]:
    """1.4826 * median |residual| per group (m or s), from raw residual blocks."""
    return {group: float(1.4826 * np.median(np.abs(raw[:, cols])))
            for group, cols in _group_columns(with_targets).items()}


def _grouped_soft_l1(counts: np.ndarray, threshold: float):
    """soft_l1 applied per number at threshold * sigma_g, after undoing the count scaling.

    With f = r / (sigma sqrt(n)), z = f^2 and u = z n / c^2 = (r / (c sigma))^2,
    rho(z) = (c^2 / n) * 2 (sqrt(1 + u) - 1), which equals z for small z.
    """
    c2 = threshold ** 2

    def loss(z: np.ndarray) -> np.ndarray:
        u = z * counts / c2
        root = np.sqrt(1 + u)
        rho = np.empty((3, z.size))
        rho[0] = (c2 / counts) * 2 * (root - 1)
        rho[1] = 1 / root
        rho[2] = -0.5 * (counts / c2) * (1 + u) ** -1.5
        return rho

    return loss


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

@dataclass
class GlobalFit:
    """Result of fit_globals. tilt in radians, indexed like the batch."""

    aero: Aero
    tilt: pd.Series
    scales: dict[str, float]
    fit_tau: bool
    cost: float
    nfev: int
    seconds: float
    status: int
    pilot_aero: Aero | None = None
    history: list[str] = field(default_factory=list)


def _run_global_least_squares(batch: ShotBatch, start_aero: Aero, start_tilt: np.ndarray,
                              scales: dict[str, float], fit_tau: bool, max_nfev: int):
    n = batch.n
    n_glob = 5 if fit_tau else 4
    divisor, counts = _column_divisors(scales, with_targets=True)
    width = len(divisor)
    tau_fixed = start_aero.tau

    def unpack(x: np.ndarray) -> tuple[Aero, np.ndarray]:
        glob = x[:n_glob]
        values = glob if fit_tau else np.append(glob, tau_fixed)
        return Aero.from_array(values), x[n_glob:]

    def residuals(x: np.ndarray) -> np.ndarray:
        aero, tilt = unpack(x)
        return (_residual_blocks(batch, aero, tilt, True) / divisor).ravel()

    sparsity = lil_matrix((n * width, n_glob + n), dtype=int)
    sparsity[:, :n_glob] = 1
    for i in range(n):
        sparsity[i * width:(i + 1) * width, n_glob + i] = 1

    x0 = np.concatenate([start_aero.as_array()[:n_glob], start_tilt])
    lower = np.concatenate([GLOBAL_LOWER[:n_glob], np.full(n, -TILT_LIMIT)])
    upper = np.concatenate([GLOBAL_UPPER[:n_glob], np.full(n, TILT_LIMIT)])
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
    result = least_squares(residuals, x0, jac_sparsity=sparsity, bounds=(lower, upper),
                           loss=_grouped_soft_l1(np.tile(counts, n), ROBUST_THRESHOLD),
                           x_scale="jac", max_nfev=max_nfev, method="trf")
    aero, tilt = unpack(result.x)
    return aero, tilt, result


def initial_tilt(batch: ShotBatch) -> np.ndarray:
    """Rough tilt (rad) from the lateral bend over the checkpoints: positive for
    shots that curve left relative to their launch direction."""
    direction = batch.vel0[:, 1] / batch.vel0[:, 0]
    bend = batch.cp_pos[:, 3, 1] - batch.cp_pos[:, 3, 0] * direction
    return np.clip(bend / 5.0, -0.5, 0.5)


def fit_globals(batch: ShotBatch, start: Aero = TEXTBOOK, fit_tau: bool = False,
                max_nfev: int = 100, verbose: bool = False) -> GlobalFit:
    """Fit global coefficients and per-shot tilts to checkpoints, apex and landing.

    Two passes: a pilot fit with PILOT_SCALES, then residual scales re-measured
    from the pilot residuals and a final fit started from the pilot solution.
    If fit_tau is False (default), tau is held at FIXED_TAU.
    """
    if not batch.has_targets:
        raise ValueError("fit_globals needs apex and landing targets")
    started = time.perf_counter()
    history = []
    tilt0 = initial_tilt(batch)
    if not fit_tau:
        start = Aero(start.cd0, start.cd1, start.cl0, start.cl1, FIXED_TAU)
    pilot_aero, pilot_tilt, pilot = _run_global_least_squares(
        batch, start, tilt0, PILOT_SCALES, fit_tau, max_nfev)
    history.append(f"pilot: status {pilot.status}, nfev {pilot.nfev}, {pilot_aero}")
    raw = _residual_blocks(batch, pilot_aero, pilot_tilt, True)
    scales = robust_group_scales(raw)
    history.append(f"scales: {scales}")
    aero, tilt, final = _run_global_least_squares(batch, pilot_aero, pilot_tilt, scales, fit_tau, max_nfev)
    history.append(f"final: status {final.status}, nfev {final.nfev}, {aero}")
    if verbose:
        print("\n".join(history))
    return GlobalFit(aero, pd.Series(tilt, index=batch.index, name="tilt"), scales, fit_tau,
                     float(final.cost), int(pilot.nfev + final.nfev), time.perf_counter() - started,
                     int(final.status), pilot_aero, history)


def fit_tilts(batch: ShotBatch, aero: Aero, scales: dict[str, float], max_nfev: int = 50) -> pd.Series:
    """Per-shot tilt (rad) from checkpoints only, with coefficients and spin fixed."""
    n = batch.n
    divisor, counts = _column_divisors(scales, with_targets=False)

    def residuals(tilt: np.ndarray) -> np.ndarray:
        return (_residual_blocks(batch, aero, tilt, False) / divisor).ravel()

    sparsity = lil_matrix((n * 12, n), dtype=int)
    for i in range(n):
        sparsity[i * 12:(i + 1) * 12, i] = 1
    result = least_squares(residuals, initial_tilt(batch), jac_sparsity=sparsity,
                           bounds=(-TILT_LIMIT, TILT_LIMIT),
                           loss=_grouped_soft_l1(np.tile(counts, n), ROBUST_THRESHOLD),
                           x_scale="jac", max_nfev=max_nfev, method="trf")
    return pd.Series(result.x, index=batch.index, name="tilt")


def predict_shot_frame(batch: ShotBatch, aero: Aero, tilt: np.ndarray) -> pd.DataFrame:
    """Simulated apex and landing in the shot frame, as SHOT_FRAME_TARGETS
    columns (spin is passed through from the batch)."""
    summary = simulate(batch.vel0, batch.spin_rpm, np.asarray(tilt), aero, batch.cp_t)
    return pd.DataFrame({
        "launch_spin_rate": batch.spin_rpm,
        "apex_t": summary.apex_t,
        "apex_d": summary.apex_pos[:, 0],
        "apex_l": summary.apex_pos[:, 1],
        "apex_h": summary.apex_pos[:, 2],
        "landing_t": summary.landing_t,
        "landing_d": summary.landing_pos[:, 0],
        "landing_l": summary.landing_pos[:, 1],
    }, index=batch.index)


def implied_speed_factor(batch: ShotBatch, checkpoint: int, iterations: int = 20) -> np.ndarray:
    """Factor on the launch velocity that makes drag-only textbook physics cross
    checkpoint ``checkpoint`` (0 = cp1) at its observed time. Bisection on
    [0.7, 1.3]; 1.0 means the launch speed and checkpoint timing agree."""
    k = checkpoint
    lower, upper = np.full(batch.n, 0.7), np.full(batch.n, 1.3)
    for _ in range(iterations):
        middle = (lower + upper) / 2
        flight = integrate(batch.vel0 * middle[:, None], batch.spin_rpm, np.zeros(batch.n), TEXTBOOK,
                           lift=False, min_time=batch.cp_t[:, k])
        early = flight.first_crossing(0, batch.cp_pos[:, k, 0], rising=True) < batch.cp_t[:, k]
        upper = np.where(early, middle, upper)
        lower = np.where(early, lower, middle)
    return (lower + upper) / 2


def tau_profile(batch: ShotBatch, reference: GlobalFit, taus: list[float], n_jobs: int = -1) -> pd.DataFrame:
    """Refit the other globals (and tilts) at each fixed tau (s), using the
    reference fit's residual scales and solution as the start. Returns the
    robust cost and fitted globals per tau."""
    from joblib import Parallel, delayed

    def one(tau: float) -> dict:
        start = Aero(reference.aero.cd0, reference.aero.cd1, reference.aero.cl0, reference.aero.cl1, tau)
        aero, _, result = _run_global_least_squares(batch, start, reference.tilt.to_numpy(),
                                                    reference.scales, False, 100)
        return {"tau": tau, "cost": float(result.cost), **vars(aero)}

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(float(tau)) for tau in taus)
    return pd.DataFrame(rows).set_index("tau")
