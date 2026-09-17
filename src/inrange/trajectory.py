"""Full ball path from one input row: radar segment, predicted flight, bounce, roll.

Frame: tee-centred shot frame from inrange.frame (d downrange, l lateral
positive left, h height above the tee, all in metres; t in seconds from
launch). The ground is at h = -launch_z; for the elevated tee T3 that is
4.125 m below the level landing.

Stitching (flight). The base path is the physics simulation with the shot's
fitted spin, tilt and speed factor (inrange.hybrid.HybridModel). Two
corrections make it pass through what we know:

* Time warp: piecewise linear and increasing, mapping the physics times at
  cp1..cp4, apex and level landing onto the observed cp_t and the predicted
  apex_t and landing_t (slope of the last segment continues after landing).
* Position offset: per coordinate, a cubic Hermite curve in physics time with
  PCHIP slopes, anchored at 0 at launch, observed minus physics at cp1..cp4,
  predicted minus physics at apex and landing, held constant after landing.
  The height offset's slope at the apex cancels the physics vertical velocity
  there, so the drawn apex is a stationary point; all slopes are 0 at landing.
  Where the result would still rise above the predicted apex, the height is
  capped at the apex (noted on the trajectory).

Bounce and roll (after the ball reaches the ground) follow Penner's (2002b)
run model, in the vertical plane of the ball's horizontal motion:

* Turf compliance: each impact is resolved in a frame rotated by
  theta_c = 15.4 deg (v / 18.6 m/s)(phi / 44.4 deg), with v the impact speed
  and phi the impact angle from the vertical (p. 934, eqs. 6 to 8).
* Restitution for the normal impact speed v_n (m/s):
  e = 0.510 - 0.0375 v_n + 0.000903 v_n^2 for v_n <= 20 m/s, else 0.120
  (p. 933, eq. 5).
* Tangential rebound (p. 933, eqs. 2 to 4, after Daish): the ball rolls out
  of the impact when the friction coefficient mu (0.40, p. 933) exceeds
  mu_c = 2 (u + r w) / (7 (1 + e) v_n); then u' = (5/7) u - (2/7) r w and
  w' = -u'/r. Otherwise it slides: u' = u - mu v_n (1 + e) and
  w' = w - 5 mu v_n (1 + e) / (2 r). Here u is the tangential speed, w the
  backspin (rad/s) and r the ball radius.
* Bouncing stops when the rebound height is below 5 mm (p. 935).
* Rolling deceleration (5/7) rho_g g with rho_g = 0.131 (p. 935). That is a
  golf-green value (range 0.065 to 0.196; Penner, 2002a, p. 85); Penner notes
  a fairway would be expected to have a larger value (p. 935).

Simplifications: flat ground, no air drag during hops, and the backspin at
impact is the model's launch spin decayed with tau and projected with
cos(tilt). There is no bounce or roll data in this competition, so none of
this is fitted or validated against our shots.

References
Penner, A. R. (2002a). The physics of putting. Canadian Journal of Physics, 80, 83-96. https://doi.org/10.1139/p01-137
Penner, A. R. (2002b). The run of a golf ball. Canadian Journal of Physics, 80(8), 931-940. https://doi.org/10.1139/p02-035
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from inrange.frame import add_shot_frame, from_shot_frame
from inrange.io import CHECKPOINT_NAMES, INPUT_COLS, REPO_ROOT
from inrange.physics import G, RADIUS, Flight, integrate, rpm_to_rad_s, summarise_flight

MODEL_PATH: Path = REPO_ROOT / "models" / "final_model.joblib"
PHASES: list[str] = ["radar", "flight", "bounce", "roll"]
KNOTS: list[str] = ["launch"] + CHECKPOINT_NAMES + ["apex", "landing"]


@dataclass(frozen=True)
class BounceParams:
    """Bounce and roll parameters (sources in the module docstring).

    restitution_scale multiplies Penner's e(v_n); theta_scale multiplies his
    turf-compliance angle theta_c; friction is mu; rolling_retention is the
    5/7 factor on the tangential speed when the ball rolls out of an impact;
    use_spin switches the backspin terms; rolling_friction is rho_g;
    stop_height (m) ends bouncing.
    """

    restitution_scale: float = 1.0
    theta_scale: float = 1.0
    friction: float = 0.40
    rolling_retention: float = 5.0 / 7.0
    use_spin: bool = True
    rolling_friction: float = 0.131
    stop_height: float = 0.005
    max_bounces: int = 50


PENNER = BounceParams()


def restitution(v_normal: float, scale: float = 1.0) -> float:
    """Penner's (2002b, p. 933) coefficient of restitution for a normal impact speed (m/s)."""
    v = abs(v_normal)
    e = 0.510 - 0.0375 * v + 0.000903 * v ** 2 if v <= 20.0 else 0.120
    return scale * e


def impact(u: float, v_down: float, spin: float, params: BounceParams = PENNER) -> tuple[float, float, float, bool]:
    """One turf impact in the plane of motion (Penner, 2002b, pp. 933-934).

    u: horizontal speed (m/s, positive forwards), v_down: downward speed
    (m/s, positive), spin: backspin (rad/s). Returns the rebound horizontal
    speed, upward speed, backspin, and whether the ball rolled out of the impact.
    """
    mirror = u < 0  # solve a backwards-moving ball as its mirror image
    if mirror:
        u, spin = -u, -spin
    speed = np.hypot(u, v_down)
    phi = np.degrees(np.arctan2(u, v_down))
    theta = np.radians(params.theta_scale * 15.4 * (speed / 18.6) * (phi / 44.4))
    u_t = u * np.cos(theta) - v_down * np.sin(theta)
    v_n = u * np.sin(theta) + v_down * np.cos(theta)
    e = restitution(v_n, params.restitution_scale)
    w = spin if params.use_spin else 0.0
    mu_c = 2 * (u_t + RADIUS * w) / (7 * (1 + e) * v_n)
    rolled = params.friction >= mu_c
    if rolled:
        u_r = params.rolling_retention * u_t - (2.0 / 7.0) * RADIUS * w
        w_r = -u_r / RADIUS
    else:
        u_r = u_t - params.friction * v_n * (1 + e)
        w_r = w - 5 * params.friction * v_n * (1 + e) / (2 * RADIUS)
    v_r = e * v_n
    u_out = u_r * np.cos(theta) - v_r * np.sin(theta)
    v_out = u_r * np.sin(theta) + v_r * np.cos(theta)
    if mirror:
        u_out, w_r = -u_out, -w_r
    return float(u_out), float(v_out), float(w_r), bool(rolled)


@dataclass
class Run:
    """Bounce and roll after the first ground impact. Distances in m (horizontal)."""

    path: pd.DataFrame
    bounce_distance: float
    roll_distance: float
    n_bounces: int
    rest_time: float
    rest_position: np.ndarray
    first_bounce_height: float


def bounce_and_roll(position: np.ndarray, velocity: np.ndarray, backspin: float, t0: float,
                    params: BounceParams = PENNER, dt: float = 0.02) -> Run:
    """Bounces then roll from a ground impact.

    position (m) and velocity (m/s) in the shot frame at impact, backspin
    (rad/s, positive = backspin) and impact time t0 (s). The ground height is
    position[2]. Returns a path with columns t, d, l, h, phase.
    """
    ground = float(position[2])
    horizontal = np.array([velocity[0], velocity[1]], dtype=float)
    speed = float(np.hypot(*horizontal))
    direction = horizontal / speed if speed > 0 else np.array([1.0, 0.0])
    along, down, spin = speed, abs(float(velocity[2])), float(backspin)
    here = np.array(position[:2], dtype=float)
    t = t0
    rows = []
    bounce_distance, n_bounces, first_height = 0.0, 0, 0.0

    for _ in range(params.max_bounces):
        along, up, spin, _ = impact(along, down, spin, params)
        height = up ** 2 / (2 * G)
        if n_bounces == 0:
            first_height = height
        if height < params.stop_height:
            break
        n_bounces += 1
        hop_time = 2 * up / G
        steps = max(int(np.ceil(hop_time / dt)), 2)
        for s in np.linspace(0, hop_time, steps + 1)[1:]:
            xy = here + direction * along * s
            rows.append((t + s, xy[0], xy[1], ground + up * s - 0.5 * G * s ** 2, "bounce"))
        here = here + direction * along * hop_time
        bounce_distance += abs(along) * hop_time
        t += hop_time
        down = up

    decel = (5.0 / 7.0) * params.rolling_friction * G
    roll_time = abs(along) / decel
    roll_distance = along ** 2 / (2 * decel)
    steps = max(int(np.ceil(roll_time / dt)), 1)
    for s in np.linspace(0, roll_time, steps + 1)[1:]:
        travelled = along * s - np.sign(along) * 0.5 * decel * s ** 2
        xy = here + direction * travelled
        rows.append((t + s, xy[0], xy[1], ground, "roll"))
    rest = here + direction * np.sign(along) * roll_distance
    path = pd.DataFrame(rows, columns=["t", "d", "l", "h", "phase"])
    return Run(path, bounce_distance, roll_distance, n_bounces, t + roll_time,
               np.array([rest[0], rest[1], ground]), first_height)


@dataclass
class Trajectory:
    """A shot's full path and what it was built from.

    path: columns t (s), d, l, h (m, shot frame), height (m above ground),
    x, y, z (m, raw range frame), phase (radar, flight, bounce, roll).
    events: apex, landing (level), impact (ground), rest; each a dict with t,
    d, l, h. distances: carry (level landing), ground_carry, bounce, roll,
    total (horizontal distance tee to rest), all in m.
    knots: per knot, physics time, drawn time, and the position offset (m).
    prediction: final model targets (shot frame) plus fitted states.
    impact_velocity: drawn velocity (m/s, shot frame) at ground contact;
    impact_backspin: backspin (rad/s) at ground contact.
    """

    track_id: str
    tee_z: float
    path: pd.DataFrame
    events: dict[str, dict[str, float]]
    distances: dict[str, float]
    knots: pd.DataFrame
    prediction: pd.Series
    impact_velocity: np.ndarray
    impact_backspin: float
    notes: list[str] = field(default_factory=list)


def load_model(path: Path = MODEL_PATH):
    """The trained HybridModel saved by make_submission.py."""
    from inrange.hybrid import HybridModel

    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: run make_submission.py first to train and save the model")
    return HybridModel.load(path)


def _as_input_frame(rows: pd.DataFrame | pd.Series) -> pd.DataFrame:
    frame = rows.to_frame().T if isinstance(rows, pd.Series) else rows
    missing = [c for c in INPUT_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"input rows are missing columns: {missing}")
    frame = frame[INPUT_COLS].copy()
    numeric = [c for c in INPUT_COLS if c != "track_id"]
    frame[numeric] = frame[numeric].astype(float)
    return frame.reset_index(drop=True)


def shot_trajectory(row: pd.Series | pd.DataFrame, model=None, bounce: BounceParams = PENNER,
                    dt: float = 0.02) -> Trajectory:
    """Full path for one shot from its 24 input columns (as in test.csv) only."""
    frame = _as_input_frame(row)
    if len(frame) != 1:
        raise ValueError("shot_trajectory takes exactly one row; use shot_trajectories for several")
    return shot_trajectories(frame, model, bounce, dt)[0]


def shot_trajectories(rows: pd.DataFrame, model=None, bounce: BounceParams = PENNER,
                      dt: float = 0.02) -> list[Trajectory]:
    """shot_trajectory for many rows, sharing one inverse solve and simulation."""
    model = load_model() if model is None else model
    inputs = _as_input_frame(rows)
    prediction = model.predict_full(inputs)
    frame = add_shot_frame(inputs)
    states = prediction.states
    aero = model.fit_result.aero

    vel0 = frame[["launch_vd", "launch_vl", "launch_vh"]].to_numpy() * states["speed"].to_numpy()[:, None]
    cp_t = frame[[f"{cp}_t" for cp in CHECKPOINT_NAMES]].to_numpy()
    tee_z = inputs["launch_z"].to_numpy()
    flight = integrate(vel0, states["spin"].to_numpy(), states["tilt"].to_numpy(), aero, t_max=15.0,
                       min_time=cp_t[:, -1], stop_height=-tee_z - 0.05)
    summary = summarise_flight(flight)
    cp_d = frame[[f"{cp}_d" for cp in CHECKPOINT_NAMES]].to_numpy()
    tau_cp = np.column_stack([flight.first_crossing(0, cp_d[:, k], rising=True) for k in range(4)])
    apex_step = np.floor(np.nan_to_num(summary.apex_t) / flight.dt).astype(int)
    tau_ground = flight.first_crossing(2, -tee_z, rising=False, after_step=apex_step)

    out = []
    for i in range(len(inputs)):
        single = Flight(flight.dt, flight.pos[:, [i]], flight.vel[:, [i]], flight.acc[:, [i]])
        observed = np.array([[frame.at[i, f"{cp}_{a}"] for a in "dlh"] for cp in CHECKPOINT_NAMES])
        target = prediction.targets.iloc[i]
        out.append(_stitch(
            track_id=str(inputs.at[i, "track_id"]),
            flight=single,
            tau=np.concatenate([[0.0], tau_cp[i], [summary.apex_t[i], summary.landing_t[i]]]),
            tau_ground=float(tau_ground[i]),
            times=np.concatenate([[0.0], cp_t[i], [target["apex_t"], target["landing_t"]]]),
            positions=np.vstack([np.zeros(3), observed,
                                 [target["apex_d"], target["apex_l"], target["apex_h"]],
                                 [target["landing_d"], target["landing_l"], 0.0]]),
            origin=inputs.loc[i, ["launch_x", "launch_y", "launch_z"]].to_numpy(dtype=float),
            backspin0=float(rpm_to_rad_s(states["spin"].iloc[i]) * np.cos(states["tilt"].iloc[i])),
            tau_decay=aero.tau,
            prediction=pd.concat([target, states.iloc[i][["spin", "tilt", "speed", "cp_rms"]]]),
            bounce=bounce,
            dt=dt,
        ))
    return out


def _stitch(track_id: str, flight: Flight, tau: np.ndarray, tau_ground: float, times: np.ndarray,
            positions: np.ndarray, origin: np.ndarray, backspin0: float, tau_decay: float,
            prediction: pd.Series, bounce: BounceParams, dt: float) -> Trajectory:
    notes = []
    tee_z = origin[2]
    physics_pos = np.array([flight.state_at(np.array([s]))[0][0] for s in tau])
    physics_vel = np.array([flight.state_at(np.array([s]))[1][0] for s in tau])
    offsets = positions - physics_pos
    offsets[0] = 0.0
    knots = pd.DataFrame({"knot": KNOTS, "tau": tau, "t": times,
                          "offset_d": offsets[:, 0], "offset_l": offsets[:, 1], "offset_h": offsets[:, 2]})

    order = np.argsort(tau)
    if not np.all(np.diff(times[order]) > 0):
        # The physics apex falls on the other side of a checkpoint from the
        # predicted apex: move the apex knot to where the warp without it
        # puts the predicted apex time.
        keep = [k for k in order if KNOTS[k] != "apex"]
        tau_apex = float(np.interp(times[5], times[keep], tau[keep]))
        notes.append(f"apex knot moved from physics time {tau[5]:.3f} s to {tau_apex:.3f} s "
                     "to keep the time warp increasing")
        tau = tau.copy()
        tau[5] = tau_apex
        pos, vel = flight.state_at(np.array([tau_apex]))
        physics_pos[5], physics_vel[5] = pos[0], vel[0]
        offsets[5] = positions[5] - physics_pos[5]
        knots.loc[5, ["tau", "offset_d", "offset_l", "offset_h"]] = [tau_apex, *offsets[5]]
        order = np.argsort(tau)
    if tau[5] < tau[4]:
        notes.append("apex before the net (cp4)")

    tau_sorted, t_sorted, off_sorted = tau[order], times[order], offsets[order]
    slope_last = (t_sorted[-1] - t_sorted[-2]) / (tau_sorted[-1] - tau_sorted[-2])

    def warp(s: np.ndarray) -> np.ndarray:
        inside = np.interp(s, tau_sorted, t_sorted)
        return np.where(s > tau_sorted[-1], t_sorted[-1] + (s - tau_sorted[-1]) * slope_last, inside)

    splines = []
    apex_pos_in_order = int(np.where(order == 5)[0][0])
    for c in range(3):
        slopes = PchipInterpolator(tau_sorted, off_sorted[:, c]).derivative()(tau_sorted)
        slopes[-1] = 0.0
        if c == 2:
            slopes[apex_pos_in_order] = -physics_vel[5, 2]
        splines.append(CubicHermiteSpline(tau_sorted, off_sorted[:, c], slopes))

    grid = np.union1d(np.arange(0.0, tau_ground, dt / 2), np.append(tau, tau_ground))
    grid = grid[grid <= tau_ground]
    pos, _ = flight.state_at(grid)
    offset = np.column_stack([np.where(grid <= tau_sorted[-1], sp(np.minimum(grid, tau_sorted[-1])),
                                       off_sorted[-1, c]) for c, sp in enumerate(splines)])
    drawn = pos + offset
    drawn[-1, 2] = -tee_z  # exact ground contact
    # The stitched curve can rise above the predicted apex when that apex sits
    # just above a checkpoint the ball passes while still climbing; cap it so
    # the predicted apex stays the highest point (a short flat top).
    excess = drawn[:, 2].max() - positions[5, 2]
    if excess > 1e-9:
        drawn[:, 2] = np.minimum(drawn[:, 2], positions[5, 2])
        notes.append(f"height capped at the predicted apex (by up to {excess:.3f} m)")
    t_drawn = warp(grid)
    phase = np.where(grid <= tau[4], "radar", "flight")
    flight_path = pd.DataFrame({"t": t_drawn, "d": drawn[:, 0], "l": drawn[:, 1], "h": drawn[:, 2], "phase": phase})

    _, impact_vel = flight.state_at(np.array([tau_ground]))
    impact_velocity = impact_vel[0] / slope_last
    backspin = backspin0 * np.exp(-tau_ground / tau_decay)
    run = bounce_and_roll(drawn[-1], impact_velocity, backspin, float(t_drawn[-1]), bounce, dt)

    path = pd.concat([flight_path, run.path], ignore_index=True)
    path["height"] = path["h"] + tee_z
    x, y, z = from_shot_frame(path["d"], path["l"], path["h"], *origin)
    path["x"], path["y"], path["z"] = x, y, z
    path = path[["t", "d", "l", "h", "height", "x", "y", "z", "phase"]]

    def event(t, p):
        return {"t": float(t), "d": float(p[0]), "l": float(p[1]), "h": float(p[2])}

    events = {
        "apex": event(times[5], positions[5]),
        "landing": event(times[6], positions[6]),
        "impact": event(t_drawn[-1], drawn[-1]),
        "rest": event(run.rest_time, run.rest_position),
    }
    distances = {
        "carry": float(np.hypot(*positions[6][:2])),
        "ground_carry": float(np.hypot(*drawn[-1][:2])),
        "bounce": run.bounce_distance,
        "roll": run.roll_distance,
        "total": float(np.hypot(*run.rest_position[:2])),
        "n_bounces": float(run.n_bounces),
        "first_bounce_height": run.first_bounce_height,
    }
    return Trajectory(track_id, float(tee_z), path, events, distances, knots, prediction,
                      impact_velocity, float(backspin), notes)


def with_bounce(trajectory: Trajectory, params: BounceParams) -> dict[str, float]:
    """Bounce, roll and total distance (m) for the same flight with other bounce parameters."""
    impact = trajectory.events["impact"]
    position = np.array([impact["d"], impact["l"], impact["h"]])
    run = bounce_and_roll(position, trajectory.impact_velocity, trajectory.impact_backspin, impact["t"], params)
    return {"bounce": run.bounce_distance, "roll": run.roll_distance,
            "total": float(np.hypot(*run.rest_position[:2])), "first_bounce_height": run.first_bounce_height}
