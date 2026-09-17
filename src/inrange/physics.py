"""Golf ball flight simulator in each shot's tee-centred frame.

Frame (see inrange.frame): d downrange, l lateral (positive left), h height,
right-handed (d x l = h), origin at the tee. Gravity acts along -h. Level
landing is where h returns to 0 on the descent.

Equation of motion (per unit mass, SI units):

    dv/dt = -g h_hat - k |v| (C_D v - C_L (s_hat x v)),   k = rho A / (2 m)

Aerodynamics, with spin ratio S = omega r / |v| and omega in rad/s:

    C_D = cd0 + cd1 S
    C_L = cl0 S^cl1
    omega(t) = omega_0 exp(-t / tau)

Spin interface: rpm outside this module, rad/s inside (RPM_TO_RAD_S).

Spin axis convention. The unit axis is fixed in space and set by one tilt
angle per shot (radians):

    s_hat = (0, -cos(tilt), sin(tilt))      in (d, l, h)

tilt = 0 is pure backspin: for a ball moving downrange, s_hat x v points up.
Positive tilt adds a +l component to s_hat x v, so the ball curves LEFT
(towards positive lateral); negative tilt curves it right.

Air density is fixed at RHO and not fitted: it multiplies every coefficient,
so it cannot be separated from them.

Integration uses a vectorised fixed-step RK4 over all shots at once
(``simulate``). Events and states at arbitrary times are found by cubic
Hermite interpolation inside a step, never by snapping to the grid.
``simulate_reference`` integrates one shot with scipy's solve_ivp and event
functions, for testing.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
from scipy.integrate import solve_ivp

MASS: float = 0.04593  # kg
DIAMETER: float = 0.04267  # m
RADIUS: float = DIAMETER / 2  # m
AREA: float = np.pi * DIAMETER ** 2 / 4  # m^2
RHO: float = 1.2  # kg/m^3, fixed
G: float = 9.81  # m/s^2
K_AERO: float = RHO * AREA / (2 * MASS)  # 1/m

RPM_TO_RAD_S: float = 2 * np.pi / 60

# s. Against solve_ivp (rtol = atol = 1e-11) on training shots, including
# extreme coefficients, 0.02 s agrees within 0.001 mm and 0.001 ms at events and
# checkpoints (0.05 s: 0.02 mm); tests/test_physics.py enforces 1 cm and 1 ms.
DEFAULT_DT: float = 0.02
DEFAULT_T_MAX: float = 12.0  # s; longest training flight lands at 7.9 s


@dataclass(frozen=True)
class Aero:
    """Global aerodynamic parameters (dimensionless, except tau in seconds)."""

    cd0: float
    cd1: float
    cl0: float
    cl1: float
    tau: float

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "Aero":
        return cls(*(float(v) for v in values))


# Starting values quoted in CLAUDE.md; to be replaced by fitted values.
TEXTBOOK = Aero(cd0=0.21, cd1=0.18, cl0=0.54, cl1=0.4, tau=25.0)


def rpm_to_rad_s(rpm: np.ndarray | float) -> np.ndarray:
    """Revolutions per minute to radians per second."""
    return np.asarray(rpm, dtype=float) * RPM_TO_RAD_S


def spin_axis(tilt: np.ndarray) -> np.ndarray:
    """Unit spin axes, shape (n, 3) in (d, l, h), from tilt angles in radians."""
    tilt = np.asarray(tilt, dtype=float)
    return np.stack([np.zeros_like(tilt), -np.cos(tilt), np.sin(tilt)], axis=-1)


def acceleration(t: float | np.ndarray, vel: np.ndarray, omega0: np.ndarray,
                 axis: np.ndarray, aero: Aero, lift: bool = True) -> np.ndarray:
    """Acceleration (m/s^2), shape (n, 3), for velocities (m/s) at time t (s).

    omega0 is launch spin in rad/s, shape (n,); axis has shape (n, 3).
    lift=False removes the Magnus term (drag and gravity only).
    """
    speed = np.sqrt(np.einsum("ij,ij->i", vel, vel))
    omega = omega0 * np.exp(-t / aero.tau)
    spin_ratio = omega * RADIUS / speed
    drag = aero.cd0 + aero.cd1 * spin_ratio
    acc = -(K_AERO * speed * drag)[:, None] * vel
    if lift:
        lift_coef = aero.cl0 * spin_ratio ** aero.cl1
        acc += (K_AERO * speed * lift_coef)[:, None] * np.cross(axis, vel)
    acc[:, 2] -= G
    return acc


@dataclass
class Flight:
    """Stored RK4 states for n shots on a fixed grid of step dt (s).

    pos, vel, acc have shape (steps + 1, n, 3) in m, m/s, m/s^2 (shot frame).
    """

    dt: float
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray

    @property
    def n_shots(self) -> int:
        return self.pos.shape[1]

    @property
    def t_end(self) -> float:
        return self.dt * (self.pos.shape[0] - 1)

    def state_at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position (m) and velocity (m/s), shape (n, 3), at one time (s) per shot."""
        times = np.asarray(times, dtype=float)
        step = np.clip(np.floor(times / self.dt).astype(int), 0, self.pos.shape[0] - 2)
        frac = times / self.dt - step
        return self._hermite(step, frac)

    def first_crossing(self, component: int, target: np.ndarray | float, rising: bool,
                       use_velocity: bool = False, after_step: np.ndarray | int = 0) -> np.ndarray:
        """Time (s) of each shot's first crossing of ``target`` in one component.

        component: 0, 1 or 2 for d, l, h. use_velocity=True looks at velocity
        instead of position. rising selects upward (True) or downward crossings.
        Only steps from ``after_step`` onwards are searched. Shots that never
        cross return NaN.
        """
        series = (self.vel if use_velocity else self.pos)[:, :, component]
        target = np.broadcast_to(np.asarray(target, dtype=float), (self.n_shots,))
        shifted = series - target[None, :]
        if rising:
            crosses = (shifted[:-1] < 0) & (shifted[1:] >= 0)
        else:
            crosses = (shifted[:-1] > 0) & (shifted[1:] <= 0)
        steps = np.arange(crosses.shape[0])[:, None]
        crosses &= steps >= np.broadcast_to(np.asarray(after_step), (self.n_shots,))[None, :]
        found = crosses.any(axis=0)
        step = np.argmax(crosses, axis=0)

        cols = np.arange(self.n_shots)
        y0, y1 = shifted[step, cols], shifted[step + 1, cols]
        frac = np.where(y1 != y0, y0 / (y0 - y1), 0.5)
        for _ in range(6):  # Newton on the Hermite cubic
            value, slope = self._hermite_component(step, frac, component, use_velocity)
            frac = np.clip(frac - (value - target) / np.where(slope != 0, slope, 1.0), 0.0, 1.0)
        times = (step + frac) * self.dt
        return np.where(found, times, np.nan)

    def _hermite(self, step: np.ndarray, frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.arange(self.n_shots)
        s = frac[:, None]
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        p0, p1 = self.pos[step, cols], self.pos[step + 1, cols]
        v0, v1 = self.vel[step, cols], self.vel[step + 1, cols]
        a0, a1 = self.acc[step, cols], self.acc[step + 1, cols]
        dt = self.dt
        pos = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1
        vel = h00 * v0 + h10 * dt * a0 + h01 * v1 + h11 * dt * a1
        return pos, vel

    def _hermite_component(self, step: np.ndarray, frac: np.ndarray, component: int,
                           use_velocity: bool) -> tuple[np.ndarray, np.ndarray]:
        """Value and d(value)/d(frac) of one interpolated component."""
        cols = np.arange(self.n_shots)
        if use_velocity:
            y, dy = self.vel, self.acc
        else:
            y, dy = self.pos, self.vel
        y0, y1 = y[step, cols, component], y[step + 1, cols, component]
        m0, m1 = dy[step, cols, component] * self.dt, dy[step + 1, cols, component] * self.dt
        s = frac
        value = ((2 * s**3 - 3 * s**2 + 1) * y0 + (s**3 - 2 * s**2 + s) * m0
                 + (-2 * s**3 + 3 * s**2) * y1 + (s**3 - s**2) * m1)
        slope = ((6 * s**2 - 6 * s) * y0 + (3 * s**2 - 4 * s + 1) * m0
                 + (-6 * s**2 + 6 * s) * y1 + (3 * s**2 - 2 * s) * m1)
        return value, slope


def integrate(vel0: np.ndarray, spin_rpm: np.ndarray, tilt: np.ndarray, aero: Aero,
              dt: float = DEFAULT_DT, t_max: float = DEFAULT_T_MAX, lift: bool = True,
              min_time: np.ndarray | float = 0.0) -> Flight:
    """RK4 for all shots at once from the tee (position 0).

    vel0: launch velocity (m/s), shape (n, 3) in the shot frame.
    spin_rpm: launch spin (rpm), shape (n,). tilt: spin axis tilt (rad), shape (n,).
    Stops at t_max, or earlier once every shot is below launch height,
    descending, and past min_time (s, per shot).
    """
    vel = np.array(vel0, dtype=float)
    n = vel.shape[0]
    omega0 = rpm_to_rad_s(spin_rpm)
    axis = spin_axis(tilt)
    min_time = np.broadcast_to(np.asarray(min_time, dtype=float), (n,))
    n_steps = int(round(t_max / dt))

    pos_hist = np.empty((n_steps + 1, n, 3))
    vel_hist = np.empty((n_steps + 1, n, 3))
    acc_hist = np.empty((n_steps + 1, n, 3))
    pos = np.zeros((n, 3))
    pos_hist[0], vel_hist[0] = pos, vel

    def f(t: float, v: np.ndarray) -> np.ndarray:
        return acceleration(t, v, omega0, axis, aero, lift)

    last = n_steps
    for i in range(n_steps):
        t = i * dt
        k1 = f(t, vel)
        acc_hist[i] = k1
        k2 = f(t + dt / 2, vel + dt / 2 * k1)
        k3 = f(t + dt / 2, vel + dt / 2 * k2)
        k4 = f(t + dt, vel + dt * k3)
        # Position uses the matching RK4 stages for x' = v.
        pos = pos + dt * (vel + dt / 6 * (k1 + k2 + k3))
        vel = vel + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pos_hist[i + 1], vel_hist[i + 1] = pos, vel
        if i % 25 == 24:
            done = (pos[:, 2] < 0) & (vel[:, 2] < 0) & (min_time <= t + dt)
            if done.all():
                last = i + 1
                break
    acc_hist[last] = f(last * dt, vel_hist[last])
    return Flight(dt, pos_hist[: last + 1], vel_hist[: last + 1], acc_hist[: last + 1])


@dataclass
class FlightSummary:
    """Events and requested states for n shots (shot frame; m, s).

    apex_t, landing_t: shape (n,). apex_pos, landing_pos: shape (n, 3).
    query_pos: shape (n, k, 3), positions at the requested times.
    landing_vel: shape (n, 3), velocity at level landing (m/s).
    """

    apex_t: np.ndarray
    apex_pos: np.ndarray
    landing_t: np.ndarray
    landing_pos: np.ndarray
    landing_vel: np.ndarray
    query_pos: np.ndarray


def summarise_flight(flight: Flight, query_times: np.ndarray | None = None) -> FlightSummary:
    """Apex (vertical velocity falls through 0), level landing (height falls
    through 0 after apex) and positions at query_times (s, shape (n, k))."""
    apex_t = flight.first_crossing(2, 0.0, rising=False, use_velocity=True)
    apex_pos, _ = flight.state_at(np.nan_to_num(apex_t))
    apex_step = np.floor(np.nan_to_num(apex_t) / flight.dt).astype(int)
    landing_t = flight.first_crossing(2, 0.0, rising=False, after_step=apex_step)
    landing_pos, landing_vel = flight.state_at(np.nan_to_num(landing_t))

    if query_times is None:
        query_pos = np.zeros((flight.n_shots, 0, 3))
    else:
        query_times = np.asarray(query_times, dtype=float)
        query_pos = np.stack([flight.state_at(query_times[:, k])[0] for k in range(query_times.shape[1])],
                             axis=1)

    missing_apex = np.isnan(apex_t)
    missing_landing = np.isnan(landing_t)
    apex_pos[missing_apex] = np.nan
    landing_pos[missing_landing] = np.nan
    landing_vel[missing_landing] = np.nan
    return FlightSummary(apex_t, apex_pos, landing_t, landing_pos, landing_vel, query_pos)


def simulate(vel0: np.ndarray, spin_rpm: np.ndarray, tilt: np.ndarray, aero: Aero,
             query_times: np.ndarray | None = None, dt: float = DEFAULT_DT,
             t_max: float = DEFAULT_T_MAX, lift: bool = True) -> FlightSummary:
    """Integrate and summarise in one call (see integrate and summarise_flight)."""
    min_time = 0.0 if query_times is None else np.max(query_times, axis=1)
    flight = integrate(vel0, spin_rpm, tilt, aero, dt, t_max, lift, min_time)
    return summarise_flight(flight, query_times)


def simulate_reference(vel0: np.ndarray, spin_rpm: float, tilt: float, aero: Aero,
                       query_times: np.ndarray | None = None, lift: bool = True,
                       rtol: float = 1e-11, atol: float = 1e-11) -> dict[str, np.ndarray]:
    """One shot with solve_ivp (DOP853) and event functions, for testing.

    Returns apex_t, apex_pos, landing_t, landing_pos, and query_pos (k, 3).
    """
    omega0 = np.array([rpm_to_rad_s(spin_rpm)])
    axis = spin_axis(np.array([tilt]))

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        acc = acceleration(t, y[None, 3:], omega0, axis, aero, lift)[0]
        return np.concatenate([y[3:], acc])

    def apex(t: float, y: np.ndarray) -> float:
        return y[5]
    apex.direction = -1

    def landing(t: float, y: np.ndarray) -> float:
        return y[2] if y[5] < 0 else 1.0
    landing.direction = -1
    landing.terminal = True

    y0 = np.concatenate([np.zeros(3), np.asarray(vel0, dtype=float)])
    sol = solve_ivp(rhs, (0.0, DEFAULT_T_MAX), y0, method="DOP853", rtol=rtol, atol=atol,
                    events=[apex, landing], dense_output=True)
    out = {
        "apex_t": sol.t_events[0][0],
        "apex_pos": sol.y_events[0][0][:3],
        "landing_t": sol.t_events[1][0],
        "landing_pos": sol.y_events[1][0][:3],
    }
    if query_times is not None:
        out["query_pos"] = np.array([sol.sol(t)[:3] for t in query_times])
    return out
