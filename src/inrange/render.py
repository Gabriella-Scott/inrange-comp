"""Render a Trajectory: interactive Plotly 3D animation (HTML) and a GIF.

Scene frame (metres): D downrange from the common tee line the checkpoints
are measured from, L lateral (positive left) from the middle of the four
tees, Z height above the ground. The shot frame of inrange.trajectory has the
same axes, so a shot is placed by shifting it to its tee.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inrange.features import TEE_POSITIONS
from inrange.frame import DOWNRANGE_UNIT, LATERAL_UNIT, TEE_LINE_PROJECTION
from inrange.trajectory import Trajectory

TEE_Z: dict[str, float] = {"T1": 0.067, "T2": 0.061, "T3": 4.125, "T4": 0.054}
CHECKPOINT_LINES: list[float] = [15.0, 30.0, 45.0]
NET_D: float = 60.0
NET_HEIGHT: float = 40.0  # m, drawn only; the real net height is not given
HEIGHT_EXAGGERATION: float = 2.0  # vertical scale of the 3D scene

STYLE = {
    "radar": {"color": "#2a78d6", "width": 7, "dash": "solid", "name": "radar (measured to the net)"},
    "flight": {"color": "#eb6834", "width": 5, "dash": "dash", "name": "predicted flight"},
    "run": {"color": "#1baf7a", "width": 5, "dash": "dot", "name": "bounce and roll"},
}
GROUND, BALCONY, INK, INK_2 = "#dfe9d8", "#b9b2a3", "#0b0b0b", "#52514e"


def _tee_offsets() -> dict[str, tuple[float, float]]:
    """(D, L) of each tee in the scene frame (m)."""
    lateral = {k: np.dot(v, LATERAL_UNIT) for k, v in TEE_POSITIONS.items()}
    middle = float(np.mean(list(lateral.values())))
    return {k: (float(np.dot(v, DOWNRANGE_UNIT) - TEE_LINE_PROJECTION), lateral[k] - middle)
            for k, v in TEE_POSITIONS.items()}


TEES: dict[str, tuple[float, float]] = _tee_offsets()


def nearest_tee(x: float, y: float) -> str:
    """Tee label for a raw launch position (m)."""
    return min(TEE_POSITIONS, key=lambda k: (TEE_POSITIONS[k][0] - x) ** 2 + (TEE_POSITIONS[k][1] - y) ** 2)


@dataclass
class ScenePath:
    """A trajectory placed in the scene frame. Columns t, D, L, Z, phase."""

    tee: str
    path: pd.DataFrame
    events: dict[str, tuple[float, float, float, float]]  # name -> (t, D, L, Z)

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, tee: str) -> "ScenePath":
        d0, l0 = TEES[tee]
        z0 = trajectory.tee_z
        p = trajectory.path
        path = pd.DataFrame({"t": p["t"], "D": d0 + p["d"], "L": l0 + p["l"], "Z": z0 + p["h"],
                             "phase": p["phase"]})
        events = {name: (e["t"], d0 + e["d"], l0 + e["l"], z0 + e["h"]) for name, e in trajectory.events.items()}
        return cls(tee, path, events)

    def place(self, d: float, l: float, h: float) -> tuple[float, float, float]:
        d0, l0 = TEES[self.tee]
        return d0 + d, l0 + l, TEE_Z[self.tee] + h


def _segments(path: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Radar, flight and run pieces, each sharing its first point with the previous piece's last."""
    radar = path[path["phase"] == "radar"]
    flight = path[path["phase"] == "flight"]
    run = path[path["phase"].isin(["bounce", "roll"])]
    return {
        "radar": radar,
        "flight": pd.concat([radar.tail(1), flight]),
        "run": pd.concat([flight.tail(1), run]),
    }


def _label(trajectory: Trajectory) -> dict[str, str]:
    """Short marker labels plus a longer summary panel (plain text with <br>)."""
    pred = trajectory.prediction
    ev, dist = trajectory.events, trajectory.distances
    apex_height = ev["apex"]["h"] + trajectory.tee_z
    return {
        "apex": f"apex {apex_height:.1f} m",
        "landing": f"carry {dist['carry']:.0f} m",
        "rest": f"rest {dist['total']:.0f} m",
        "summary": "<br>".join([
            "<b>Predicted from the first 60 m</b>",
            f"launch spin: {pred['launch_spin_rate']:.0f} rpm",
            f"apex: {apex_height:.1f} m high at {ev['apex']['t']:.2f} s",
            f"carry (level landing): {dist['carry']:.1f} m at {ev['landing']['t']:.2f} s",
            f"bounce: {dist['bounce']:.1f} m ({dist['n_bounces']:.0f} hops), roll: {dist['roll']:.1f} m",
            f"at rest: {dist['total']:.1f} m from the tee after {ev['rest']['t']:.1f} s",
            f"physics state: spin {pred['spin']:.0f} rpm, axis tilt {np.degrees(pred['tilt']):+.1f} deg, "
            f"launch speed x{pred['speed']:.3f}",
        ]),
    }


def plotly_figure(trajectory: Trajectory, tee: str, truth: dict | None = None, title: str | None = None,
                  frame_step: float = 0.1, start_at_end: bool = False, camera: str = "behind the golfer"):
    """Animated Plotly 3D figure of one shot.

    truth (training shots): {"apex": (d, l, h), "landing": (d, l, h)} in the
    shot frame, shown with a toggle. frame_step: seconds of flight per frame.
    start_at_end draws the whole path before playing (for previews); camera
    picks the starting view ("behind the golfer", "side-on", "top-down").
    """
    import plotly.graph_objects as go

    scene = ScenePath.from_trajectory(trajectory, tee)
    path = scene.path
    d_max = max(path["D"].max(), 100.0) + 10
    l_span = max(path["L"].abs().max(), 25.0) + 8
    z_max = max(path["Z"].max(), 10.0) * 1.15 + 2

    traces = []
    # Ground, balcony, tees, checkpoint lines, net, distance markers.
    traces.append(go.Mesh3d(x=[-10, d_max, d_max, -10], y=[-l_span, -l_span, l_span, l_span], z=[0, 0, 0, 0],
                            i=[0, 0], j=[1, 2], k=[2, 3], color=GROUND, opacity=1.0, hoverinfo="skip",
                            name="ground", showlegend=False))
    d3, l3 = TEES["T3"]
    bx, by = [d3 - 3, d3 + 1.5, d3 + 1.5, d3 - 3], [l3 - 2.5, l3 - 2.5, l3 + 2.5, l3 + 2.5]
    traces.append(go.Mesh3d(x=bx * 2, y=by * 2, z=[TEE_Z["T3"]] * 4 + [0] * 4, alphahull=0,
                            color=BALCONY, opacity=0.55, hoverinfo="skip", name="T3 balcony", showlegend=False))
    traces.append(go.Scatter3d(x=[TEES[k][0] for k in TEES], y=[TEES[k][1] for k in TEES],
                               z=[TEE_Z[k] for k in TEES], mode="markers+text", text=list(TEES),
                               textposition="top center", marker={"size": 4, "color": INK, "symbol": "square"},
                               name="tees", hoverinfo="text", showlegend=False))
    for d in CHECKPOINT_LINES:
        traces.append(go.Scatter3d(x=[d, d], y=[-l_span, l_span], z=[0.02, 0.02], mode="lines",
                                   line={"color": INK_2, "width": 2, "dash": "dash"}, hoverinfo="skip",
                                   showlegend=False))
    traces.append(go.Mesh3d(x=[NET_D] * 4, y=[-l_span, l_span, l_span, -l_span], z=[0, 0, NET_HEIGHT, NET_HEIGHT],
                            i=[0, 0], j=[1, 2], k=[2, 3], color="#7f8c9a", opacity=0.10, hoverinfo="skip",
                            name="net", showlegend=False))
    marks = [m for m in range(50, int(d_max), 50)]
    traces.append(go.Scatter3d(x=[15, 30, 45, 60] + marks, y=[-l_span + 4] * (4 + len(marks)),
                               z=[0.3] * (4 + len(marks)), mode="text",
                               text=["15 m", "30 m", "45 m", "60 m net"] + [f"{m} m" for m in marks],
                               textfont={"color": INK_2, "size": 11}, hoverinfo="skip", showlegend=False))
    n_static = len(traces)

    # Ball path (animated) and ball.
    segments = _segments(path)
    for key, seg in segments.items():
        st = STYLE[key]
        shown = seg if start_at_end else seg.iloc[:1]
        traces.append(go.Scatter3d(x=shown["D"], y=shown["L"], z=shown["Z"], mode="lines",
                                   line={"color": st["color"], "width": st["width"], "dash": st["dash"]},
                                   name=st["name"], hoverinfo="skip"))
    ball_at = path.iloc[-1] if start_at_end else path.iloc[0]
    traces.append(go.Scatter3d(x=[ball_at["D"]], y=[ball_at["L"]], z=[ball_at["Z"]],
                               mode="markers", marker={"size": 6, "color": "white", "line": {"color": INK, "width": 2}},
                               name="ball", hoverinfo="skip", showlegend=False))
    animated = list(range(n_static, n_static + 4))

    # Event markers.
    labels = _label(trajectory)
    for key, symbol in [("apex", "diamond"), ("landing", "circle"), ("rest", "x")]:
        t, d, l, z = scene.events[key]
        traces.append(go.Scatter3d(x=[d], y=[l], z=[z], mode="markers+text", text=[labels[key]],
                                   textposition="top center", textfont={"size": 11, "color": INK},
                                   marker={"size": 5, "color": INK, "symbol": symbol},
                                   name=f"predicted {key}", hovertext=labels["summary"].replace("<br>", "; "),
                                   hoverinfo="text"))
    truth_index = None
    if truth is not None:
        tx, ty, tz, text = [], [], [], []
        for key in ["apex", "landing"]:
            x, y, z = scene.place(*truth[key])
            tx.append(x), ty.append(y), tz.append(z)
            text.append(f"true {key}")
        traces.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="markers+text", text=text, textposition="bottom center",
                                   marker={"size": 6, "color": "#e34948", "symbol": "diamond-open"},
                                   textfont={"color": "#e34948", "size": 11}, name="true apex and landing",
                                   visible=False))
        truth_index = len(traces) - 1

    # Frames: path grows with time.
    t_end = path["t"].iloc[-1]
    frame_times = np.append(np.arange(0.0, t_end, frame_step), t_end)
    frames = []
    for ft in frame_times:
        data = []
        for key, seg in segments.items():
            part = seg[seg["t"] <= ft + 1e-9]
            if part.empty:
                part = seg.iloc[:1]
            data.append(go.Scatter3d(x=part["D"], y=part["L"], z=part["Z"]))
        now = path.iloc[int(np.searchsorted(path["t"].to_numpy(), ft, side="right")) - 1]
        data.append(go.Scatter3d(x=[now["D"]], y=[now["L"]], z=[now["Z"]]))
        frames.append(go.Frame(data=data, traces=animated, name=f"{ft:.1f}"))

    fig = go.Figure(data=traces, frames=frames)
    aspect = {"x": 1.6, "y": 1.6 * 2 * l_span / (d_max + 10),
              "z": 1.6 * HEIGHT_EXAGGERATION * z_max / (d_max + 10)}
    cameras = {
        "behind the golfer": {"eye": {"x": -1.05, "y": 0.0, "z": 0.22}, "center": {"x": 0.05, "y": 0, "z": -0.12},
                              "up": {"x": 0, "y": 0, "z": 1}},
        "side-on": {"eye": {"x": 0.0, "y": -1.35, "z": 0.12}, "center": {"x": 0, "y": 0, "z": -0.08},
                    "up": {"x": 0, "y": 0, "z": 1}},
        "top-down": {"eye": {"x": 0.0, "y": 0.0, "z": 2.1}, "center": {"x": 0, "y": 0, "z": 0},
                     "up": {"x": 1, "y": 0, "z": 0}},
    }
    menus = [
        {"type": "buttons", "direction": "left", "x": 0.0, "y": 0.0, "xanchor": "left", "yanchor": "top",
         "pad": {"t": 10, "r": 10}, "showactive": False,
         "buttons": [
             {"label": "Play", "method": "animate",
              "args": [None, {"frame": {"duration": int(frame_step * 1000), "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]},
             {"label": "Pause", "method": "animate",
              "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
         ]},
        {"type": "buttons", "direction": "left", "x": 1.0, "y": 1.0, "xanchor": "right", "yanchor": "bottom",
         "showactive": True, "active": list(cameras).index(camera),
         "buttons": [{"label": name, "method": "relayout", "args": [{"scene.camera": cam}]}
                     for name, cam in cameras.items()]},
    ]
    if truth_index is not None:
        menus.append({"type": "buttons", "direction": "left", "x": 1.0, "y": 0.93, "xanchor": "right",
                      "yanchor": "bottom", "showactive": True, "active": 1,
                      "buttons": [
                          {"label": "Show truth", "method": "restyle", "args": [{"visible": True}, [truth_index]]},
                          {"label": "Hide truth", "method": "restyle", "args": [{"visible": False}, [truth_index]]},
                      ]})
    slider = {"active": len(frames) - 1 if start_at_end else 0, "x": 0.12, "len": 0.86, "y": 0.0,
              "yanchor": "top", "pad": {"t": 10}, "font": {"color": "rgba(0,0,0,0)"}, "ticklen": 0,
              "currentvalue": {"prefix": "t = ", "suffix": " s", "font": {"color": INK}},
              "steps": [{"label": f.name, "method": "animate",
                         "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                        for f in frames]}
    fig.update_layout(
        title={"text": title or f"Shot {trajectory.track_id}", "x": 0.01, "y": 0.98, "font": {"size": 15}},
        annotations=[{"text": labels["summary"], "x": 0.0, "y": 0.62, "xref": "paper", "yref": "paper",
                      "xanchor": "left", "yanchor": "top", "align": "left", "showarrow": False,
                      "font": {"size": 11, "color": INK}, "bgcolor": "rgba(255,255,255,0.75)"}],
        scene={"xaxis": {"title": "downrange (m)", "range": [-10, d_max], "backgroundcolor": "#f4f6f8"},
               "yaxis": {"title": "lateral (m, + left)", "range": [-l_span, l_span], "backgroundcolor": "#f4f6f8"},
               "zaxis": {"title": f"height (m, drawn x{HEIGHT_EXAGGERATION:g})", "range": [0, z_max],
                         "backgroundcolor": "#f4f6f8"},
               "aspectmode": "manual", "aspectratio": aspect, "camera": cameras[camera],
               "domain": {"x": [0.25, 1.0], "y": [0.06, 0.92]}},
        updatemenus=menus, sliders=[slider], margin={"l": 0, "r": 0, "t": 50, "b": 0},
        legend={"x": 0.01, "y": 0.93, "bgcolor": "rgba(255,255,255,0.7)"},
        paper_bgcolor="#fcfcfb", font={"color": INK},
    )
    return fig


def write_html(fig, path: Path) -> int:
    """Standalone HTML with plotly.js from the CDN. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, auto_play=False)
    return path.stat().st_size


def write_gif(trajectory: Trajectory, tee: str, path: Path, fps: int = 12, title: str | None = None,
              frame_step: float | None = None) -> int:
    """Animated GIF (side view above, top view below) of one shot. Returns bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    scene = ScenePath.from_trajectory(trajectory, tee)
    path_df = scene.path
    segments = _segments(path_df)
    step = frame_step or 1.0 / fps
    t_end = path_df["t"].iloc[-1]
    times = np.append(np.arange(0.0, t_end, step), [t_end] * fps)  # hold the last frame for a second

    d_max = path_df["D"].max() + 10
    l_span = max(path_df["L"].abs().max(), 20.0) + 8
    fig, (ax_side, ax_top) = plt.subplots(2, 1, figsize=(7.2, 4.6), height_ratios=[1.1, 1], sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax_side, ax_top):
        ax.set_facecolor("#fcfcfb")
        ax.axvspan(-10, NET_D, color="#2a78d6", alpha=0.05, lw=0)
        ax.axvline(NET_D, color=INK, lw=1.2)
        for d in CHECKPOINT_LINES:
            ax.axvline(d, color=INK_2, lw=0.6, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
    ax_side.fill_between([-10, d_max], -2, 0, color=GROUND)
    d3, _ = TEES["T3"]
    ax_side.add_patch(plt.Rectangle((d3 - 3, 0), 4.5, TEE_Z["T3"], color=BALCONY))
    ax_side.set_xlim(-10, d_max)
    ax_side.set_ylim(-1, path_df["Z"].max() * 1.25 + 2)
    ax_side.set_ylabel("height (m)")
    ax_top.set_ylim(-l_span, l_span)
    ax_top.set_ylabel("lateral (m, + left)")
    ax_top.set_xlabel("downrange from the tee line (m)")
    ax_side.annotate("net", (NET_D, ax_side.get_ylim()[1] * 0.95), xytext=(3, 0), textcoords="offset points",
                     fontsize=8, color=INK_2)
    ax_side.set_title(title or f"Shot {trajectory.track_id[:8]}", fontsize=11, fontweight="bold", loc="left")

    lines = {}
    for key, seg in segments.items():
        st = STYLE[key]
        ls = {"solid": "-", "dash": "--", "dot": ":"}[st["dash"]]
        lines[key] = (ax_side.plot([], [], ls, color=st["color"], lw=2.2, label=st["name"])[0],
                      ax_top.plot([], [], ls, color=st["color"], lw=2.2)[0])
    ball_side = ax_side.plot([], [], "o", color="white", mec=INK, mew=1.5, ms=7)[0]
    ball_top = ax_top.plot([], [], "o", color="white", mec=INK, mew=1.5, ms=7)[0]
    labels = _label(trajectory)
    t_apex, d_a, _, z_a = scene.events["apex"]
    t_land, d_l, l_l, z_l = scene.events["landing"]
    apex_text = ax_side.annotate(labels["apex"].replace("<br>", " "), (d_a, z_a), xytext=(0, 8),
                                 textcoords="offset points", ha="center", fontsize=8, visible=False)
    land_text = ax_side.annotate(labels["landing"].split("<br>")[0], (d_l, z_l), xytext=(0, 10),
                                 textcoords="offset points", ha="center", fontsize=8, visible=False)
    clock = ax_top.annotate("", (0.99, 0.05), xycoords="axes fraction", ha="right", fontsize=9, color=INK_2)
    ax_side.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    t_array = path_df["t"].to_numpy()

    def draw(ft):
        for key, seg in segments.items():
            part = seg[seg["t"] <= ft + 1e-9]
            lines[key][0].set_data(part["D"], part["Z"])
            lines[key][1].set_data(part["D"], part["L"])
        now = path_df.iloc[int(np.searchsorted(t_array, ft, side="right")) - 1]
        ball_side.set_data([now["D"]], [now["Z"]])
        ball_top.set_data([now["D"]], [now["L"]])
        apex_text.set_visible(ft >= t_apex)
        land_text.set_visible(ft >= t_land)
        clock.set_text(f"t = {ft:4.1f} s")
        return []

    anim = FuncAnimation(fig, draw, frames=times, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path.stat().st_size


GALLERY_CRITERIA: dict[str, str] = {
    "wedge": "ball speed below 42 m/s and predicted spin at least 8500 rpm; the one with median predicted carry",
    "mid iron": "ball speed 55 to 62 m/s, predicted spin 5000 to 7000 rpm, landing within 5 m of the launch "
                "line; median predicted carry",
    "driver": "ball speed at least 70 m/s; lowest predicted spin",
    "strong curve": "ground tee; largest predicted curve (landing lateral offset from the launch direction line)",
    "balcony": "elevated tee T3, ball speed 55 to 65 m/s; median predicted carry",
}


def select_gallery(rows: pd.DataFrame, targets: pd.DataFrame, states: pd.DataFrame) -> dict[str, str]:
    """track_id per gallery slot, by GALLERY_CRITERIA.

    rows: input rows; targets and states: HybridModel.predict_full output for
    them (same index). Shots whose checkpoint fit is in the worst 10% are
    excluded so the gallery shows well-tracked flights.
    """
    from inrange.features import ball_speed, build_features

    features = build_features(rows, use_session=False)
    info = pd.DataFrame({
        "track_id": rows["track_id"],
        "speed": ball_speed(rows),
        "spin": targets["launch_spin_rate"],
        "carry": np.hypot(targets["landing_d"], targets["landing_l"]),
        "curve": targets["landing_l"] - targets["landing_d"] * np.tan(np.radians(features["launch_direction"])),
        "elevated": rows["launch_z"] > 1.0,
        "cp_rms": states["cp_rms"],
    }, index=rows.index)
    info = info[info["cp_rms"] <= info["cp_rms"].quantile(0.9)]

    def median_carry(group: pd.DataFrame) -> str:
        return group.loc[(group["carry"] - group["carry"].median()).abs().idxmin(), "track_id"]

    ground = info[~info["elevated"]]
    return {
        "wedge": median_carry(ground[(ground["speed"] < 42) & (ground["spin"] >= 8500)]),
        "mid iron": median_carry(ground[ground["speed"].between(55, 62) & ground["spin"].between(5000, 7000)
                                        & (ground["curve"].abs() < 5)]),
        "driver": ground.loc[ground[ground["speed"] >= 70]["spin"].idxmin(), "track_id"],
        "strong curve": ground.loc[ground["curve"].abs().idxmax(), "track_id"],
        "balcony": median_carry(info[info["elevated"] & info["speed"].between(55, 65)]),
    }
