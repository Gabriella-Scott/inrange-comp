"""Write an interactive 3D animation of one shot, from its input columns only.

Usage:
    .venv/bin/python animate_shot.py --track-id <uuid> [--out path.html]

Works for any track_id in data/raw/train.csv or data/raw/test.csv. For
training shots the true apex and landing can be toggled on beside the
prediction. Needs the trained model from make_submission.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inrange.features import ball_speed
from inrange.frame import add_shot_frame
from inrange.io import INPUT_COLS, REPO_ROOT, load_test, load_train
from inrange.render import nearest_tee, plotly_figure, write_html
from inrange.trajectory import MODEL_PATH, load_model, shot_trajectory

ANIMATIONS_DIR: Path = REPO_ROOT / "outputs" / "animations"


def find_shot(track_id: str):
    """(input row, truth dict or None, split name) for a track_id."""
    train = load_train()
    match = train[train["track_id"] == track_id]
    if len(match):
        frame = add_shot_frame(match).iloc[0]
        truth = {"apex": (frame["apex_d"], frame["apex_l"], frame["apex_h"]),
                 "landing": (frame["landing_d"], frame["landing_l"], frame["landing_h"])}
        return match.iloc[0][INPUT_COLS], truth, "train"
    test = load_test()
    match = test[test["track_id"] == track_id]
    if len(match):
        return match.iloc[0][INPUT_COLS], None, "test"
    raise SystemExit(f"track_id {track_id} is not in train.csv or test.csv")


def animate(track_id: str, out: Path | None = None, model_path: Path = MODEL_PATH, title: str | None = None) -> Path:
    """Build and write the HTML animation for one track_id; returns the path."""
    row, truth, split = find_shot(track_id)
    trajectory = shot_trajectory(row, load_model(model_path))
    tee = nearest_tee(row["launch_x"], row["launch_y"])
    speed = float(ball_speed(row.to_frame().T.astype({c: float for c in INPUT_COLS if c != "track_id"})).iloc[0])
    title = title or (f"{split} shot {track_id[:8]} from tee {tee}: ball speed {speed:.1f} m/s, "
                      f"carry {trajectory.distances['carry']:.0f} m, total {trajectory.distances['total']:.0f} m")
    fig = plotly_figure(trajectory, tee, truth=truth, title=title)
    out = Path(out) if out else ANIMATIONS_DIR / f"shot_{track_id}.html"
    size = write_html(fig, out)
    print(f"wrote {out} ({size / 1e6:.2f} MB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track-id", required=True, help="track_id from train.csv or test.csv")
    parser.add_argument("--out", type=Path, help="output HTML path (default outputs/animations/shot_<id>.html)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="trained model from make_submission.py")
    args = parser.parse_args()
    animate(args.track_id, args.out, args.model)


if __name__ == "__main__":
    main()
