"""Checks for the stitched trajectory and the bounce model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inrange.io import CHECKPOINT_NAMES, RAW_DIR, load_test, load_train
from inrange.trajectory import MODEL_PATH, PENNER, bounce_and_roll, impact, restitution, shot_trajectories, shot_trajectory

needs_model = pytest.mark.skipif(not (MODEL_PATH.exists() and (RAW_DIR / "test.csv").exists()),
                                 reason="run make_submission.py first to train the final model")


def test_restitution_matches_penner() -> None:
    assert restitution(0.0) == pytest.approx(0.510)
    assert restitution(10.0) == pytest.approx(0.510 - 0.375 + 0.0903)
    assert restitution(25.0) == pytest.approx(0.120)
    assert restitution(-10.0) == restitution(10.0)


@pytest.mark.parametrize("speed, angle, spin, height", [(26.2, 46.2, 315.0, 1.17), (26.8, 68.0, 224.0, 2.82)])
def test_first_bounce_matches_penner_drives(speed: float, angle: float, spin: float, height: float) -> None:
    # Penner (2002b, p. 936): first bounce heights for drives landing at these speeds and angles.
    u, down = speed * np.sin(np.radians(angle)), speed * np.cos(np.radians(angle))
    _, up, _, rolled = impact(u, down, spin)
    assert rolled
    assert up ** 2 / (2 * 9.81) == pytest.approx(height, abs=0.02)


def test_run_without_spin_moves_forward_and_stops() -> None:
    run = bounce_and_roll(np.array([100.0, 0.0, 0.0]), np.array([15.0, 0.0, -18.0]), 0.0, 6.0)
    assert run.rest_position[0] > 100.0
    assert run.rest_time > 6.0
    assert (run.path["h"] >= -1e-9).all()
    assert set(run.path["phase"]) <= {"bounce", "roll"}


@pytest.fixture(scope="module")
def trajectories():
    train, test = load_train(), load_test()
    rows = pd.concat([train[test.columns].sample(12, random_state=1), test.sample(12, random_state=1)])
    return rows.reset_index(drop=True), shot_trajectories(rows)


@needs_model
def test_path_hits_checkpoints_apex_and_landing(trajectories) -> None:
    rows, trajs = trajectories
    from inrange.frame import add_shot_frame

    frame = add_shot_frame(rows)
    for i, tj in enumerate(trajs):
        path = tj.path
        for cp in CHECKPOINT_NAMES:
            point = path.loc[(path["t"] - frame.at[i, f"{cp}_t"]).abs().idxmin()]
            assert point["t"] == pytest.approx(frame.at[i, f"{cp}_t"], abs=1e-3)
            for axis in "dlh":
                assert point[axis] == pytest.approx(frame.at[i, f"{cp}_{axis}"], abs=0.01)
        for name, target_t, keys in [("apex", "apex_t", "dlh"), ("landing", "landing_t", "dl")]:
            point = path.loc[(path["t"] - tj.prediction[target_t]).abs().idxmin()]
            assert point["t"] == pytest.approx(tj.prediction[target_t], abs=1e-3)
            for axis in keys:
                assert point[axis] == pytest.approx(tj.prediction[f"{name}_{axis}"], abs=0.01)


@needs_model
def test_apex_is_highest_and_ball_stays_above_ground(trajectories) -> None:
    _, trajs = trajectories
    for tj in trajs:
        path = tj.path
        assert path["h"].max() <= tj.events["apex"]["h"] + 0.01
        airborne = path[(path["t"] > 0.05) & (path["t"] < tj.events["landing"]["t"])]
        assert (airborne["height"] > 0).all()
        assert (path["height"] >= -1e-6).all()
        assert path["t"].is_monotonic_increasing


@needs_model
def test_single_row_uses_only_input_columns() -> None:
    row = load_train().iloc[0]  # includes target columns, which must be ignored
    tj = shot_trajectory(row)
    assert tj.track_id == row["track_id"]
    assert list(tj.path["phase"].unique()) == ["radar", "flight", "bounce", "roll"]
