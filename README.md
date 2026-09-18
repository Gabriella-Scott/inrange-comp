# Inrange Trajectory Prediction

**Animations:** https://gabriella-scott.github.io/inrange-comp/ (GitHub Pages, served from `/docs` on `main`)

On a compact urban driving range, the radar only sees the first 60 m of each ball flight before a net stops the ball. This project predicts the rest of the flight from that opening portion alone. The inputs are the launch position and velocity plus four checkpoint crossings at 15, 30, 45 and 60 m downrange. The outputs are launch spin rate, apex position and time, and level landing position and time. The approach is a hybrid: a physics simulator inverted for spin, with a machine learning model correcting its residuals. It was built for the Inrange Student Competition on Kaggle.

## The task

Inrange installs radars at driving ranges to track every ball hit. On compact urban ranges a net stops the ball after about 60 m, so the rest of the flight has to be modelled rather than measured. This entry to the [Inrange student competition](https://www.kaggle.com/competitions/inrange-competition) predicts, from the launch conditions and four checkpoint crossings alone, the launch spin rate, the apex position and time, and the level landing position and time, and then draws the whole flight as an animation. The training data is 491 shots recorded near Stellenbosch by a full-flight radar with an in-bay launch monitor; the 559 test shots are scored on the hidden targets.

## Data

The competition data is not included in this repository and is not redistributed here. Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition's Data tab on Kaggle and place them in `data/raw/`:

```
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

## Reproduce

Requires Python 3.10.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/python make_submission.py     # trains the model; the tests below need it
.venv/bin/python -m pytest tests
```

To run the notebooks, register this environment as a Jupyter kernel and name it when executing them. Two traps are worth avoiding: the default `python3` kernel launches whichever `python` is first on PATH (which may be another environment), and `python -m jupyter nbconvert` dispatches to the `jupyter-nbconvert` on PATH rather than the one in this environment. Calling `nbconvert` as a module avoids both:

```bash
.venv/bin/python -m ipykernel install --prefix .venv --name inrange
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=inrange --ExecutePreprocessor.timeout=7200 \
    notebooks/01_eda.ipynb
```

Animations are written to `outputs/animations/`, which is gitignored: `animate_shot.py` regenerates any of them, and `docs/animations/` holds the published copies that GitHub Pages serves. Submissions are written to `outputs/submissions/` and validated against `test.csv` before they are saved: `make_submission.py` writes a dated `submission_hybrid_b_<date>.csv`, and `outputs/submissions/submission.csv` is the copy attached to the Kaggle writeup (the same predictions, to within CSV rounding). `make_submission.py` must run before `pytest`, because three trajectory tests need the trained model and skip without it. `make_submission.py` rebuilds everything from `data/raw/` in about a minute: the final model (saved to `models/final_model.joblib`, gitignored), the submission, and the per-shot physics states and coefficients in `data/processed/`; the notebooks cache their longer cross-validation fits in `data/processed/` and recompute them if the cache is missing (about 20 minutes on 8 cores). Notebooks in `notebooks/` import the `inrange` package installed by `pip install -e .`.

## Results

The competition metric is hidden. `src/inrange/scoring.py` defines our own approximation: mean landing and apex position errors (Euclidean, m), mean apex and landing time errors (s) and mean spin error (rpm), each divided by the error of predicting the training mean, then weighted 0.40 / 0.25 / 0.125 / 0.125 / 0.10. A composite of 1.0 means no better than the training mean; lower is better.

Every model is cross-validated two ways: each practice session held out in turn (leave-one-session-out, 11 folds), and random holdouts within each session matching the real test split (within-session, 10 repeats). "Test mix" reweights the speed-band results to the test set's larger share of shots above 70 m/s; its spread is a bootstrap standard deviation over shots. Component columns are unscaled overall means.

| Model | CV strategy | Composite | Test-mix composite (sd) | Landing pos. (m) | Apex pos. (m) | Apex t (s) | Landing t (s) | Spin (rpm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Training mean (`submission_mean.csv`) | leave-one-session-out | 1.016 | 1.118 (0.019) | 42.54 | 31.86 | 0.476 | 0.877 | 2073 |
| Training mean (`submission_mean.csv`) | within-session | 1.017 | 1.115 (0.018) | 42.83 | 31.92 | 0.473 | 0.867 | 2075 |
| LightGBM baseline (`submission_lgbm_baseline_2026-09-16.csv`) | leave-one-session-out | 0.185 | 0.190 (0.005) | 6.02 | 3.44 | 0.093 | 0.161 | 1051 |
| LightGBM baseline (`submission_lgbm_baseline_2026-09-16.csv`) | within-session | 0.187 | 0.197 (0.006) | 6.51 | 3.95 | 0.096 | 0.172 | 876 |
| Physics from inputs only (step 6) | leave-one-session-out | 0.198 | 0.205 (0.005) | 6.46 | 3.48 | 0.113 | 0.204 | 983 |
| Physics from inputs only (step 6) | within-session | 0.191 | 0.199 (0.006) | 6.49 | 3.45 | 0.112 | 0.197 | 871 |
| Hybrid, step 6 selection (`submission_hybrid_2026-09-17.csv`) | leave-one-session-out | 0.170 | 0.177 (0.005) | 4.99 | 2.65 | 0.094 | 0.182 | 1008 |
| Hybrid, step 6 selection (`submission_hybrid_2026-09-17.csv`) | within-session | 0.152 | 0.160 (0.005) | 4.94 | 2.66 | 0.078 | 0.149 | 846 |
| **Final hybrid** (`submission.csv`) | leave-one-session-out | 0.165 | 0.172 (0.005) | 4.99 | 2.63 | 0.086 | 0.163 | 1008 |
| **Final hybrid** (`submission.csv`) | within-session | 0.152 | 0.160 (0.005) | 4.95 | 2.64 | 0.078 | 0.149 | 846 |

**Noise floor.** Fold-to-fold spread mostly reflects how hard each held-out session is, which every model shares, so models are compared fold by fold (`scoring.compare_results`). For two near-identical models (LightGBM with and without the session feature) the paired standard error of the composite difference is 0.002 leave-one-session-out and 0.0003 within-session, against unpaired fold standard deviations of about 0.02 and 0.005. Composite differences smaller than about 0.005 (leave-one-session-out) should be treated as noise.

**Hybrid (step 6).** Each shot's spin, spin-axis tilt and launch speed factor are fitted from its checkpoints by the physics model, with a weak prior on spin from LightGBM. The simulated flight and those states then feed LightGBM, either as extra features (spin) or as a baseline whose residual LightGBM predicts (positions and times). Components were chosen per target by paired CV, and the combined model beats the LightGBM baseline by 0.011 (standard error 0.007) leave-one-session-out and 0.035 (0.001) within sessions.

**Final hybrid (step 7, the submission).** The same approach using variant b (LightGBM on the physics residual) for every position and time, and variant a (physics features) for spin. Session id was dropped because it adds nothing within sessions and hurts on new sessions. The predicted apex height is also floored at the highest measured checkpoint height. Paired against the LightGBM baseline on the full composite, the final model is better by 0.016 (standard error 0.009, 10 of 11 sessions) leave-one-session-out and 0.035 (0.001) within sessions. In the test speed mix it is better by 0.018 and 0.037 (bootstrap sd 0.005). At 70 m/s and above it gains 0.048 (0.004) within sessions, but nothing when the session is new (+0.013, standard error 0.040, 8 folds).

**Robustness to the hidden weights.** Our weights are a guess, so the saved out-of-fold predictions (`data/processed/oof_predictions.csv`) were rescored under 1000 random weight vectors that respect the stated ordering (landing position > apex position > each time > spin), and under an alternative scaling that divides each component by the spread of that target instead of by the error of predicting the training mean. The final model beat the LightGBM baseline in every one of those 4000 rescorings.

| Scaling | CV strategy | Final better | Paired difference: median (range) |
| --- | --- | --- | --- |
| Mean-prediction error (ours) | leave-one-session-out | 1000 / 1000 | -0.016 (-0.021 to -0.010) |
| Mean-prediction error (ours) | within-session | 1000 / 1000 | -0.036 (-0.039 to -0.033) |
| Target standard deviation | leave-one-session-out | 1000 / 1000 | -0.014 (-0.018 to -0.009) |
| Target standard deviation | within-session | 1000 / 1000 | -0.032 (-0.034 to -0.028) |

Differences are averages over folds of the per-fold composite, so they differ slightly from the shot-pooled numbers in the table above.

**Physics ceiling (step 5, not submittable).** A physics simulator with fitted global coefficients and the *true* launch spin, evaluated leave-one-session-out. Spin is an input, so the composite excludes the spin term; the LightGBM row is rescored the same way on the same folds.

| Model | CV strategy | Composite without spin | Test-mix (sd) | Landing pos. (m) | Apex pos. (m) | Apex t (s) | Landing t (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Physics, oracle spin | leave-one-session-out | 0.167 | 0.176 (0.006) | 7.93 | 3.91 | 0.108 | 0.213 |
| LightGBM baseline | leave-one-session-out | 0.133 | 0.141 (0.005) | 6.02 | 3.44 | 0.093 | 0.161 |

## Animating a shot

Any shot can be drawn from its 24 input columns alone: the radar segment to the net, the predicted flight beyond it, then bounce and roll. First run `make_submission.py` once to train and save the model, then:

```bash
.venv/bin/python animate_shot.py --track-id <track_id>
```

This writes `outputs/animations/shot_<track_id>.html`, a standalone interactive 3D scene (plotly.js loads from a CDN, so an internet connection is needed to view it). The scene shows:

* the four tees, with T3 on its balcony;
* the checkpoint lines, the net at 60 m and distance markers;
* the ball's path in three line styles, and labelled apex, landing and rest points.

It has play and pause controls, a time slider, and three camera presets: behind the golfer, side-on and top-down. For training shots, a button shows the true apex and landing beside the prediction. Heights are drawn at twice their true scale.

In Python:

```python
from inrange.io import load_test
from inrange.trajectory import shot_trajectory

trajectory = shot_trajectory(load_test().iloc[0])   # uses models/final_model.joblib
trajectory.path          # t, d, l, h, height, x, y, z, phase (radar, flight, bounce, roll)
trajectory.distances     # carry, bounce, roll, total (m)
```

The path is the physics simulation with the shot's fitted spin, spin-axis tilt and launch speed factor. It is time-warped and smoothly offset so that it passes exactly through the measured checkpoints and the submitted apex and landing. Bounce and roll follow Penner's (2002b) model of a golf ball's run on turf. There is no bounce data to check that against, so those distances are illustrative only. `docs/animations/` holds the published gallery of five test shots (a wedge, a mid iron, a driver, a strong curve and a balcony shot) and a training shot with the truth toggle, and `docs/trajectory_mid_iron.gif` is the animated still used in the writeup. `notebooks/04_trajectory.ipynb` explains the method and its checks.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). The licence covers the code only. The competition data is not included, not redistributed and not covered by it.
