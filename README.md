# Inrange Trajectory Prediction

On a compact urban driving range, the radar only sees the first 60 m of each ball flight before a net stops the ball. This project predicts the rest of the flight from that opening portion alone. The inputs are the launch position and velocity plus four checkpoint crossings at 15, 30, 45 and 60 m downrange. The outputs are launch spin rate, apex position and time, and level landing position and time. The approach is a hybrid: a physics simulator inverted for spin, with a machine learning model correcting its residuals. It was built for the Inrange Student Competition on Kaggle.

## Data

The data is not included in this repository. Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition's Data tab on Kaggle and place them in `data/raw/`:

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
.venv/bin/python -m pytest tests
.venv/bin/python make_submission.py
```

Submissions are written to `outputs/submissions/` and validated against `test.csv` before they are saved. `make_submission.py` fits the physics model and the hybrid on all training rows, which takes a few minutes; the notebooks cache their longer cross-validation fits in `data/processed/` and recompute them if the cache is missing (about 20 minutes on 8 cores). Notebooks in `notebooks/` import the `inrange` package installed by `pip install -e .`.

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
| Hybrid (`submission_hybrid_2026-09-17.csv`) | leave-one-session-out | 0.170 | 0.177 (0.005) | 4.99 | 2.65 | 0.094 | 0.182 | 1008 |
| Hybrid (`submission_hybrid_2026-09-17.csv`) | within-session | 0.152 | 0.160 (0.005) | 4.94 | 2.66 | 0.078 | 0.149 | 846 |

**Noise floor.** Fold-to-fold spread mostly reflects how hard each held-out session is, which every model shares, so models are compared fold by fold (`scoring.compare_results`). For two near-identical models (LightGBM with and without the session feature) the paired standard error of the composite difference is 0.002 leave-one-session-out and 0.0003 within-session, against unpaired fold standard deviations of about 0.02 and 0.005. Composite differences smaller than about 0.005 (leave-one-session-out) should be treated as noise.

**Hybrid (step 6).** Each shot's spin, spin-axis tilt and launch speed factor are fitted from its checkpoints by the physics model, with a weak prior on spin from LightGBM. The simulated flight and those states then feed LightGBM, either as extra features (spin) or as a baseline whose residual LightGBM predicts (positions and times). Components were chosen per target by paired CV, and the combined model beats the LightGBM baseline by 0.011 (standard error 0.007) leave-one-session-out and 0.035 (0.001) within sessions.

**Physics ceiling (step 5, not submittable).** A physics simulator with fitted global coefficients and the *true* launch spin, evaluated leave-one-session-out. Spin is an input, so the composite excludes the spin term; the LightGBM row is rescored the same way on the same folds.

| Model | CV strategy | Composite without spin | Test-mix (sd) | Landing pos. (m) | Apex pos. (m) | Apex t (s) | Landing t (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Physics, oracle spin | leave-one-session-out | 0.167 | 0.176 (0.006) | 7.93 | 3.91 | 0.108 | 0.213 |
| LightGBM baseline | leave-one-session-out | 0.133 | 0.141 (0.005) | 6.02 | 3.44 | 0.093 | 0.161 |
