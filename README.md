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

Submissions are written to `outputs/submissions/` and validated against `test.csv` before they are saved. Notebooks in `notebooks/` import the `inrange` package installed by `pip install -e .`.

## Results

The competition metric is hidden. `src/inrange/scoring.py` defines our own approximation: mean landing and apex position errors (Euclidean, m), mean apex and landing time errors (s) and mean spin error (rpm), each divided by the error of predicting the training mean, then weighted 0.40 / 0.25 / 0.125 / 0.125 / 0.10. A composite of 1.0 means no better than the training mean; lower is better.

Every model is cross-validated two ways: whole practice sessions held out (session-grouped, 5 folds), and random holdouts within each session matching the real test split (within-session, 10 repeats). "Test mix" reweights the speed-band results to the test set's larger share of shots above 70 m/s. Component columns are unscaled overall means.

| Model | CV strategy | Composite | Test-mix composite | Landing pos. (m) | Apex pos. (m) | Apex t (s) | Landing t (s) | Spin (rpm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Training mean (benchmark) | session-grouped | 1.006 | 1.107 | 42.28 | 31.54 | 0.465 | 0.857 | 2089 |
| Training mean (benchmark) | within-session | 1.017 | 1.115 | 42.83 | 31.92 | 0.473 | 0.867 | 2075 |
