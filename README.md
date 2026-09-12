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
.venv/bin/python make_submission.py
```

Submissions are written to `outputs/submissions/` and validated against `test.csv` before they are saved.

## Results

| Submission | Model | CV composite | Landing pos. (m) | Apex pos. (m) | Apex t (s) | Landing t (s) | Spin (rpm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
