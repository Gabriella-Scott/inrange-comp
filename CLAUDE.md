# CLAUDE.md — Inrange Student Competition

Context and working rules for Claude Code on this repo. Read this fully before acting.

---

## 1. What this project is

A Kaggle private community hackathon run by **Inrange**, a golf radar and range-technology company based in Stellenbosch. Entry is part of a graduate software developer application, so code quality and the written explanation matter as much as the score.

**Deadline: 19 September 2026, 17:00 GMT+2.**

### The problem

Radar on a compact urban driving range only sees the first 60 m of a ball flight before a net stops the ball. Given only that opening portion, predict the rest of the flight.

**Inputs** (available for both train and test):

| Column | Meaning |
| --- | --- |
| `track_id` | Unique identifier (UUID string) |
| `launch_time` | Absolute Unix/POSIX timestamp of the shot |
| `launch_x/y/z` | Launch position, metres |
| `launch_vx/vy/vz` | Launch velocity, metres per second |
| `cp1_t/x/y/z` | Crossing of the 15 m downrange line |
| `cp2_t/x/y/z` | Crossing of the 30 m line |
| `cp3_t/x/y/z` | Crossing of the 45 m line |
| `cp4_t/x/y/z` | Crossing of the 60 m line (the net) |

**Targets** (train only, nine values to predict):

`launch_spin_rate` (rpm), `apex_t`, `apex_x`, `apex_y`, `apex_z`, `landing_t`, `landing_x`, `landing_y`, `landing_z`.

### Things that are easy to get wrong

- The range heading is **diagonal** across the x-y plane. Carry shows up as change in both x and y. Nothing is axis-aligned.
- `z` is height above ground. Larger z is higher.
- `apex_t` and `landing_t` are **durations in seconds from that shot's own launch**, starting at 0. `launch_time` is wall-clock and is a different kind of quantity entirely. Never mix them.
- Landing is defined as the **level landing point**: where the ball first returns to launch height for that shot, interpolated from the tracked flight. It is not ground contact.
- Spin is in **revolutions per minute**, not radians per second. Convert carefully in the physics code.
- `cp*_t` values are interpolated crossing times, not raw radar sample times.

### Data on disk

- `data/raw/train.csv` — 491 rows, 33 columns
- `data/raw/test.csv` — 559 rows, 24 columns
- `data/raw/sample_submission.csv` — 559 rows, 10 columns, every target prefilled with the training-set mean

Roughly 1050 shots total, split near-evenly and mostly at random.

### Scoring

A single hidden weighted composite. Landing position is weighted most heavily, then apex position, then apex and landing times, then launch spin. Position errors are Euclidean distances in metres, times in seconds, spin in rpm, each scaled before combining. Lower is better. The exact weights are held back, so we build our own approximation and always report components separately.

### Rubric (100 points)

| Criterion | Points |
| --- | --- |
| Prediction accuracy | 50 |
| Approach (how training data was used) | 10 |
| Data analysis and visualisation | 10 |
| Novelty | 5 |
| Display of a full animated trajectory from inputs only | 20 |
| Bounce and roll considered and shown | 5 |
| Writeup under 3000 words and fewer than 25 graphics | Pass/fail |

The animation is worth 20 points. It is not an afterthought.

### Submission mechanism

There is no live leaderboard. The submission is a **Kaggle Writeup** with:

1. An attached `submission.csv` (one row per test `track_id`, the nine target columns filled in, `track_id` unchanged)
2. A link to a public notebook or public repo
3. Optionally a project link or a YouTube video of three minutes or less

A draft writeup already exists on the Kaggle account. Anything private that is attached becomes public after the deadline.

---

## 2. Approach

Hybrid: physics forward model, per-track inverse solve for spin, then machine learning on the residuals. A pure-ML baseline is built first as a benchmark and a safety net.

Rationale: 491 training rows is too few for a model to learn projectile motion, drag and the Magnus effect from scratch, and a pure-ML model cannot extrapolate outside the shots it has seen. Physics supplies all of that. More importantly, the first 60 m encodes spin implicitly (backspin slows the vertical fall relative to a drag-only parabola, sidespin bends the path laterally), so a simulator lets us invert for spin directly rather than hoping a tree ensemble infers it.

### The physics

Single ODE, integrated with `scipy.integrate.solve_ivp`:

```
dv/dt = -g * z_hat - (rho * A / (2 * m)) * |v| * (C_D * v - C_L * (s_hat x v))
```

Constants for a golf ball:

- mass `m = 0.04593` kg
- diameter `d = 0.04267` m, so area `A = pi * d^2 / 4 ≈ 1.43e-3` m²
- air density `rho ≈ 1.2` kg/m³ (Stellenbosch is near sea level; treat as a fittable parameter)
- `g = 9.81` m/s²
- `s_hat` is the unit spin axis

Coefficients depend on spin ratio `S = omega * r / |v|`. Textbook starting points are `C_D ≈ 0.21 + 0.18 * S` and `C_L ≈ 0.54 * S^0.4`, but **these must be fitted to our training data, not trusted**. Add exponential spin decay `omega(t) = omega_0 * exp(-t / tau)` with `tau` in the region of 20 to 30 seconds, also fitted.

Apex and landing come from `solve_ivp` event functions: `vz = 0` for apex, `z = launch_z` on the descent for landing. Do not detect them by scanning a fixed time grid.

### The inverse solve

Per track, unknowns are spin magnitude and spin axis orientation (two or three parameters). Residuals are the twelve numbers from four checkpoint positions. Use `scipy.optimize.least_squares`. It is heavily over-determined, so it should converge quickly and robustly. Validate on train by comparing fitted spin against the true `launch_spin_rate`.

### Then ML

A small gradient-boosting model predicts the physics model's residuals for apex and landing, using launch features, checkpoint-derived features and the fitted spin.

---

## 3. Conventions

### Code

- Python, `numpy` / `pandas` / `scipy` / `scikit-learn` / `matplotlib` / `plotly`, LightGBM for the boosted models.
- Reusable logic lives in `src/inrange/`, notebooks import from it. Notebooks are for narrative and figures, not for hiding logic.
- Type hints on public functions. Docstrings that state units. Units are the main source of bugs in this project.
- Prefer explicit, readable code over clever vectorisation tricks.

### Working style

- **Step by step.** Finish and verify one step before starting the next. Do not run ahead to later steps even if they seem obvious.
- At the end of each step, report what was done, what the numbers were, and what is uncertain. Then stop and wait.
- Never state a result that has not actually been computed and printed. No estimated or remembered numbers.
- Prefer targeted edits over rewriting whole files.

### Writing

- South African English spelling (modelling, visualise, metres, analyse).
- No em-dashes.
- Concise paragraphs.
- APA author-date citations where sources are used, with page numbers where applicable.

### Git

- One commit per completed step, present-tense imperative message.
- Every generated submission is committed to `outputs/submissions/` with a dated filename, and the commit message carries the CV score.
- `data/raw/` and `data/processed/` are gitignored.

### Repo layout

```
inrange-trajectory/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # gitignored
│   └── processed/        # gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_physics.ipynb
│   └── 03_modelling.ipynb
├── src/inrange/
│   ├── __init__.py
│   ├── io.py             # loading, column groups
│   ├── frame.py          # rotation into downrange / lateral / up
│   ├── physics.py        # forward simulator
│   ├── inverse.py        # spin fitting from checkpoints
│   ├── features.py
│   ├── models.py
│   └── scoring.py        # composite metric and CV
├── outputs/
│   ├── figures/
│   └── submissions/
└── report/
    └── writeup.md
```

---

## 4. Plan

Each step ends with a working artefact. Do not start step N+1 until step N is verified.

1. **Repo setup and format proof.** Structure, environment, loaders, a script that writes a valid submission. Detail below.
2. **EDA and frame rotation.** Determine the range heading, rotate into downrange / lateral / up, check whether `landing_z` is simply `launch_z`, cluster `launch_time` into practice sessions, produce the core exploratory figures.
3. **Scoring harness.** Composite metric approximating the hidden one, plus session-grouped cross-validation.
4. **Pure-ML baseline.** Engineered features, gradient boosting per target, first real submission.
5. **Physics simulator.** Forward model with fitted aerodynamic coefficients.
6. **Inverse spin solve.** Fit spin per track, validate against known spin on train.
7. **Hybrid residual model.** ML correction on top of physics, final predictions.
8. **Trajectory animation.** Full flight from inputs only.
9. **Bounce and roll.** Simple restitution and friction model.
10. **Writeup and submission.**

---

## 5. Step 1 — the only step to work on right now

**Goal: a valid `submission.csv` produced by our own code, from a clean repo, with no modelling at all.**

Tasks:

1. Create the folder structure above. Add `.gitkeep` files where needed so empty directories survive.
2. Write `.gitignore` covering `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/raw/*`, `data/processed/*`, `.DS_Store`, with a negation for the `.gitkeep` files.
3. Create the virtual environment and install `numpy pandas scipy scikit-learn matplotlib plotly jupyter lightgbm`. Freeze to `requirements.txt`.
4. Copy the three CSVs into `data/raw/`.
5. Write `src/inrange/io.py` with:
   - constants for the input and target column names
   - `load_train()`, `load_test()`, `load_sample_submission()`
   - a `validate_submission(df, test_df)` function that checks row count, exact column names and order, that every test `track_id` appears exactly once, that no `track_id` has been altered, and that there are no NaNs or infinities
6. Write `make_submission.py` at the repo root: reads `data/raw/test.csv`, calls a `predict(rows)` function, writes `outputs/submissions/submission_baseline_zero.csv`. The initial `predict` returns zeros for all nine targets. Plain standard library plus pandas, no model. This is the placeholder that later steps replace.
7. Run it. Run `validate_submission` against the output. Print the validation result.
8. Also produce `outputs/submissions/submission_mean.csv`, which is just `sample_submission.csv` copied through the same validation path. This is the mean-prediction fallback and a slightly less embarrassing safety submission than zeros.
9. Write a first-pass `README.md`: one-paragraph problem statement, how to obtain the data, how to reproduce, and a results table with headers and no rows yet.
10. `git init -b main`, commit, and stop.

**Do not** load the training targets, compute statistics, fit anything or make plots in this step. The point is purely to prove the format end to end.

**Report back with:** the validation output, the first three rows of the zero submission, the file tree, and anything about the data that blocked you.