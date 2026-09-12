# Inrange Student Competition — SPEC

## Dataset Description

### Inrange Student Competition Data

#### Overview

These shots were recorded on a real outdoor golf driving range in the region of Stellenbosch, South Africa, where two things happened at once: the radar had a clear view of the entire flight -- launch, apex and landing -- and a high-quality launch monitor measured the spin of each shot as it was struck. That combination is rare: for every shot we know both the opening portion of the flight and the truth about how it finished.

The goal is to reproduce that knowledge on an ordinary range that has neither luxury -- no launch monitor, and a net a short way downrange that stops each ball long before it lands. On such a range a tracker only ever sees the first stretch of the flight. Your job is to recover the rest of the story from that opening portion alone: the spin imparted at launch (which the radar never measures directly), the apex the ball reaches, and where and when it comes down (both hidden behind the net).

Each recorded shot is called a track: one ball's flight, sampled by the radar. In this data every track has been trimmed to just the part a live system would see -- from launch to a set of downrange checkpoints ending at the 60 m net -- so the models you build here would work unchanged on a netted range.

#### The Problem

Build a model that, given only the launch conditions and the four checkpoint crossings of a track, predicts:

Launch spin rate
Apex position and time
Landing position and time
How you get there -- physics-based modelling, machine learning, or a hybrid -- is entirely up to you.

#### Data Description

##### Coordinate system

Units are written out in full below to avoid any ambiguity:

Position is in metres (m).
Velocity is in metres per second (m/s).
Spin is in revolutions per minute (rpm) -- revolutions, not radians, and per minute, not per second.
Time is in seconds (s).

Positions are given in a fixed coordinate frame local to the range. The x and y axes lie in the flat, horizontal ground plane and z is height above the ground (larger z is higher). Crucially, the range does not point along either axis: the downrange direction is a fixed compass heading that cuts diagonally across the x-y plane, so a shot's carry shows up as changes in both x and y, not in one axis alone. The plan view below (looking straight down from above, z coming up out of the page) shows the layout:

```
        y (m)
          ^                              * landing
          |                          .'
          |                      .'
          |                 cp4 (net)
          |               .' cp3
          |            .' cp2      downrange heading
          |         .' cp1        (fixed, diagonal)
          |      .'
     tee  o---.'--------------------------------> x (m)
   (launch)
```

Times are in seconds measured from each track's own launch (so every track starts at t = 0).

##### Files

| File | Description |
| --- | --- |
| train.csv | Full data: launch conditions, checkpoints, AND known spin/apex/landing |
| test.csv | Launch conditions + checkpoints only -- predict the rest |
| sample_submission.csv | Template showing the submission format |
| make_submission.py | Runnable starter script that writes a valid submission |

##### Input columns (present in both train.csv and test.csv)

| Column | Description |
| --- | --- |
| track_id | Unique identifier |
| launch_time | Launch timestamp (Unix/POSIX time -- seconds since 1 Jan 1970) |
| launch_x/y/z | Launch position (m) |
| launch_vx/vy/vz | Launch velocity (m/s) |
| cp1_t, cp1_x, cp1_y, cp1_z | Checkpoint at 15m |
| cp2_t, cp2_x, cp2_y, cp2_z | Checkpoint at 30m |
| cp3_t, cp3_x, cp3_y, cp3_z | Checkpoint at 45m |
| cp4_t, cp4_x, cp4_y, cp4_z | Checkpoint at 60m (netting) |

##### Target columns (present in train.csv only -- predict these)

| Column | Description |
| --- | --- |
| launch_spin_rate | Launch spin rate (rpm) |
| apex_t, apex_x, apex_y, apex_z | Apex time (s) and position (m) |
| landing_t, landing_x, landing_y, landing_z | Landing time (s) and position (m) |

#### Evaluation

Submissions are scored by a single weighted composite of the prediction errors. Landing position is weighted most heavily -- it is what matters most in practice -- followed by apex position, then the apex and landing times, with launch spin contributing the least. Position errors are Euclidean distances in metres, timing errors are in seconds and spin error is in rpm; each is scaled so the components are comparable before they are combined. Lower is better. The exact weightings are held back with the scoring key.

#### Submission Format

A submission is a CSV with one row per track_id in test.csv and the target columns filled in:

```
track_id,launch_spin_rate,apex_t,apex_x,apex_y,apex_z,landing_t,landing_x,landing_y,landing_z
0f2a1c9d-...,3200,4.1,88.0,47.0,22.5,7.9,171.2,92.4,0.1
...
```

sample_submission.csv is exactly this file with each target column pre-filled with the training-set mean; overwrite those columns with your predictions and leave track_id unchanged.

To get started quickly, run the included make_submission.py (plain Python, no extra packages). It reads test.csv and writes a valid submission with every prediction set to zero -- proof the format is right end to end. Replace the zeros in its predict() function with your model and re-run.

#### Notes

There are around 1050 shots in total, divided into the training and test sets by a near-even mostly random split.
"Level landing position" is where the ball first returns to launch height (the tee is the reference height for that shot); landing_t is the time from launch to that point, interpolated from the tracked flight.
Each cp*_t is the interpolated time the ball crossed that downrange line, not a raw sample time; the checkpoint x, y, z is the crossing position.
launch_time is an absolute wall-clock timestamp (Unix/POSIX time -- seconds since 1 Jan 1970), whereas apex_t and landing_t are durations in seconds measured from each shot's own launch.

---

## sample_submission.csv (extract)

sample_submission.csv (113.57 kB) — 10 of 10 columns, 559 unique track_id values. Every target column is pre-filled with a single constant (the training-set mean), so all 559 rows are identical apart from track_id.

Column constants:

| Column | Value |
| --- | --- |
| launch_spin_rate | 6107.306697991963 |
| apex_t | 2.869938900203666 |
| apex_x | 70.49173456203758 |
| apex_y | 80.13298935996255 |
| apex_z | 23.974071261129613 |
| landing_t | 5.519042769857434 |
| landing_x | 121.24631957147443 |
| landing_y | 105.01091796934331 |
| landing_z | 1.201555650888779 |

First few rows as an example:

```
track_id,launch_spin_rate,apex_t,apex_x,apex_y,apex_z,landing_t,landing_x,landing_y,landing_z
065c1384-11e8-44d2-bf1a-db288c0ed636,6107.306697991963,2.869938900203666,70.49173456203758,80.13298935996255,23.974071261129613,5.519042769857434,121.24631957147443,105.01091796934331,1.201555650888779
a97c3116-70e1-40b2-afd4-1a00b7147363,6107.306697991963,2.869938900203666,70.49173456203758,80.13298935996255,23.974071261129613,5.519042769857434,121.24631957147443,105.01091796934331,1.201555650888779
c2dd368c-3e5a-4ed9-8e6e-db8551dd47ac,6107.306697991963,2.869938900203666,70.49173456203758,80.13298935996255,23.974071261129613,5.519042769857434,121.24631957147443,105.01091796934331,1.201555650888779
dc80735e-d8e3-4716-a20b-73c189f9895d,6107.306697991963,2.869938900203666,70.49173456203758,80.13298935996255,23.974071261129613,5.519042769857434,121.24631957147443,105.01091796934331,1.201555650888779
9552ec97-ca7d-447f-8d11-5b17bd3dd31d,6107.306697991963,2.869938900203666,70.49173456203758,80.13298935996255,23.974071261129613,5.519042769857434,121.24631957147443,105.01091796934331,1.201555650888779
...
```

Data Explorer: 600.12 kB — 3 files (sample_submission.csv, test.csv, train.csv), 67 columns.

---

## Competition Brief

A compact urban driving range stops the ball in a net after 60 metres, so the rest of the flight has to be modelled rather than measured. Your goal is to use measured data to build that model and show its output as a realistic, animated ball flight.

### Description

At Inrange we aim to transform the way people experience golf at a driving range. We install radars at driving ranges that allow us to track the flight of all the golf balls hit on the range. This data can then be used by keen players on the apps we develop to track their progress and improve. We also enable ranges to be transformed into an entertainment space where golf driven games can be enjoyed by everyone. This technology has been installed and is in operation on ranges all around the world.

This competition is about using measured data to predict the flight of a golf ball, also known as its trajectory. Driving ranges come in all shapes and sizes and for many ranges measuring data from the launch to landing (the way we prefer to operate) is simply not possible. Ranges in urban areas are typically surrounded by netting and we need to use modelling to predict how a ball would fly after it hits the net, as if the netting was not present. In some of the most constrained ranges the amount of flight we see might be limited to just the first 60m. An example of such a range is Swing City in Australia.

Your task in this competition is to build a model using real measured data that can accurately predict such a trajectory. The input to the model will be what is seen at such a netted range: launch conditions plus four checkpoint crossings we have added at 15 m, 30 m, 45 m and 60 m — that last one being the net. From that alone, we would like to see how accurately you can predict launch spin rate, apex position and time, and landing position and time.

While the checkpoint positions are somewhat artificial of the provided data is based on real measurements. Your training data consists of 492 shots recorded near Stellenbosch, South Africa. The shots were recorded on an open range with an unobstructed full-flight radar and an in-bay launch monitor system. For every shot you are provided with the spin, apex and landing, alongside the trimmed view a netted range would have had.

### What we want you to build

Two things, weighted roughly equally.

* Predictions for the test set, submitted as a CSV and scored at the end of the competition
* A writeup (or a link in the write-up to a notebook) that explains your approach, presents your analysis of the data and how you predicted the data

### What we provide

* A detailed description of the data
* The full training data set with inputs and outputs (train.csv in the data section)
* The test set with inputs only against which your predictions will be scored (test.csv in the data section)
* A sample submission showing how the provided data needs to look to be scored (submission_example.csv)
* A sample submission showing how a writeup can look along with a structure that can be followed. This includes code that can be used to read in the data and create a submission.csv

### Submission Requirements

A valid submission must contain the following:

1. Kaggle Writeup - a short summary of what you learned from the data and how you approached the problem. Please try and keep this brief -
   1. Link to a Notebook or code respository used to generate your submission.csv
   2. Attached submission.csv using your model to be scored against the answers for the test set
   3. Attached Project Link if you would like to include additional external information or a link to a repo (optional)

Your final Submission must be made prior to the deadline. Any un-submitted or draft Writeups by the hackathon deadline will not be considered by the Judges.

To create a new Writeup, click on the "New Writeup" button [here](https://www.kaggle.com/competitions/inrange-competition/projects). After you have saved your Writeup, you should see a "Submit" button in the top right corner.

Note: If you attach a private Kaggle Resource to your public Kaggle Writeup, your private Resource will automatically be made public after the deadline.

1. Kaggle Writeup

The Kaggle Writeup serves as your project report. This should include a title, subtitle, uploaded submission csv file, link to your code (notebook/repo) and details of your submission. You must select a Track for your Writeup in order to submit (there is only one track to select). You can use the sample writeup as a guide.

The below assets must be attached to the Writeup to be eligible.

a. Code used to create the submission

Your code should be submitted as a public notebook in the `Project Files` field or as a link to a public notebook. Your notebook should be publicly accessible and not require a login or paywall. If you use a private Kaggle Notebook, it will automatically be made public after the deadline.

b. Attached submission.csv using your model to be scored for the test set

c. Public Project Link (optional)

A URL to an interactive demo. This allows judges to experience your project firsthand, if applicable. It should be publicly accessible and not require a login or paywall. If a live demo is not feasible, a link to your public code repository (e.g., GitHub) is required, including detailed setup instructions.

If you attach video it should be 3 minutes or less, and should be published to Youtube.

### Evaluation

Application: Evaluation Rubric (100 points total)

| Criteria | Points Possible |
| --- | --- |
| Prediction Challenge: Grading on the results of the prediction provided. | 0-50 points |
| Approach: How the training data was used to create a model to perform predictions. | 0-10 points |
| Data Analysys and Visualisation: How was the training and test data analysed and conclusions presented. | 0-10 points |
| Novelty: Was the data used in a new and inventive manner. | 0-5 points |
| Display of a full trajectory: With inputs only can a realistic, full trajectory be displayed and animated. | 0-20 points |
| Bounce and Roll: Was bounce and roll considered and shown. | 0-5 points |
| Required Elements: The writeup is less than 3000 words and uses a fewer than 25 graphics. | Yes/No |