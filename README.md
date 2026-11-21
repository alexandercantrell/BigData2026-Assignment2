# Big Data Analytics 2026 - Assignment 2: Predicting GitHub User Activity

## Introduction

For each of the 4,465 test GitHub users, we predict which activities from the set of (`open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`) they will complete in the test week, ranging from 2026-11-22 to 2026-11-29.

When downloaded and preprocessed using the given preprocessing file, 8 relational tables are generated representing one heterogeneous graph per week with every row becoming a node and every foreign key an edge (along with its reverse). To predict the activities of the coming week, a heterogeneous GraphSAGE model is used to predict the 6 activities for each user.

## Data Source

The data comes from [GH Archive](https://www.gharchive.org/), a public archive of GitHub events, and is processed with the provided `preprocess.py` file which filters the cohort and anonymizes the data.

Three of the files in `src/` are copied **unchanged** from `scripts/` in the course repository [BigData2026-Assignment2](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2) (commit `af9f335`):
`preprocess.py`, `cohort.txt`, and `evaluate.py`

An additional file (`test_input.csv`) is copied into the root directory from the same repository's `submission/` folder.

**Date Range Used:** hourly files `2026-08-29-0` to `2026-11-14-16` (UTC), 1,865 files, approx 28 GB.
The model only uses data before **2026-11-14 16:00 UTC**, exactly 7 days before the test week.

## Reproducibility

This repository was built and tested using Python 3.14.6 on Windows 11, using the CPU only. Everything should be run from the repository root in bash (Git Bash on Windows).
```bash
#1. Install requirements
pip install -r requirements.txt

#2. Download the raw GH Archive files
# In lieu of wget (e.g. Git Bash on Windows has no wget) the following can be used: curl --create-dirs --output-dir raw --remote-name-all <same URLs>
wget -P raw https://data.gharchive.org/2026-08-{29..31}-{0..23}.json.gz
wget -P raw https://data.gharchive.org/2026-09-{01..30}-{0..23}.json.gz
wget -P raw https://data.gharchive.org/2026-10-{01..31}-{0..23}.json.gz
wget -P raw https://data.gharchive.org/2026-11-{01..13}-{0..23}.json.gz
wget -P raw https://data.gharchive.org/2026-11-14-{0..16}.json.gz

# 3. Preprocess the data to build the 8 tables using the course's given preprocessing script (writes to data/db/*.parquet)
python src/preprocess.py --raw raw --out data

#4. Train every experiment, predict the test week with each trained model, and check the submission format
bash run.sh [student ID]
#The student ID is an optional flag, but without it, the data will be saved without the student ID in the path
```

`run.sh` writes `outputs/output_<student ID>.csv` - the submission from the final model - plus an additional `outputs/output_<student ID>_<name>.csv` per experiment where `<name>` is the name of the experiment. In order to only predict with the final model (`outputs/model.ckpt` - included in the repository) and skip training you can run:
```bash
python src/predict.py --id [student ID]
#Again, student ID is optional but recommended to get the correct final output
```

The seed is fixed in `constants.py` as `SEED=42`. Re-running `train.py` on CPU gave identical results, but depending on device this might not be the case. Training took about 3–8 minutes per model on a laptop CPU, depending on load. `run.sh` trains 8 models (5 feature experiments and 3 relational ablations), which takes roughly 30–60 minutes.

## Training options

| `src/train.py` flag | Default | Effect |
|---|---|---|
| `--time` / `--no-time` | on | time encoding: log(1 + age in days) of every row |
| `--bools` | off | boolean columns as node features |
| `--present` | off | "is this value filled in" flags as node features |
| `--category` | off | one-hot categories (review state, comment type, branch/tag) |
| `--aggr {all,mean}` | `all` | neighbour aggregation: `all` = mean + max + sum, `mean` = mean only |
| `--layers N` | 2 | number of message-passing layers |
| `--no-reverse` | off | leave out the 20 reverse edge types (only foreign-key direction) |
| `--name` | `model` | saves `outputs/<name>.ckpt` (checkpoint) and `outputs/<name>.json` (validation results) |


Every node also gets a constant feature. The checkpoint stores the feature flags and these settings so `predict.py --name <name>` can rebuild the matching graph and model.

## Code

| File | What it does |
|---|---|
| `src/constants.py` | holds constants such as paths, dates, model settings, the table schema (`SCHEMA`), optional features (`FEATURES`), and activity definitions (`ACTIVITY`) |
| `src/data.py` | loads the tables, takes snapshots (rows before a time, with "known only later" columns hidden), generates labels, computes node features, builds the graph, takes weekly samples, and copy-last-week baseline |
| `src/model.py` | defines the heterogeneous GraphSAGE model: one encoder per table/node type, 2 message-passing layers (mean + max + sum), 6 outputs per user |
| `src/metrics.py` | defines functions for F1, Score = (macro-F1 + micro-F1) / 2 as in `evaluate.py`, per-activity threshold tuning |
| `src/train.py` | trains and runs the model, evaluates it, and saves the model along with its finalized best results |
| `src/predict.py` | runs prediction on the model with a given test set and saves the output |
| `src/timeline.py` | draws `figures/timeline.png`|
| `src/preprocess.py`, `src/cohort.txt`, `src/evaluate.py` | from the course, unchanged: builds the 8 tables, the cohort list, format check and scoring |
| `test_input.csv` | from the course, unchanged: the 4,465 test users |
| `run.sh` | trains all 8 experiments, predicts the test week with each, and checks the submission format |
| `figures/` | holds the figures generated in code or hand drawn |
| `outputs/` | holds the model checkpoints and finalized results |


## Results

Validation weeks Oct 31 and Nov 7 (7-day gap like the test week, 10,417 user-weeks).
Score = (macro-F1 + micro-F1) / 2, as in the course's `evaluate.py`.

| Model | Node features | Best epoch | Score | macro-F1 | micro-F1 | Train time |
|---|---|---|---|---|---|---|
| Copy last week (baseline) | – | – | 27.98 % | 29.27 % | 26.69 % | – |
| `no_time` (graph structure only) | constant | 34 | 36.58 % | 36.10 % | 37.07 % | 8.1 min |
| **`model` (final)** | **constant + log age** | **35** | **38.35 %** | **38.12 %** | **38.58 %** | **7.4 min** |
| `bools` | + boolean columns | 38 | 38.36 % | 38.06 % | 38.65 % | 7.4 min |
| `columns` | + booleans, present flags, categories | 33 | 38.46 % | 38.23 % | 38.69 % | 8.0 min |
| `columns_no_time` | booleans, present flags, categories, no time | 40 | 36.56 % | 36.12 % | 37.01 % | 8.4 min |
| `mean_only` | constant + log age; mean-only aggregation | 40 | 28.82 % | 28.70 % | 28.94 % | 2.3 min |
| `one_layer` | constant + log age; 1 layer | 39 | 37.16 % | 36.91 % | 37.41 % | 2.6 min |
| `no_reverse` | constant + log age; no reverse edges | 39 | 38.56 % | 38.33 % | 38.78 % | 3.7 min |

F1 per activity (%):

| Model | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| Copy last week | 29.7 | 38.7 | 31.1 | 32.5 | 23.3 | 20.4 |
| `no_time` | 41.0 | 41.1 | 34.8 | 38.5 | 27.2 | 34.0 |
| **`model` (final)** | **41.6** | **42.6** | **37.8** | **40.8** | **32.5** | **33.4** |
| `bools` | 42.7 | 42.5 | 37.2 | 40.4 | 32.5 | 33.1 |
| `columns` | 41.7 | 42.5 | 38.0 | 41.2 | 32.3 | 33.8 |
| `columns_no_time` | 40.5 | 41.2 | 34.5 | 38.0 | 28.8 | 33.7 |
| `mean_only` | 33.3 | 34.0 | 26.9 | 33.5 | 23.3 | 21.2 |
| `one_layer` | 42.2 | 39.8 | 37.9 | 40.7 | 29.2 | 31.8 |
| `no_reverse` | 42.3 | 43.8 | 38.2 | 40.2 | 32.7 | 32.7 |

Ablations were run on different features and how they impacted performance. The time encoding adds about 1.8 points (36.58 to 38.35 %). The column features add at most 0.1 points,
so the final model uses only the constant and log age.

Several ablations were also run on the model structure and the relations between tables. Aggregation matters most:
with mean-only aggregation the score drops to 28.82 % (about copy-last-week level), because a mean of near-identical
neighbours cannot tell how many there are; sum and max keep "how many" and "how recent". A second layer adds 1.2 points
(37.16 to 38.35 %). Leaving out the reverse edges gives 38.56 % (+0.2), within run-to-run noise, so they have no
measurable effect.

Thresholds of the final model (one per activity, tuned on the validation weeks): 0.14 / 0.16 / 0.10 / 0.14 / 0.24 / 0.52. This is likely due to the imbalance in data.
The best epoch and the thresholds are chosen on these same weeks, so the GNN scores are somewhat optimistic.
GH Archive is missing a lot of data from late October on, which lowers every method's score in these weeks.
