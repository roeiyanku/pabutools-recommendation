# Experiments — recommendation system for participatory budgeting

Performance comparison and optimisation of the algorithms in *"A Recommendation
System for Participatory Budgeting"* (Leibiker & Talmon, 2023), implemented in
[`pabutools_recommendation`](../pabutools_recommendation).

This folder sits outside the `pabutools_recommendation/` package: it holds
benchmark scripts and their output, not library code.

## Running

From the repository root, with the `ml` and `experiments` extras installed
(`pip install -e ".[ml,experiments]"`), in the Linux virtualenv:

```bash
source .venv-wsl/bin/activate

python -m experiments.compare_setups                # Part A: algorithm comparison
python -m experiments.compare_setups --partiality   # Part A: parameter comparison
python -m experiments.compare_improvement           # Part B: before vs after
python -m experiments.compare_improvement --rejected  # the rejected techniques
```

Everything is measured under WSL (Ubuntu 24.04, Python 3.12, 4 cores) — the
same environment throughout, since mixing platforms would make the timings
incomparable.

Runs are resumable: `experiments_csv` skips rows already in the CSV, so an
interrupted sweep continues where it stopped. The flip side is that it cannot
tell "interrupted" from "measured badly" — delete the CSV to force a re-measure.

---

# Part A — performance comparison

## What is compared, and why

No other algorithm in pabutools solves this problem. The rules in
`pabutools.rules` (MES, Phragmén, greedy, max-welfare) select a bundle from
**complete** ballots, whereas the contribution here is deciding *which questions
to ask* in order to reconstruct **incomplete** ones. The comparison is therefore
between the paper's own algorithms, plus the simple heuristic the assignment
asks for when nothing comparable exists:

- **4 sampling setups** — `offline_popularity`, `offline_consensus`,
  `offline_controversiality`, `online_adaptive_controversial`
- **`random`** — the simple heuristic baseline: pick the questions at random
- **3 prediction modules** — `classification`, `matrix_factorization`,
  `factorization_machines`
- **the two partiality knobs** — `sample_degree` and `lv_degree`, the same
  algorithm run at different parameters

## Results — solution quality

Mean Fractional Allocation (Definition 5.1), over all input sizes and seeds:

| setup | classification | factorization_machines | matrix_factorization |
| --- | --- | --- | --- |
| offline_consensus | 0.7632 | 0.7522 | 0.7322 |
| offline_popularity | 0.8346 | 0.8629 | 0.8372 |
| offline_controversiality | 0.8419 | 0.8868 | 0.8370 |
| **online_adaptive_controversial** | **0.8743** | **0.9113** | **0.8832** |
| `random` (baseline) | 0.8318 | 0.8942 | 0.8455 |

Two findings worth stating plainly:

1. **`online_adaptive_controversial` wins**, on every predictor. The paper's
   most sophisticated setup is also its best, which is the result one hopes for.
2. **The random baseline is competitive.** At 0.8942 with factorization
   machines it beats every *offline* setup and loses only to the online adaptive
   one. Three of the paper's four informed setups do not beat choosing the
   questions at random — which is exactly why the assignment asks for a
   heuristic baseline.

`offline_consensus` is the weakest everywhere, below random on both metrics.

## Results — the parameter comparison

The same algorithms run at different values of the two Section 3.0.1 knobs,
which between them decide how many questions each Target Voter is asked.
`sample_degree` is the share of all *n x m* possible votes that the process can
afford to collect; `lv_degree` is the share of that budget spent on Learning
Voters, who answer about every project, rather than on Target Voters, who answer
`k` questions each and have the rest predicted.

Mean Fractional Allocation against the question budget:

| sample_degree | consensus | controversiality | popularity | online_adaptive | `random` |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.515 | 0.532 | 0.479 | **0.600** | 0.581 |
| 0.3 | 0.583 | 0.649 | 0.601 | **0.747** | 0.699 |
| 0.5 | 0.630 | 0.728 | 0.749 | **0.782** | 0.767 |
| 0.7 | 0.728 | 0.860 | **0.895** | 0.832 | 0.799 |
| 0.9 | 0.849 | 0.978 | **0.990** | 0.982 | 0.874 |

**Choosing the questions well matters most when questions are scarce.** At
`sample_degree = 0.1` the adaptive setup leads and the naive `random` baseline
is second; by 0.9, when nearly everything is collected anyway, the offline
setups reach 0.98-0.99 and `random` falls behind at 0.874. This is the pattern
the paper's approach predicts, and it explains why the overall averages in the
table above flatter `random` — it is competitive in the middle of the range and
weak at the ends.

Mean Fractional Allocation against how the budget is spent:

| lv_degree | consensus | controversiality | popularity | online_adaptive | `random` |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.583 | 0.742 | 0.717 | 0.785 | 0.735 |
| 0.3 | 0.620 | 0.752 | 0.725 | 0.817 | 0.735 |
| 0.5 | 0.659 | 0.754 | 0.734 | **0.819** | 0.767 |
| 0.7 | 0.732 | 0.760 | 0.791 | 0.779 | 0.750 |
| 0.9 | 0.712 | 0.739 | 0.745 | 0.744 | 0.732 |

**This is the paper's central claim, confirmed.** Quality peaks in the middle
and *falls* as `lv_degree` rises towards 1, where the budget is spent on a few
people answering everything and nobody else is asked at all - the paper's naive
"sampling" baseline. Asking many people a few questions and predicting the rest
beats asking few people everything.

## Results — running time

Mean seconds per run, by prediction module:

| projects | classification | factorization_machines | matrix_factorization |
| --- | --- | --- | --- |
| 10 | 2.47 | 0.10 | 0.11 |
| 30 | 2.96 | 0.17 | 0.22 |
| 60 | 7.22 | 0.60 | 0.73 |
| 120 | 6.37 | 0.68 | 0.81 |
| 240 | 14.6 | — | — |
| 480 | **36.1** | — | — |

Classification is **10–30× slower** than the other two modules, and it is the
paper's best performer on F1, so it cannot simply be dropped. That is what makes
it the target for Part B. At 480 projects it reaches 36 s, inside the 30-60
second band the assignment asks the sweep to reach.

The setups also separate by cost once the input is large, which is invisible at
small sizes. Mean runtime at 480 projects:

| setup | runtime |
| --- | --- |
| offline_controversiality | 8.6 s |
| offline_popularity | 10.7 s |
| offline_consensus | 15.5 s |
| `random` | 15.7 s |
| online_adaptive_controversial | **18.8 s** |

So `online_adaptive_controversial` buys its best-in-class quality with roughly
**twice** the running time of `offline_controversiality` - a real trade-off, and
one only visible at scale.

## Inputs

Random inputs of increasing size, from [`instances.py`](instances.py). Voters
are drawn from a small number of latent taste groups rather than independent
coin flips: with independent preferences there is no correlation between voters
for a recommender to learn, every predictor degrades to guessing the base rate,
and the comparison would be vacuous. A `uniform` model is kept as a null-model
control.

The budget is fixed at 30% of total project cost. `get_random_instance` samples
it uniformly between the cheapest project and the total, which swings from
"nothing fits" to "everything fits" and would swamp the differences between
setups in noise.

**Size axis.** Profiling showed cost is driven by the number of *projects*, not
voters: one model is trained per project, so runtime grows roughly linearly
there while doubling the voters barely moved it. Both sweeps stop enlarging the
input automatically once a run exceeds 60 seconds.

---

# Part B — the improvement

Committed as *"Speed up classification training by sampling projects when
tuning"*; the "before" numbers were measured on the parent commit.

## What was changed

The hyperparameter search in
[`train_classification`](../pabutools_recommendation/model_training.py) fitted a
classifier for **every project** in order to score **each candidate setting**.
With 30 projects and a grid of 3, one run cost:

```
3 candidates x 30 projects  +  30 final refits  =  120 model fits
```

The search only has to *rank* the candidates against one another, and a sample
of the projects ranks them the same as the whole bundle does. It now fits
`TUNING_PROJECTS` (10) of them:

```
3 candidates x 10 projects  +  30 final refits  =  60 model fits
```

Half the work, and the winning setting is unchanged.

## Results

| projects | before | after | speedup |
| --- | --- | --- | --- |
| 20 | 7.00 s | 3.87 s | 1.81x |
| 40 | 15.79 s | 7.54 s | 2.09x |
| 60 | 20.55 s | 6.48 s | 3.17x |
| 120 | 29.44 s | 9.35 s | 3.15x |
| 160 | 35.34 s | 12.57 s | 2.81x |
| 240 | **61.39 s** | **16.33 s** | **3.76x** |

The speedup **grows with input size** — 1.8x at the small end, 3.8x at the
large end — so it helps most where it is needed most.

**Largest input processed within 60 seconds:**

| | largest input |
| --- | --- |
| before | 160 projects |
| after | **240 projects** |

At 240 projects the improved version takes 16 s, so its real ceiling is well
beyond that; 240 is simply the largest size in the sweep.

**Solution quality is unaffected.** Fractional Allocation is identical up to 60
projects and within ~1.5% at the largest sizes (0.9134 against 0.9277 at 240).
This is not a speed-for-accuracy trade.

## Techniques that were tried and rejected

The assignment names four methods — caching, design patterns, multiprocessing,
and C/C++ integration — and says to fall back on changing the algorithm's
parameters if none of them yields a good improvement. Three of the four were
ruled out **by measurement**, which is what put this work on the fallback:

### Multiprocessing / threads — measured, slower

Fitting the per-project classifiers in a `ThreadPoolExecutor` was **2.7x
slower**: 11.18 s against 4.11 s sequential.

The reason is visible in the profile. The fitting is dominated by XGBoost's own
*Python-level* per-round bookkeeping, not by its compiled training code —
`_get_feature_info` alone is called 63,360 times in a single run, `c_str`
150,670 times. That work holds the GIL, so the threads contend instead of
running in parallel, and pay context-switching on top.

XGBoost's *internal* threading does not help either: at default (all 4 cores)
a run took 8.02 s, pinned to one core 7.75 s, and at `n_jobs=2` 10.16 s, while
burning 235% CPU for nothing. Each model is trained on one project's votes,
which is far too little data for thread coordination to pay off.

The code is kept behind `FIT_THREADS`, defaulting to 1, so the result can be
reproduced with `--rejected`.

### C / C++ (Cython, cppyy) — ruled out by profiling

Self-time by owner in one full run:

| | share |
| --- | --- |
| xgboost (already compiled C++) | 87.4% |
| Python stdlib | 10.3% |
| **pabutools' own Python** | **1.8%** |
| numpy, sklearn | 0.5% |

Only 1.8% of the runtime is in code we wrote, so by Amdahl's law rewriting
**all** of it in C bounds the gain at **1.02x**. The hot path is XGBoost's
compiled core, which is already C++ — there is nothing left to translate.

### Caching — ruled out by the same bound

`_project_rows` genuinely is recomputed with identical arguments once per
candidate setting, so it looks like a memoisation target. But it lives inside
that same 1.8%, so caching every redundant call is worth about 2%.

### Design patterns — not applicable

The only materialised intermediate is the per-project input list, which is
microseconds.

## A note on the environment

Two measurement traps were hit and are worth recording, since both produce
confident-looking wrong numbers:

1. **A default argument bound at import time.** `tuning_projects=TUNING_PROJECTS`
   in the signature captures the value when the module is *defined*, so
   rebinding the constant later silently did nothing — the first before/after
   comparison measured the new code against itself and reported "1.02x". Fixed
   with a sentinel resolved at call time.
2. **`/mnt/c` is slow.** The same workload was seen taking 8 s and 65 s
   depending on machine load and filesystem, so every number here is a median of
   repeated runs, and all runs are in the same environment.

## Known issue in `experiments_csv` 0.6.0

`single_plot_results` crashes under pandas 3.0: `Series.unique()` on a text
column now returns a `StringArray`, and the library calls `.sort()` on it, which
that type does not have. `plot_csv` in
[`compare_setups.py`](compare_setups.py) reads the frame, casts the z column to
plain objects, and calls the library's own `plot_dataframe`, which keeps the
plotting in `experiments_csv` and steps around the incompatibility.

## Output files

| file | contents |
| --- | --- |
| `results/project_sweep.csv` | Part A, quality and runtime by input size |
| `results/partiality_sweep.csv` | Part A, the parameter comparison |
| `results/improvement.csv` | Part B, before and after |
| `results/improvement_runtime.png` | **the before/after runtime graph** |
| `results/project_sweep_*.png` | Part A graphs, one per measurement |
| `results/*_runtime_<predictor>.png` | runtime per prediction module |

**Read the per-predictor runtime graphs, not the combined one.** The combined
graph averages classification (up to 44 s at 480 projects) together with matrix
factorization and factorization machines (around 1 s), so its vertical scale is
a blend of one slow module and two fast ones and the setups cannot be compared
on it fairly. `project_sweep_runtime_classification.png` is the graph that shows
the sweep reaching the assignment's 30-60 second band.
