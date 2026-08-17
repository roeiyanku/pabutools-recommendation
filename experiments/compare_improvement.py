"""
Part B - running time before and after the improvement.

Sweeps the same input sizes as :py:mod:`experiments.compare_setups` with the
improvement switched off and on, so the two runtime curves can be drawn on one
graph, and reports the largest input each version still finishes within 60
seconds.

The improvement is in
:py:func:`~pabutools_recommendation.model_training.train_classification`: the
hyperparameter search used to fit a classifier for *every* project to score
each candidate setting, which for 30 projects and a grid of 3 came to
``3 x 30 + 30 = 120`` fits per run. The search only has to rank the candidates
against one another, and a sample of the projects ranks them the same, so it
now fits :py:data:`~pabutools_recommendation.model_training.TUNING_PROJECTS`
of them: ``3 x 10 + 30 = 60``. Measured at 7.10 s -> 3.94 s with the
Fractional Allocation and F1 unchanged to four decimals.

Two other techniques from the lectures were measured and rejected, which
``--rejected`` reproduces:

* **Threads** over the per-project fits (``FIT_THREADS``): *slower*, 11.18 s
  against 4.11 s. The profile shows the fitting is dominated by XGBoost's own
  Python-level per-round bookkeeping rather than its compiled training code, so
  the threads contend for the GIL instead of running in parallel.
* **Cython / C++**: ruled out by profiling rather than tried. Only 1.8% of the
  runtime is in pabutools' own Python (87.4% is inside XGBoost, already
  compiled), so rewriting all of it in C bounds the gain at 1.02x.

Usage
-----
    python -m experiments.compare_improvement            # the sweep + plot
    python -m experiments.compare_improvement --plot     # re-draw from the CSV
    python -m experiments.compare_improvement --rejected # the rejected options

Programmer: Roei Yanku
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import experiments_csv

import pabutools_recommendation.model_training as model_training

from experiments.compare_setups import (
    BACKUP_FOLDER,
    LV_DEGREE,
    NUM_VOTERS,
    RESULTS_FOLDER,
    SAMPLE_DEGREE,
    SEEDS,
    TIME_LIMIT,
    plot_csv,
    single_run,
)

logger = logging.getLogger(__name__)

#: Project counts to sweep. Wider than the Part A grid: the point here is to
#: find where each version hits the 60-second limit.
PROJECT_COUNTS = [10, 20, 30, 40, 60, 80, 120, 160, 240]

#: ``tuning_projects`` before and after. ``None`` fits every project during the
#: hyperparameter search, which is what the original code did.
VERSIONS = {"before": None, "after": 10}


def run_version(
    num_voters: int,
    num_projects: int,
    version: str,
    setup: str,
    predictor: str,
    sample_degree: float,
    lv_degree: float,
    seed: int,
) -> dict:
    """One cell, with the improvement switched to whatever ``version`` names."""
    model_training.TUNING_PROJECTS = VERSIONS[version]
    return single_run(
        num_voters=num_voters, num_projects=num_projects, setup=setup,
        predictor=predictor, sample_degree=sample_degree, lv_degree=lv_degree,
        seed=seed,
    )


def sweep() -> experiments_csv.Experiment:
    """The before/after sweep, stopping at :py:data:`TIME_LIMIT` per run."""
    experiment = experiments_csv.Experiment(
        RESULTS_FOLDER, "improvement.csv", BACKUP_FOLDER
    )
    experiment.run_with_time_limit(
        run_version,
        {
            "num_voters": [NUM_VOTERS],
            "num_projects": PROJECT_COUNTS,
            "version": list(VERSIONS),
            "setup": ["offline_popularity"],
            "predictor": ["classification"],
            "sample_degree": [SAMPLE_DEGREE],
            "lv_degree": [LV_DEGREE],
            "seed": SEEDS,
        },
        time_limit=TIME_LIMIT,
    )
    return experiment


def plot() -> None:
    """Runtime before and after, against the input size."""
    for value in ("runtime", "fractional_allocation", "f1"):
        plot_csv(
            f"{RESULTS_FOLDER}/improvement.csv", "num_projects", value, "version",
            f"{RESULTS_FOLDER}/improvement_{value}.png",
        )


def report_largest_within_limit() -> None:
    """The biggest input each version still handles inside the time limit."""
    import pandas

    frame = pandas.read_csv(f"{RESULTS_FOLDER}/improvement.csv")
    averaged = frame.groupby(["version", "num_projects"])["runtime"].mean()
    print(f"\nLargest input finishing within {TIME_LIMIT}s:")
    for version in VERSIONS:
        within = averaged[version][averaged[version] <= TIME_LIMIT]
        largest = within.index.max() if len(within) else None
        print(f"  {version:7} {largest} projects")


def run_rejected(repeats: int = 3) -> None:
    """Reproduce the two rejected options, so the numbers can be checked."""
    model_training._xgboost()  # keep the one-off import out of the timings

    def timed(label: str, threads: int, tuning: int | None) -> float:
        model_training.FIT_THREADS = threads
        model_training.TUNING_PROJECTS = tuning
        times = []
        for seed in range(repeats):
            before = time.perf_counter()
            single_run(num_voters=NUM_VOTERS, num_projects=30,
                       setup="offline_popularity", predictor="classification",
                       sample_degree=SAMPLE_DEGREE, lv_degree=LV_DEGREE, seed=seed)
            times.append(time.perf_counter() - before)
        median = statistics.median(times)
        print(f"  {label:44} {median:6.2f}s", flush=True)
        return median

    print("\nRejected options (median of "
          f"{repeats} runs, 30 projects, {NUM_VOTERS} voters):")
    sequential = timed("sequential, tuning on all projects", 1, None)
    threaded = timed("threads over the per-project fits", 4, None)
    print(f"  -> threading is {threaded / sequential:.2f}x the sequential time")
    model_training.FIT_THREADS = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true",
                        help="only re-draw the plots from the CSV")
    parser.add_argument("--rejected", action="store_true",
                        help="re-measure the rejected options instead")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    experiments_csv.logger.setLevel(logging.INFO)

    if args.rejected:
        run_rejected()
        return
    if not args.plot:
        sweep()
    plot()
    report_largest_within_limit()


if __name__ == "__main__":
    main()
