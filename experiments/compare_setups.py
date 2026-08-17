"""
Part A - performance comparison of the recommendation-system algorithms.

Runs the Section 3 pipeline over a grid of

    input size  x  sampling setup  x  prediction module

on random inputs of increasing size, and records both the solution-quality
metrics and the running time into a CSV, using the ``experiments_csv`` library.

Which algorithms are compared, and why: no other algorithm in pabutools solves
this problem - the rules in ``pabutools.rules`` pick a bundle from *complete*
ballots, whereas the contribution here is choosing which questions to ask in
order to reconstruct *incomplete* ones. So the comparison is between the
paper's own four sampling setups, crossed with its three prediction modules,
against the ``random`` setup as the simple heuristic baseline.

Usage
-----
    python -m experiments.compare_setups           # the size sweep (Part A)
    python -m experiments.compare_setups --plot    # re-draw plots from the CSV

Programmer: Roei Yanku
"""

from __future__ import annotations

import argparse
import logging
import os
from time import perf_counter

import experiments_csv

from pabutools_recommendation.analytics import (
    classification_metrics,
    fractional_allocation_score,
)
from pabutools.election import ApprovalProfile
from pabutools_recommendation import (
    PREDICTORS,
    as_predictor,
    complete_ballots,
    greedy_approval,
    split_lv_tv,
)

from experiments.instances import random_setting

logger = logging.getLogger(__name__)

RESULTS_FOLDER = "experiments/results"

#: Where ``experiments_csv`` keeps its timestamped copy of a results file.
#: Named explicitly because it otherwise defaults to ``results_backup`` in the
#: working directory, which puts it at the repository root.
BACKUP_FOLDER = "experiments/results_backup"

#: The four informed setups plus ``random``, the simple heuristic baseline.
SETUPS_COMPARED = [
    "random",
    "offline_popularity",
    "offline_consensus",
    "offline_controversiality",
    "online_adaptive_controversial",
]

#: Primary size axis. Profiling showed the cost is driven by the number of
#: *projects*, not voters: ``_fit_per_project`` trains one model per project,
#: so runtime grows roughly linearly here (12 -> 24 -> 48 projects measured at
#: about 6 -> 12 -> 23 seconds) while doubling the voters barely moved it.
#: Grows until a single run passes the time limit; the harness then stops
#: enlarging on its own. Reaches into the assignment's 30-60 second band: the
#: unimproved code needs about 61 s at 240 projects and the improved code about
#: 16 s, so the sweep has to go well past 120 for the slower curve to get there.
PROJECT_COUNTS = [10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480]

#: Held fixed while projects grow, so the graph has one moving part. Swept
#: separately in :py:func:`sweep_voters` as the secondary size axis.
NUM_VOTERS = 100

#: Secondary size axis, with the projects held fixed.
VOTER_COUNTS = [50, 100, 200, 400, 800, 1600]

#: Partiality knobs (Section 3.0.1), fixed at a mid-range value for the size
#: sweep - they are swept separately in :py:func:`sweep_partiality`.
SAMPLE_DEGREE = 0.5
LV_DEGREE = 0.5

#: The values those two knobs take in :py:func:`sweep_partiality`, which is the
#: assignment's "one algorithm with several parameters, so compare different
#: runs with different parameters". Trimmed from the paper's own grid
#: (:py:data:`~pabutools_recommendation.analytics.SAMPLE_DEGREES` and
#: ``LV_DEGREES``), which is 6 x 7 and would not finish.
SAMPLE_DEGREES = [0.1, 0.3, 0.5, 0.7, 0.9]
LV_DEGREES = [0.1, 0.3, 0.5, 0.7, 0.9]

#: Projects used for the partiality sweep - small, since that grid is wide.
PARTIALITY_PROJECTS = 30

#: Repetitions per cell; the LV/TV split is random, so a single draw is noisy.
#: Three, not the paper's 50: at 5 setups x 3 predictors x 7 sizes a single
#: repetition already costs on the order of an hour.
SEEDS = [0, 1, 2]

#: A single run may not exceed this many seconds (the assignment's cap).
TIME_LIMIT = 60


def single_run(
    num_voters: int,
    num_projects: int,
    setup: str,
    predictor: str,
    sample_degree: float,
    lv_degree: float,
    seed: int,
) -> dict:
    """
    One cell of the design matrix: build a random setting, split it into Local
    Voters and Target Voters, expose k projects per TV under ``setup``,
    complete the hidden votes with ``predictor``, and score the resulting
    bundle against the bundle the ideal profile would have produced.

    Returns the dict of measurements that ``experiments_csv`` writes as a row.
    """
    instance, profile = random_setting(num_voters, num_projects, seed=seed)
    lv_profile, tv_ballots, k = split_lv_tv(
        instance, profile, sample_degree, lv_degree, seed=seed
    )

    # Time the pipeline itself, excluding input generation and the reference
    # bundle - the harness's own ``runtime`` column covers the whole call.
    before = perf_counter()
    completed, exposed = complete_ballots(
        instance, lv_profile, tv_ballots, k,
        setup=setup, predict=as_predictor(predictor), seed=seed,
    )
    combined = ApprovalProfile(
        list(lv_profile) + [completed[vid] for vid in tv_ballots]
    )
    predicted_bundle = greedy_approval(instance, combined)
    pipeline_runtime = perf_counter() - before

    # Section 5.1 - classification accuracy, over the hidden projects only,
    # averaged across the Target Voters.
    all_projects = set(instance)
    scores = [
        classification_metrics(
            tv_ballots[vid], set(completed[vid]), all_projects - exposed[vid]
        )
        for vid in tv_ballots
    ]
    mean = lambda key: sum(s[key] for s in scores) / len(scores) if scores else 0.0

    # Section 5.2 - bundle quality against the real bundle.
    real_bundle = greedy_approval(instance, profile)
    return {
        "k": k,
        "num_tv": len(tv_ballots),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "fractional_allocation": fractional_allocation_score(
            real_bundle, predicted_bundle, instance.budget_limit
        ),
        "symmetric_distance": len(set(real_bundle) ^ set(predicted_bundle)),
        "pipeline_runtime": pipeline_runtime,
    }


def sweep_projects() -> experiments_csv.Experiment:
    """
    The main sweep: quality and running time as the number of projects grows,
    for every setup x predictor pair. Stops enlarging the input once a run
    exceeds :py:data:`TIME_LIMIT`.
    """
    experiment = experiments_csv.Experiment(
        RESULTS_FOLDER, "project_sweep.csv", BACKUP_FOLDER
    )
    experiment.run_with_time_limit(
        single_run,
        {
            "num_voters": [NUM_VOTERS],
            "num_projects": PROJECT_COUNTS,
            "setup": SETUPS_COMPARED,
            "predictor": list(PREDICTORS),
            "sample_degree": [SAMPLE_DEGREE],
            "lv_degree": [LV_DEGREE],
            "seed": SEEDS,
        },
        time_limit=TIME_LIMIT,
    )
    return experiment


def sweep_voters() -> experiments_csv.Experiment:
    """
    The secondary sweep: the same measurements as the number of voters grows,
    with the projects held fixed.
    """
    experiment = experiments_csv.Experiment(
        RESULTS_FOLDER, "voter_sweep.csv", BACKUP_FOLDER
    )
    experiment.run_with_time_limit(
        single_run,
        {
            "num_voters": VOTER_COUNTS,
            "num_projects": [20],
            "setup": SETUPS_COMPARED,
            "predictor": list(PREDICTORS),
            "sample_degree": [SAMPLE_DEGREE],
            "lv_degree": [LV_DEGREE],
            "seed": SEEDS,
        },
        time_limit=TIME_LIMIT,
    )
    return experiment


#: The measurements plotted against each size axis.
PLOTTED = ("runtime", "f1", "fractional_allocation", "symmetric_distance")


def plot_csv(csv_path: str, x_field: str, y_field: str, z_field: str,
             save_to: str, where: dict | None = None,
             title: str | None = None) -> None:
    """
    One graph, a line per value of ``z_field``, averaged over the repetitions.
    ``where`` keeps only the rows matching every column/value pair it gives,
    which is how the per-predictor graphs are drawn.

    ``experiments_csv.single_plot_results`` would be the natural call here, but
    it crashes under pandas 3: ``Series.unique()`` on a text column now returns
    a ``StringArray``, and the library calls ``.sort()`` on it, which that type
    does not have. Reading the frame here and handing it to the library's own
    ``plot_dataframe`` with the z column as plain objects keeps the plotting in
    ``experiments_csv`` and steps around the incompatibility.
    """
    import pandas
    from matplotlib import pyplot as plt

    frame = pandas.read_csv(csv_path)
    for column, value in (where or {}).items():
        frame = frame[frame[column] == value]
    frame[z_field] = frame[z_field].astype(object)
    plt.figure()
    experiments_csv.plot_dataframe(plt, frame, x_field, y_field, z_field, mean=True)
    plt.legend(prop={"size": 8})
    plt.xlabel(x_field)
    plt.ylabel(f"mean {y_field}")
    plt.title(title or f"{y_field} by {x_field}")
    plt.savefig(save_to, bbox_inches="tight")
    plt.close()
    logger.info("plot_csv: wrote %s", save_to)


def sweep_partiality() -> experiments_csv.Experiment:
    """
    The parameter comparison: the same algorithm run at different values of the
    two Section 3.0.1 partiality knobs, which between them decide how many
    questions k each Target Voter is asked. Input size is held fixed here, so
    the only thing moving is the parameters.
    """
    experiment = experiments_csv.Experiment(
        RESULTS_FOLDER, "partiality_sweep.csv", BACKUP_FOLDER
    )
    experiment.run_with_time_limit(
        single_run,
        {
            "num_voters": [NUM_VOTERS],
            "num_projects": [PARTIALITY_PROJECTS],
            "setup": SETUPS_COMPARED,
            "predictor": list(PREDICTORS),
            "sample_degree": SAMPLE_DEGREES,
            "lv_degree": LV_DEGREES,
            "seed": SEEDS,
        },
        time_limit=TIME_LIMIT,
    )
    return experiment


def plot_sweep(csv_name: str, x_field: str) -> None:
    """
    Draw one graph per measurement from the named CSV, a curve per setup, plus
    one runtime graph per prediction module.

    The per-predictor graphs are the ones to read for cost: the combined graph
    averages classification (36 s at 480 projects) together with the other two
    (around 1 s), so its scale is a blend of a slow module and two fast ones and
    the setups cannot be compared on it fairly.
    """
    csv_path = f"{RESULTS_FOLDER}/{csv_name}.csv"
    for value in PLOTTED:
        plot_csv(
            csv_path, x_field, value, "setup",
            f"{RESULTS_FOLDER}/{csv_name}_{value}.png",
        )
    for predictor in PREDICTORS:
        plot_csv(
            csv_path, x_field, "runtime", "setup",
            f"{RESULTS_FOLDER}/{csv_name}_runtime_{predictor}.png",
            where={"predictor": predictor},
            title=f"runtime by {x_field} - {predictor}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true", help="only re-draw plots from the CSVs"
    )
    parser.add_argument(
        "--voters", action="store_true", help="run the secondary voter sweep too"
    )
    parser.add_argument(
        "--partiality", action="store_true",
        help="run the parameter sweep over the two partiality knobs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    experiments_csv.logger.setLevel(logging.INFO)

    if not args.plot:
        if args.partiality:
            sweep_partiality()
        else:
            sweep_projects()
            if args.voters:
                sweep_voters()
    # Plot whichever sweeps have actually been run - ``--plot`` is meant to
    # redraw everything on disk, and a sweep that was never run has no CSV.
    for csv_name, x_field in (
        ("partiality_sweep", "sample_degree"),
        ("project_sweep", "num_projects"),
        ("voter_sweep", "num_voters"),
    ):
        if os.path.exists(f"{RESULTS_FOLDER}/{csv_name}.csv"):
            plot_sweep(csv_name, x_field)


if __name__ == "__main__":
    main()
