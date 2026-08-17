"""
Evaluation of the recommendation system of
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

:py:mod:`~pabutools_recommendation.recommendation` runs the system - partial
ballots, sampling setups, prediction and the voting rule. This module measures
how well it did: the Section 5 accuracy metrics and the sweep over the paper's
treatment matrix that reports them. It complements pabutools' own
:py:mod:`pabutools.analysis` tools, which score a *completed* election rather
than the prediction that produced it.

Programmer: Roei Yanku
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    total_cost,
)
from pabutools_recommendation.recommendation import (
    PREDICTORS,
    SETUPS,
    complete_ballots,
    greedy_approval,
    split_lv_tv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 5.1 - Classification accuracy metrics.
# ---------------------------------------------------------------------------
def classification_metrics(
    real_approved: set[Project],
    predicted_approved: set[Project],
    hidden: set[Project],
) -> dict[str, float]:
    """
    Section 5.1 (Classification Accuracy Metrics): precision, recall and F1 of a
    prediction module, measured over one Target Voter's *hidden* projects (the
    test set - the exposed votes are known, not predicted, so they are excluded).
    Approval is the positive class, so from the confusion matrix over the hidden
    projects:

    * TP = hidden projects the voter really approves and we predicted approve,
    * FP = hidden projects we predicted approve but she really rejects,
    * FN = hidden projects she really approves but we predicted reject,

    then ``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)`` and
    ``F1 = 2 * precision * recall / (precision + recall)``. A denominator of 0
    (no predicted or no real approvals among the hidden projects) yields 0.0 for
    that metric, the usual convention for an undefined score.

    Parameters
    ----------
        real_approved : set[:py:class:`~pabutools.election.instance.Project`]
            The projects the voter really approves (her ideal ballot A_v).
        predicted_approved : set[:py:class:`~pabutools.election.instance.Project`]
            The projects the prediction module marked as approved.
        hidden : set[:py:class:`~pabutools.election.instance.Project`]
            The voter's hidden set H_v, i.e. the projects scored (from
            :py:func:`~pabutools_recommendation.recommendation.hidden_projects`).

    Returns
    -------
        dict[str, float]
            ``{"precision": ..., "recall": ..., "f1": ...}``, each in [0, 1].

    Examples
    --------
    Four hidden projects, the voter really approves {p1, p2}, the model predicts
    {p1, p3}: one hit (p1), one false alarm (p3), one miss (p2), so precision =
    recall = F1 = 0.5. A perfect prediction scores 1.0 across the board.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> hidden = {p1, p2, p3, p4}
    >>> classification_metrics({p1, p2}, {p1, p3}, hidden)
    {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
    >>> classification_metrics({p1, p2}, {p1, p2}, hidden)
    {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    # Restrict everything to the hidden projects: the exposed votes are known.
    real = real_approved & hidden
    predicted = predicted_approved & hidden
    tp = len(real & predicted)
    fp = len(predicted - real)
    fn = len(real - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    logger.info(
        "classification_metrics: TP=%d FP=%d FN=%d -> P=%.3f R=%.3f F1=%.3f",
        tp, fp, fn, precision, recall, f1,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Section 5.2 - Bundle evaluation metrics.
# ---------------------------------------------------------------------------
# The Symmetric Distance (Section 5.2.1), |rb △ pb|, is a one-liner
# ``len(real_bundle ^ predicted_bundle)`` computed inline where needed (see
# ``run_all_experiments``), so it gets no function of its own. (The paper's
# second toy example says SD({1,2,3}, {1,3,4}) = 1; by its own definition the
# symmetric difference is {2, 4}, i.e. 2 - its first and third examples do
# match |rb △ pb|.)
def fractional_allocation_score(
    real_bundle: Iterable[Project],
    predicted_bundle: Iterable[Project],
    budget_limit: int,
) -> float:
    """
    Definition 5.1 (Fractional Allocation score): the total cost of the
    projects predicted correctly (those in both bundles) divided by the budget
    limit, FA = lambda / B with lambda = sum of cost(p) over p in pb ∩ rb.

    Parameters
    ----------
        real_bundle : Iterable[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the real (full) ballots - any iterable of
            projects, e.g. the :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            returned by :py:func:`~pabutools_recommendation.recommendation.greedy_approval`,
            as is.
        predicted_bundle : Iterable[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the predicted ballots (same, any iterable).
        budget_limit : int
            The budget limit B of the instance.

    Returns
    -------
        float
            The fractional allocation score, in [0, 1].

    Examples
    --------
    pb = rb = {p1, p2}, costs 3 and 3, budget 6, so FA = 6/6 = 1.0. Disjoint
    bundles give FA = 0/6 = 0.0.

    >>> p1, p2, p3 = Project("p1", 3), Project("p2", 3), Project("p3", 6)
    >>> fractional_allocation_score({p1, p2}, {p1, p2}, budget_limit=6)
    1.0
    >>> fractional_allocation_score({p3}, {p1}, budget_limit=6)
    0.0
    """
    real_bundle, predicted_bundle = set(real_bundle), set(predicted_bundle)
    # lambda = total cost of the correctly-predicted projects (pb ∩ rb).
    correctly_predicted = real_bundle & predicted_bundle
    score = total_cost(correctly_predicted) / budget_limit
    logger.info(
        "fractional_allocation_score: %d/%d projects correct, FA=%.3f",
        len(correctly_predicted), len(real_bundle | predicted_bundle), score,
    )
    return float(score)


# ---------------------------------------------------------------------------
# Section 6 - The paper's treatment matrix.
# ---------------------------------------------------------------------------
#: The partiality grid swept in the paper's experiments (Section 6).
SAMPLE_DEGREES = (0.1, 0.15, 0.3, 0.5, 0.7, 0.9)
LV_DEGREES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def run_all_experiments(
    instance: Instance,
    profile: ApprovalProfile,
    *,
    setups: Iterable[str] = SETUPS,
    predictors: Iterable[str] = tuple(PREDICTORS),
    sample_degrees: Iterable[float] = SAMPLE_DEGREES,
    lv_degrees: Iterable[float] = LV_DEGREES,
    n_repeat: int = 50,
    seed: int | None = None,
) -> dict[tuple[float, float, str, str], dict[str, float]]:
    """
    The paper's full treatment matrix (Section 6, Figure 4). For every cell -
    setup x predictor x sample_degree x lv_degree - the ideal ``profile`` is
    split into LV/TV with
    :py:func:`~pabutools_recommendation.recommendation.split_lv_tv` (which also
    derives k, the number of questions per TV voter, from the two degrees -
    Section 3.0.1), the pipeline is run with
    :py:func:`~pabutools_recommendation.recommendation.run_pipeline`, and the
    predicted bundle is compared to the *real* bundle (greedy approval on the
    whole ideal profile). Because the split is random, each cell is repeated
    ``n_repeat`` times and the two bundle metrics - Fractional Allocation
    (:py:func:`fractional_allocation_score`) and the Symmetric Distance
    ``|rb △ pb|`` (Section 5.2.1, computed inline) - are averaged.

    .. warning::
        **The defaults reproduce the paper's grid and will not finish.** They
        describe 6 sample degrees x 7 LV degrees x 5 setups x 3 predictors = 630
        cells, each repeated 50 times, i.e. 31 500 pipeline runs. Measured on a
        real Warsaw district (Praga-Poludnie 2022, 10 424 voters, 96 projects),
        one run takes 30-165 seconds depending on the predictor - so the full
        default sweep is of the order of a year of compute.

        Scale it down deliberately. Shrinking ``sample_degrees``,
        ``lv_degrees``, ``setups``, ``predictors`` and ``n_repeat`` all help
        proportionally; passing a one-element ``param_grid`` through to
        :py:func:`pabutools_recommendation.model_training.train_classification` removes the
        Section 5 hyperparameter search, which costs roughly 10x a plain fit.
        A few hundred voters, one sample degree and ``n_repeat=3`` runs in
        minutes and is enough to see the trends.

    .. note::
        The paper repeats the sampling module 20 times and the prediction module
        50 times; ``n_repeat`` collapses both into one knob (default 50).
        ``classification`` needs ``xgboost`` installed, matrix factorization
        needs ``scikit-surprise``, and the hybrid needs ``lightfm``.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The ideal instance's full ballots (all voters).
        setups : Iterable[str], optional
            The setups to sweep (names from
            :py:data:`~pabutools_recommendation.recommendation.SETUPS`).
        predictors : Iterable[str], optional
            The predictors to sweep (names from
            :py:data:`~pabutools_recommendation.recommendation.PREDICTORS`).
            Defaults to all three paper predictors (classification / MF / FM).
        sample_degrees, lv_degrees : Iterable[float], optional
            The partiality grid (Section 3.0.1). Default to the paper's ranges.
        n_repeat : int, optional
            Random splits averaged per cell (default 50).
        seed : int, optional
            Seed for the whole sweep, for reproducibility.

    Returns
    -------
        dict[tuple[float, float, str, str], dict[str, float]]
            Keyed by ``(sample_degree, lv_degree, setup, predictor)``. Each value
            holds the two bundle metrics of Section 5.2 - ``"FA"`` and ``"SD"`` -
            and the three classification metrics of Section 5.1 -
            ``"precision"``, ``"recall"`` and ``"f1"``, averaged over the Target
            Voters and scored only on the votes that were predicted rather than
            answered. All five are means over ``n_repeat`` random splits.

    Examples
    --------
    The "perfect" case, swept over a tiny grid. Every cell yields an FA in
    [0, 1] and a non-negative SD. Only matrix factorization is swept here so
    that the example runs wherever ``scikit-surprise`` is installed, without
    also needing ``lightfm``.

    >>> p1, p2, p3, p4 = (Project("p1", 3), Project("p2", 3),
    ...                   Project("p3", 4), Project("p4", 4))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2])] * 4)
    >>> results = run_all_experiments(
    ...     inst, prof, predictors=("matrix_factorization",),
    ...     sample_degrees=(0.5,), lv_degrees=(0.5,), n_repeat=2, seed=0)
    >>> len(results) == 5 * 1  # 5 setups x 1 predictor x 1 x 1 cells
    True
    >>> all(0.0 <= c["FA"] <= 1.0 and c["SD"] >= 0 for c in results.values())
    True

    Every cell carries Section 5.1's ballot metrics as well as Section 5.2's
    bundle ones.

    >>> sorted(results[(0.5, 0.5, "random", "matrix_factorization")])
    ['FA', 'SD', 'f1', 'precision', 'recall']
    """
    setups, predictors = tuple(setups), tuple(predictors)
    sample_degrees, lv_degrees = tuple(sample_degrees), tuple(lv_degrees)
    logger.info(
        "run_all_experiments: sweeping %d setups x %d predictors x "
        "%d sample degrees x %d LV degrees = %d cells, %d repeats each",
        len(setups), len(predictors), len(sample_degrees), len(lv_degrees),
        len(setups) * len(predictors) * len(sample_degrees) * len(lv_degrees),
        n_repeat,
    )
    # The real bundle: greedy approval on the whole ideal profile (all voters).
    real_bundle = set(greedy_approval(instance, profile))
    logger.info(
        "run_all_experiments: real bundle has %d projects (the ground truth "
        "every cell is scored against)", len(real_bundle),
    )
    rng = random.Random(seed)
    results: dict[tuple[float, float, str, str], dict[str, float]] = {}
    for sample_degree in sample_degrees:
        for lv_degree in lv_degrees:
            for setup in setups:
                for name in predictors:
                    predict = PREDICTORS[name]
                    totals = dict.fromkeys(
                        ("FA", "SD", "precision", "recall", "f1"), 0.0
                    )
                    for repeat in range(1, n_repeat + 1):
                        # The split derives k from the two degrees (Section 3.0.1).
                        lv_profile, tv_ballots, k = split_lv_tv(
                            instance, profile, sample_degree, lv_degree,
                            seed=rng.randrange(2**32),
                        )
                        completed, exposed = complete_ballots(
                            instance, lv_profile, tv_ballots, k,
                            setup=setup, predict=predict,
                            seed=rng.randrange(2**32),
                        )
                        combined = ApprovalProfile(
                            list(lv_profile)
                            + [completed[vid] for vid in tv_ballots]
                        )
                        predicted = set(greedy_approval(instance, combined))
                        fa = fractional_allocation_score(
                            real_bundle, predicted, instance.budget_limit
                        )
                        # Symmetric Distance (Section 5.2.1): |rb △ pb|.
                        sd = len(real_bundle ^ predicted)
                        totals["FA"] += fa
                        totals["SD"] += sd
                        # Section 5.1, over the votes that were predicted rather
                        # than answered, averaged across the Target Voters (a
                        # voter with no hidden approvals scores 0, the usual
                        # convention for an undefined precision or recall).
                        for vid in tv_ballots:
                            scores = classification_metrics(
                                set(tv_ballots[vid]), set(completed[vid]),
                                set(instance) - exposed[vid],
                            )
                            for metric, value in scores.items():
                                totals[metric] += value / max(len(tv_ballots), 1)
                        logger.debug(
                            "run_all_experiments: repeat %d/%d of "
                            "(sample=%.2f, lv=%.2f, %s, %s): FA=%.3f SD=%d",
                            repeat, n_repeat, sample_degree, lv_degree,
                            setup, name, fa, sd,
                        )
                    cell = {k2: v / n_repeat for k2, v in totals.items()}
                    results[(sample_degree, lv_degree, setup, name)] = cell
                    logger.info(
                        "run_all_experiments: sample=%.2f lv=%.2f setup=%s "
                        "predict=%s -> mean FA=%.3f mean SD=%.2f over %d repeats",
                        sample_degree, lv_degree, setup, name,
                        cell["FA"], cell["SD"], n_repeat,
                    )
    return results
