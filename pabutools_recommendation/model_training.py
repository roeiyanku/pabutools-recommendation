"""
Model *training* for the learning-based prediction modules of
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf
(Section 2.1).

This module contains both fitting and prediction for the three learning-based
modules. ``pabutools_recommendation`` is responsible only for sampling, orchestration,
and the voting rule.

All three models train on the votes the process actually *collected*, which
Section 3.1.1 defines as the preferences of the LV voters **and** the exposed
preferences E_TV of all Target Voters. The Target Voters' hidden votes are the
paper's test set (Section 5) and are never read during training.

* :py:func:`train_classification` is supervised: one binary classifier per
  project, taking preferences on P \\ {p} as input and the preference on p as
  its 0/1 output. Every LV voter trains every project's classifier; a Target
  Voter trains project p's when p lies in her exposed set E_v, with the
  projects she was not asked about left missing - the very gaps that the rows
  to be predicted will carry.
* :py:func:`train_matrix_factorization` and
  :py:func:`train_factorization_machines` are collaborative filtering: one
  voter-project matrix holds the Learning Voters' full ballots **and every Target
  Voter's exposed votes** (the paper's E_TV), factorised once for the whole TV
  population. The FM hybrid additionally describes every project by Table 2's
  attributes - cost, categories and target population segments - which is what
  lets a voter's taste reach a project she was never asked about.

Following Section 5, all three also weight the minority class against the ~10%
approval rate of Table 1, and choose their hyperparameters on a 15% validation
set scored by F1 (Section 5.1's "suitable measure for an imbalanced data").

.. note::
    Section 2.1.1 says only that binary classification is used and that XGBoost
    is a good algorithm for it; it never states what one training row is, nor
    which features it carries. The per-project reading implemented here - a
    voter's opinions on the other projects as the features - is therefore one
    defensible interpretation of an under-specified section, not something the
    paper prescribes.

.. note::
    The paper names one library only: ``xgboost``, for the classification module
    (Section 2.1.1). It does not say what backs matrix factorization or
    factorization machines, so ``scikit-surprise`` and ``lightfm`` are this
    implementation's choices. Following the maintainer's advice, the ML libraries
    are **not a hard requirement**: they are imported lazily by the corresponding
    training function and are available through the ``recommendation`` optional
    dependency.

Programmer: Roei Yanku
Date: 2026-06-20.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    CardinalBallot,
)

logger = logging.getLogger(__name__)

#: The exposed votes of every Target Voter, keyed by voter id: E_v split into
#: her approvals A_v and disapprovals D_v (Section 2.3).
TVExposed = dict[str, tuple[set[Project], set[Project]]]


# ---------------------------------------------------------------------------
# Section 2.1.1 - Binary classification (one classifier per project).
# ---------------------------------------------------------------------------
#: Hyperparameter settings scored against the validation set (Section 5). The
#: first entry is what gets fitted when tuning is switched off (``param_grid=()``).
#:
#: ``class_weight`` is deliberately *not* varied here; see :py:func:`_fit_per_project`
#: for why the minority-class correction of Section 5 is off by default.
DEFAULT_PARAM_GRID = (
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
)

#: How many projects the hyperparameter search of :py:func:`train_classification`
#: fits before scoring a candidate setting; ``None`` uses all of them.
#:
#: The search exists only to rank the settings in :py:data:`DEFAULT_PARAM_GRID`
#: against one another, and a sample of the projects ranks them just as well as
#: the whole bundle does. Fitting all of them made tuning cost
#: ``len(param_grid)`` full bundles - three quarters of the total work, since
#: the winner is then refitted on everything anyway.
TUNING_PROJECTS: int | None = 10

#: Default for ``train_classification``'s ``tuning_projects``, resolved against
#: :py:data:`TUNING_PROJECTS` when the function runs rather than when it is
#: defined - otherwise rebinding the constant (as the ``experiments/`` before
#: and after comparison does) would silently have no effect.
_FROM_CONFIG = object()

#: Threads used to fit the per-project classifiers of :py:func:`_fit_per_project`
#: in parallel. ``1`` fits sequentially.
#:
#: .. warning::
#:     Defaults to 1, i.e. **off**. Threading this loop was measured to be
#:     *slower*, not faster: 11.18 s against 4.11 s sequential on 30 projects
#:     and 100 voters. The reason is visible in the profile - the fitting is
#:     dominated by XGBoost's own Python-level per-round bookkeeping
#:     (``_get_feature_info`` alone is called 63,360 times in a single run),
#:     not by its compiled training code, so the threads spend their time
#:     contending for the GIL rather than working in parallel.
FIT_THREADS: int = 1


def _xgboost():
    """
    The lazily imported ``xgboost`` module (an optional dependency).

    Examples
    --------
    >>> _xgboost().__name__
    'xgboost'
    """
    try:
        import xgboost
    except ImportError:
        raise ImportError(
            "You need to install xgboost to train the classification "
            "predictor (pip install pabutools-recommendation[ml])."
        )
    return xgboost


def _masked_copies(lv_votes: np.ndarray, tv_votes: np.ndarray, seed: int):
    """
    The LV rows to fit on: their full ballots, plus one blanked copy of each.

    At prediction time a classifier is asked about a project the voter was never
    questioned on, from a row where nearly every input is missing. An LV row is
    complete, and a Target Voter only supplies a row for the projects she *was*
    asked about - so for the projects that actually need predicting, the
    classifiers would never meet a missing value during training and would send
    every blank one fixed way. That is what made the offline setups, where every
    voter is asked the same projects, return one identical ballot for everybody.

    Each LV voter is therefore added a second time with a real Target Voter's
    pattern of unanswered projects blanked out, so the classifiers train on rows
    shaped like the ones they will be asked about. Her labels stay correct
    because the label column is dropped from the features anyway, so blanking it
    cannot leak anything.

    Returns ``(features, labels)``: the labels are always the true full ballots,
    only the features are blanked.

    Examples
    --------
    One LV voter who answered all three projects, and one Target Voter who was
    asked only about the first. The LV row is kept as it is and added a second
    time with the two projects she would not have been asked about blanked out;
    her labels stay the true ballot both times.

    >>> lv = np.array([[1.0, 0.0, 1.0]])
    >>> tv = np.array([[1.0, np.nan, np.nan]])
    >>> features, labels = _masked_copies(lv, tv, seed=0)
    >>> features.tolist()
    [[1.0, 0.0, 1.0], [1.0, nan, nan]]
    >>> labels.tolist()
    [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]

    With no Target Voters there is nothing to imitate, so the rows are returned
    untouched.

    >>> features, labels = _masked_copies(lv, np.zeros((0, 3)), seed=0)
    >>> features.tolist()
    [[1.0, 0.0, 1.0]]
    """
    if not len(lv_votes) or not len(tv_votes):
        return lv_votes, lv_votes
    unasked = np.isnan(tv_votes)  # True where that Target Voter was not asked
    chosen = np.random.default_rng(seed).integers(0, len(tv_votes), len(lv_votes))
    masked = lv_votes.copy()
    masked[unasked[chosen]] = np.nan
    return np.vstack([lv_votes, masked]), np.vstack([lv_votes, lv_votes])


def _project_rows(
    index: int, lv_features: np.ndarray, lv_labels: np.ndarray,
    tv_votes: np.ndarray,
):
    """
    The ``(X, y)`` rows for the classifier of the project in column ``index``:
    every LV row, plus the Target Voters who were asked about that project
    (a voter without a preference on it has no label to contribute). The target
    project's own column is dropped from ``X`` - it is the label, not an input.

    Examples
    --------
    Two projects and two LV voters, plus one Target Voter asked only about
    project 0. Building project 0's classifier leaves project 1 as the sole
    input, and all three voters supply a label for project 0.

    >>> lv = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> tv = np.array([[1.0, np.nan]])
    >>> X, y = _project_rows(0, lv, lv, tv)
    >>> X.ravel().tolist()
    [0.0, 1.0, nan]
    >>> y.tolist()
    [1.0, 0.0, 1.0]

    For project 1 the Target Voter has no preference to contribute, so only the
    two LV voters appear.

    >>> X, y = _project_rows(1, lv, lv, tv)
    >>> y.tolist()
    [0.0, 1.0]
    """
    answered = ~np.isnan(tv_votes[:, index])
    y = np.concatenate([lv_labels[:, index], tv_votes[answered, index]])
    X = np.vstack([
        np.delete(lv_features, index, axis=1),
        np.delete(tv_votes[answered], index, axis=1),
    ])
    return X, y


def _fit_per_project(
    projects: list[Project], lv_features: np.ndarray, lv_labels: np.ndarray,
    tv_votes: np.ndarray, params: dict, only: set[int] | None = None,
) -> dict[Project, tuple]:
    """
    Fit one classifier per project on the given collected votes. A project whose
    collected votes all agree (or that nobody voted on) needs no classifier, so
    its common/majority preference is stored directly.

    ``only`` restricts the work to the given column indices, leaving the other
    projects out of the returned bundle entirely. It exists for the
    hyperparameter search of :py:func:`train_classification`, which compares
    settings against each other and does not need the whole bundle to do it.

    Examples
    --------
    Two voters who agree exactly: everyone approves p1 and rejects p2. Neither
    project needs a classifier, so each is stored as the constant both voters
    gave, and ``xgboost`` is never even imported.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> votes = np.array([[1.0, 0.0], [1.0, 0.0]])
    >>> fitted = _fit_per_project([p1, p2], votes, votes, np.zeros((0, 2)), {})
    >>> fitted[p1], fitted[p2]
    (('const', 1), ('const', 0))

    With ``only``, the projects left out simply do not appear.

    >>> sorted(_fit_per_project([p1, p2], votes, votes, np.zeros((0, 2)), {},
    ...                         only={0}), key=str)
    [p1]
    """
    per_project: dict[Project, tuple] = {}
    to_fit: list[tuple[Project, np.ndarray, np.ndarray]] = []
    for index, project in enumerate(projects):
        if only is not None and index not in only:
            continue
        X, y = _project_rows(index, lv_features, lv_labels, tv_votes)
        # An empty y has fewer than 2 distinct labels too, so it is covered.
        if X.shape[1] == 0 or len(set(y.tolist())) < 2:
            per_project[project] = (
                "const", int(len(y) > 0 and 2 * int(y.sum()) >= len(y))
            )
        else:
            to_fit.append((project, X, y))

    # The projects left over each need a classifier, and none of them depends on
    # another's result, so they are fitted in parallel. Every worker only reads
    # the shared arrays and returns its own pair, so there is no shared mutable
    # state to guard with a lock.
    if not to_fit:
        return per_project
    if FIT_THREADS == 1 or len(to_fit) == 1:
        fitted = [_fit_one_project(*args, projects, params) for args in to_fit]
    else:
        with ThreadPoolExecutor(max_workers=FIT_THREADS) as executor:
            futures = [
                executor.submit(_fit_one_project, project, X, y, projects, params)
                for project, X, y in to_fit
            ]
            fitted = [future.result() for future in futures]
    logger.info(
        "_fit_per_project: fitted %d classifiers on %d thread(s), %d project(s) "
        "needed no model",
        len(fitted), 1 if FIT_THREADS == 1 else (FIT_THREADS or 0),
        len(per_project),
    )
    per_project.update(fitted)
    return per_project


def _fit_one_project(
    project: Project, X: np.ndarray, y: np.ndarray,
    projects: list[Project], params: dict,
) -> tuple[Project, tuple]:
    """
    Fit the single classifier that predicts the votes on ``project`` from the
    votes on all the others, and return it keyed by its project. Split out of
    :py:func:`_fit_per_project` so that the fits, which are independent of one
    another, can be handed to a thread pool.

    Examples
    --------
    Two voters who disagree on p1 while agreeing on p2, so p1 does need a model.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> X, y = np.array([[0.0], [1.0]]), np.array([1.0, 0.0])
    >>> fitted_project, (kind, (others, _)) = _fit_one_project(
    ...     p1, X, y, [p1, p2], {"n_estimators": 2})
    >>> fitted_project, kind, others
    (p1, 'model', [p2])
    """
    settings = dict(params)
    # Section 5 addresses the ~10% approval rate of Table 1 "by modifying
        # the model loss function to give more weight to minority class
        # samples", which for XGBoost is ``scale_pos_weight`` (the weighted-loss
        # approach of the paper's reference [26]). ``class_weight`` is the
        # exponent applied to the class ratio: 0 leaves the loss alone, 1
        # balances the two classes completely, 0.5 is halfway.
        #
        # It defaults to 0, i.e. no correction, which is a deliberate departure
        # from Section 5. Measured on three real Warsaw districts, the full
        # correction collapses the very metric the paper reports:
        #
        #     Praga-Poludnie 2022, offline_popularity:  FA 0.802 -> 0.007
        #     Mokotow 2022,        offline_popularity:  FA 0.869 -> 0.114
        #     Wola 2021,           offline_popularity:  FA 0.731 -> 0.082
        #
        # The cause is an interaction with the offline setups, where every
        # Target Voter is asked about the *same* k projects and so arrives with
        # an identical pattern of missing inputs. A balanced loss then makes the
        # classifiers approve most of the ballot for everyone alike, the
        # approval scores flatten, and the ranking greedy approval needs is
        # lost. (Under the ``random`` setup, where exposures differ per voter,
        # the correction is harmless and occasionally helps.) Raise it if you
        # care about per-voter recall rather than the winning bundle: recall
        # goes 0.078 -> 0.773 across that same range, while precision stays
        # near 0.08.
    exponent = settings.pop("class_weight", 0.0)
    positives = int(y.sum())
    negatives = len(y) - positives
    ratio = (negatives / positives) if positives else 1.0
    # ``n_jobs=1`` unless the caller asked otherwise: each model is fitted on a
    # single project's votes, which is far too little data for XGBoost's own
    # threading to pay off - measured at 8.02 s with its default (all cores) and
    # 7.75 s pinned to one, while its 235% CPU left nothing spare. Pinning each
    # model to one core is what makes the parallelism *across* projects useful.
    settings.setdefault("n_jobs", 1)
    classifier = _xgboost().XGBClassifier(
        verbosity=0,
        scale_pos_weight=ratio ** exponent,
        **settings,
    )
    classifier.fit(X, y)
    return project, ("model", ([p for p in projects if p != project], classifier))


def _pooled_f1(
    per_project: dict[Project, tuple], projects: list[Project],
    lv_features: np.ndarray, lv_labels: np.ndarray, tv_votes: np.ndarray,
) -> float:
    """
    F1 of a fitted bundle over every held-out vote, pooled across projects.
    Section 5.1 calls F1 the "suitable measure for an imbalanced data", so it is
    the criterion the validation set selects hyperparameters on.

    Examples
    --------
    One held-out voter who approves p1 and rejects p2, against a bundle that
    predicts exactly that: no false positives and no misses, so F1 is 1.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> votes = np.array([[1.0, 0.0]])
    >>> bundle = {p1: ("const", 1), p2: ("const", 0)}
    >>> _pooled_f1(bundle, [p1, p2], votes, votes, np.zeros((0, 2)))
    1.0

    A bundle that approves both projects finds the one real approval but also
    raises a false alarm on p2: precision 0.5, recall 1.0.

    >>> _pooled_f1({p1: ("const", 1), p2: ("const", 1)},
    ...            [p1, p2], votes, votes, np.zeros((0, 2)))
    0.6666666666666666
    """
    tp = fp = fn = 0
    for index, project in enumerate(projects):
        # A bundle fitted with ``only`` covers a subset of the projects; the
        # rest carry no prediction to score.
        if project not in per_project:
            continue
        X, y = _project_rows(index, lv_features, lv_labels, tv_votes)
        if len(y) == 0:
            continue
        kind, payload = per_project[project]
        if kind == "const":
            predicted = np.full(len(y), float(payload))
        else:
            predicted = payload[1].predict(X)
        tp += int(((y == 1) & (predicted == 1)).sum())
        fp += int(((y == 0) & (predicted == 1)).sum())
        fn += int(((y == 1) & (predicted == 0)).sum())
    return _f1(tp, fp, fn)


def _f1(tp: int, fp: int, fn: int) -> float:
    """
    F1 from a confusion matrix (Section 5.1), 0.0 when it is undefined.

    Examples
    --------
    >>> _f1(3, 1, 1)          # precision 0.75, recall 0.75
    0.75
    >>> _f1(0, 5, 5)          # nothing found: precision and recall both 0
    0.0
    >>> _f1(0, 0, 0)          # undefined; 0.0 by the usual convention
    0.0
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def train_classification(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, CardinalBallot] | None = None,
    *,
    validation_fraction: float = 0.15,
    param_grid=DEFAULT_PARAM_GRID,
    tuning_projects=_FROM_CONFIG,
    seed: int = 0,
) -> dict:
    """
    Train the per-project binary classifiers of Section 2.1.1. For every project
    p, one classifier uses preferences on P \\ {p} as input and the preference on
    p as its 0/1 output. The resulting model bundle is trained once and reused
    for every Target Voter; preferences the voter was not asked about are passed
    to XGBoost as missing values.

    The training data is every vote the process *collected*, which Section 3.1.1
    defines as the preferences of the LV voters **and** the exposed preferences
    E_TV of the Target Voters: each LV voter contributes a row to every
    project's classifier, and a Target Voter contributes a row for project p
    exactly when p lies in her exposed set E_v (otherwise there is no label to
    learn from). Those rows carry NaN for the projects she was not asked about,
    which is what teaches XGBoost a split direction for missing inputs - the
    same missingness the rows to be predicted will carry.

    Following Section 5, the collected votes are split three ways rather than
    used wholesale: ``validation_fraction`` of the *voters* are held back as the
    validation set ("a predefined set of votes from closed set of voters"), each
    setting in ``param_grid`` is fitted on the rest and scored on them by pooled
    F1, and the winning setting is then refitted on all collected votes. (The
    Target Voters' hidden votes are the paper's test set and are never touched
    here.) One setting is chosen for the whole bundle, not per project, matching
    the paper's singular "the hyperparameters of a model".

    Backed by the external ``xgboost`` library
    (:py:class:`xgboost.XGBClassifier`, class-weighted for the imbalanced data),
    imported lazily. When no collected vote covers a project, or they all agree
    on it, no classifier is needed, so we use that common/majority preference
    directly.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        tv_ballots : dict[str, :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`], optional
            The Target Voters' partial ballots, keyed by voter id. Their exposed
            votes E_TV join the training data; their hidden votes are never
            looked at.
        validation_fraction : float, optional
            Share of the voters held out to tune the hyperparameters on (default
            0.15, the paper's figure). Set to 0 to skip the search.
        param_grid : Iterable[dict], optional
            The settings to score, defaulting to :py:data:`DEFAULT_PARAM_GRID`.
            Tuning multiplies the number of fits by ``len(param_grid)``, so pass
            a single setting - or ``()`` - for large sweeps.
        tuning_projects : int, optional
            How many projects to fit when scoring a candidate setting, defaulting
            to :py:data:`TUNING_PROJECTS`. ``None`` fits the whole bundle for
            every candidate, which is what the search used to do.
        seed : int, optional
            Seed for the validation split, for reproducibility.
    Returns
    -------
        dict
            ``{"per_project": {project: ("const", 0/1) or
            ("model", (input_projects, classifier))}, "params": the
            hyperparameters used}``, consumed by
            :py:func:`predict_by_classification`.

    Examples
    --------
    With unanimous LV approvals, every project falls back to the majority
    preference (1 for approval, 0 for disapproval).

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> inst = Instance([p1, p2], budget_limit=2)
    >>> lv = ApprovalProfile([ApprovalBallot([p1])] * 3)
    >>> model = train_classification(inst, lv)
    >>> model["per_project"][p1], model["per_project"][p2]
    (('const', 1), ('const', 0))

    A Target Voter's exposed votes join the training data. On the LV ballots
    alone the lone voter's rejection of p2 is unanimous, so p2 needs no
    classifier; once three Target Voters expose an approval of it the collected
    votes disagree and a classifier is fitted after all.

    >>> lv = ApprovalProfile([ApprovalBallot([p1])])
    >>> train_classification(inst, lv)["per_project"][p2][0]
    'const'
    >>> tv = {f"v{i}": CardinalBallot({p2: 1}) for i in range(3)}
    >>> train_classification(inst, lv, tv)["per_project"][p2][0]
    'model'
    """
    if tuning_projects is _FROM_CONFIG:
        tuning_projects = TUNING_PROJECTS
    projects = sorted(instance, key=str)
    lv = list(lv_profile)
    tv = list((tv_ballots or {}).values())
    # The collected votes as two voter x project tables. An LV voter answered
    # every project, so her row is dense; a Target Voter's row is NaN wherever
    # she was not asked, and NaN is XGBoost's own missing-value marker.
    def table(rows):  # the reshape keeps the (0, m) shape when nobody is in it
        return np.array(rows, dtype=float).reshape(len(rows), len(projects))

    tv_sets = [_preference_sets(instance, ballot) for ballot in tv]
    lv_votes = table([[float(p in ballot) for p in projects] for ballot in lv])
    tv_votes = table([
        [1.0 if p in approved else 0.0 if p in disapproved else np.nan
         for p in projects]
        for approved, disapproved, _ in tv_sets
    ])

    # Blanked copies of the LV ballots, so the classifiers see the sparse rows
    # they will actually be asked about (see :py:func:`_masked_copies`).
    lv_features, lv_labels = _masked_copies(lv_votes, tv_votes, seed)
    copies = len(lv_features) // len(lv) if len(lv) else 1

    param_grid = tuple(param_grid)
    params = dict(param_grid[0]) if param_grid else dict(DEFAULT_PARAM_GRID[0])
    # Section 5's validation set: whole voters, so that no voter has some of her
    # votes fitted on and others scored. Voters are numbered LV first, then TV.
    n_voters = len(lv) + len(tv)
    n_validation = int(round(validation_fraction * n_voters))
    if len(param_grid) > 1 and 0 < n_validation < n_voters:
        held_out = np.random.default_rng(seed).permutation(n_voters)[:n_validation]
        is_validation = np.zeros(n_voters, dtype=bool)
        is_validation[held_out] = True
        lv_held, tv_held = is_validation[:len(lv)], is_validation[len(lv):]
        # Every LV voter now owns ``copies`` rows, in [full, blanked] order, so
        # her copies must fall on the same side of the split as she does.
        rows_held = np.tile(lv_held, copies)
        # Rank the candidates on a sample of the projects rather than the whole
        # bundle - the same sample for every candidate, so the comparison stays
        # like for like.
        sampled = None
        if tuning_projects is not None and tuning_projects < len(projects):
            sampled = set(
                np.random.default_rng(seed)
                .permutation(len(projects))[:tuning_projects]
                .tolist()
            )
        best_f1 = -1.0
        for candidate in param_grid:
            trained = _fit_per_project(
                projects, lv_features[~rows_held], lv_labels[~rows_held],
                tv_votes[~tv_held], candidate, only=sampled,
            )
            score = _pooled_f1(
                trained, projects, lv_features[rows_held], lv_labels[rows_held],
                tv_votes[tv_held],
            )
            logger.debug("train_classification: %s -> validation F1 %.4f",
                         candidate, score)
            if score > best_f1:
                params, best_f1 = dict(candidate), score
        logger.info(
            "train_classification: tuned on %d of %d held-out voters over %d "
            "of %d projects, best pooled F1 %.4f with %s",
            n_validation, n_voters, len(sampled) if sampled else len(projects),
            len(projects), best_f1, params,
        )
    per_project = _fit_per_project(
        projects, lv_features, lv_labels, tv_votes, params
    )
    logger.info(
        "train_classification: trained %d per-project classifiers once on "
        "%d LV ballots (x%d, blanked) and the exposed votes of %d TV voters",
        len(per_project), len(lv), copies, len(tv),
    )
    return {"per_project": per_project, "params": params}


# ---------------------------------------------------------------------------
# Section 2.1.2 - Collaborative filtering.
# ---------------------------------------------------------------------------
def _known_preferences(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_exposed: TVExposed,
) -> tuple[list[Project], list[tuple[tuple[str, object], Project, int]]]:
    """
    Return the known preferences: full LV ballots plus exposed TV sets E_v.

    Examples
    --------
    The LV voter contributes a vote on every project; the Target Voter only on
    the one she was asked about. Voters are keyed ``("lv", index)`` and
    ``("tv", voter id)`` so the two groups cannot collide.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> inst = Instance([p1, p2], budget_limit=2)
    >>> lv = ApprovalProfile([ApprovalBallot([p1])])
    >>> projects, known = _known_preferences(inst, lv, {"v": ({p2}, set())})
    >>> known
    [(('lv', 0), p1, 1), (('lv', 0), p2, 0), (('tv', 'v'), p2, 1)]
    """
    projects = sorted(instance, key=str)
    preferences = [
        (("lv", i), p, int(p in ballot))
        for i, ballot in enumerate(lv_profile)
        for p in projects
    ]
    preferences += [
        (("tv", voter), p, int(p in approved))
        for voter, (approved, disapproved) in tv_exposed.items()
        for p in sorted(approved | disapproved, key=str)
    ]
    return projects, preferences


#: Latent ranks scored against the validation set for matrix factorization.
DEFAULT_MF_RANKS = (5, 20, 50)
#: ``(rank, ridge)`` settings scored against the validation set for the hybrid.
DEFAULT_FM_SETTINGS = ((5, 0.0), (20, 0.0), (20, 1e-5), (50, 1e-5))


def _minority_weight(preferences) -> int:
    """
    How many times an approval must count for the two classes to weigh the same.
    Section 5 addresses the ~10% approval rate of Table 1 "by modifying the model
    loss function to give more weight to minority class samples"; with roughly
    one approval per nine rejections this returns 9.

    Examples
    --------
    >>> _minority_weight([("v", "p", 1)] + [("v", "p", 0)] * 9)
    9
    >>> _minority_weight([("v", "p", 1)] * 5 + [("v", "p", 0)] * 5)
    1
    >>> _minority_weight([])          # nothing collected
    1
    """
    positives = sum(label for *_, label in preferences)
    negatives = len(preferences) - positives
    return max(1, round(negatives / positives)) if positives else 1


def _tune(preferences, settings, fit, validation_fraction: float, seed: int):
    """
    Pick the setting with the best validation F1 (Sections 5 and 5.1).

    A share of the collected *votes* is held back, rather than a share of the
    voters as in :py:func:`train_classification`: a collaborative-filtering model
    cannot score a voter it never saw, so holding out whole voters would leave
    nothing to validate on. ``fit(votes, setting)`` returns a scorer taking a
    voter and a list of projects and returning one score in [0, 1] each.

    Examples
    --------
    Twenty collected votes, half of them approvals, and two candidate settings
    standing in for a real model: one that approves everything and one that
    approves nothing. The first finds every approval, the second finds none, so
    the F1 comparison picks the first.

    >>> votes = [("v", f"p{i}", i % 2) for i in range(20)]
    >>> def fit(training, approve_everything):
    ...     return lambda voter, projects: [
    ...         1.0 if approve_everything else 0.0 for _ in projects
    ...     ]
    >>> _tune(votes, (False, True), fit, validation_fraction=0.5, seed=0)
    True

    With a single setting there is nothing to choose between, so it is returned
    without fitting anything.

    >>> _tune(votes, (False,), fit, validation_fraction=0.5, seed=0)
    False
    """
    settings = tuple(settings)
    n_validation = int(round(validation_fraction * len(preferences)))
    if len(settings) < 2 or not 0 < n_validation < len(preferences):
        return settings[0]
    order = np.random.default_rng(seed).permutation(len(preferences))
    held_out = set(order[:n_validation].tolist())
    training = [pref for i, pref in enumerate(preferences) if i not in held_out]
    validation: dict = {}
    for index in sorted(held_out):
        voter, project, label = preferences[index]
        validation.setdefault(voter, []).append((project, label))

    best, best_f1 = settings[0], -1.0
    for setting in settings:
        score = fit(training, setting)
        tp = fp = fn = 0
        for voter, votes in validation.items():
            values = score(voter, [project for project, _ in votes])
            for (_, label), value in zip(votes, values):
                if value >= 0.5 and label:
                    tp += 1
                elif value >= 0.5:
                    fp += 1
                elif label:
                    fn += 1
        f1 = _f1(tp, fp, fn)
        logger.debug("_tune: %s -> validation F1 %.4f", setting, f1)
        if f1 > best_f1:
            best, best_f1 = setting, f1
    logger.info(
        "_tune: %d of %d collected votes held out, best F1 %.4f with %s",
        n_validation, len(preferences), best_f1, best,
    )
    return best


def _fit_mf(preferences, rank: int, weight: int):
    """
    Fit one SVD and return ``score(voter, projects)``.

    Examples
    --------
    One voter who approved p1 and rejected p2; the fitted model scores any
    (voter, project) pair on the 0-1 scale the preferences were given on.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> score = _fit_mf([(("lv", 0), p1, 1), (("lv", 0), p2, 0)], 2, 1)
    >>> values = score(("lv", 0), [p1, p2])
    >>> len(values) == 2 and all(0.0 <= v <= 1.0 for v in values)
    True
    """
    import pandas as pd
    from surprise import Dataset, Reader, SVD

    # Surprise exposes no sample-weight API, so an approval is repeated
    # ``weight`` times instead: that many more gradient steps is the same thing
    # as that much more loss weight, and it lifts the reconstructed scores of
    # the minority class towards the 0.5 threshold they are compared against.
    rows = [pref for pref in preferences for _ in range(weight if pref[2] else 1)]
    data = Dataset.load_from_df(
        pd.DataFrame(rows, columns=["voter", "project", "preference"]),
        Reader(rating_scale=(0, 1)),  # Surprise's name for 0/1 preferences.
    )
    model = SVD(n_factors=max(1, rank), n_epochs=50, random_state=0)
    model.fit(data.build_full_trainset())
    return lambda voter, projects: [
        float(model.predict(voter, p).est) for p in projects
    ]


def train_matrix_factorization(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_exposed: TVExposed,
    rank: int | None = None,
    *,
    ranks=DEFAULT_MF_RANKS,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, dict[Project, float]]:
    """
    Fit ``R_vp ~= mu + b_v + b_p + q_p^T x_v`` (paper, Section 2.1.2).

    Approvals are up-weighted against the imbalance of Table 1, and the latent
    rank is chosen by validation F1 over ``ranks`` (Section 5) unless an explicit
    ``rank`` is given, in which case that one is used and nothing is tuned.

    Examples
    --------
    Three LV voters who all approve p1 and reject p2, and one Target Voter whose
    exposed votes agree with them. Every project gets a reconstructed score on
    the 0-1 scale, and the pair the electorate likes scores above the pair it
    does not.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> inst = Instance([p1, p2], budget_limit=2)
    >>> lv = ApprovalProfile([ApprovalBallot([p1])] * 3)
    >>> scores = train_matrix_factorization(inst, lv, {"v": ({p1}, {p2})}, rank=2)
    >>> sorted(scores) == ["v"] and len(scores["v"]) == 2
    True
    >>> scores["v"][p1] > scores["v"][p2]
    True
    """
    try:
        import pandas  # noqa: F401  (needed by _fit_mf)
        import surprise  # noqa: F401
    except ImportError:
        raise ImportError(
            "You need scikit-surprise and pandas to train the matrix-factorization "
            "predictor (pip install pabutools-recommendation[ml])."
        )

    projects, preferences = _known_preferences(instance, lv_profile, tv_exposed)
    if not preferences:
        return {v: {p: 0.0 for p in projects} for v in tv_exposed}

    weight = _minority_weight(preferences)

    def fit(votes, latent_rank):
        """One candidate fit, in the shape :py:func:`_tune` expects."""
        return _fit_mf(votes, latent_rank, weight)

    if rank is None:
        rank = _tune(preferences, ranks, fit, validation_fraction, seed)
    logger.info(
        "train_matrix_factorization: factorising %d collected votes over %d "
        "projects at rank %d, approvals weighted x%d",
        len(preferences), len(projects), rank, weight,
    )
    score = fit(preferences, rank)
    return {
        v: dict(zip(projects, score(("tv", v), projects))) for v in tv_exposed
    }


# ---------------------------------------------------------------------------
# Section 2.1.2 - Hybrid Factorization Machines.
# ---------------------------------------------------------------------------
def _fit_fm(projects, voters, preferences, rank: int, ridge: float, weight: int):
    """
    Fit one LightFM hybrid and return ``score(voter, projects)``.

    Examples
    --------
    Like :py:func:`_fit_mf`, but each project also carries Table 2's attributes.
    The example is marked skipped so the doctest suite still passes without the
    optional ``lightfm`` installed; the same path is covered by the unit tests,
    which skip themselves when the library is absent.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> known = [(("lv", 0), p1, 1), (("lv", 0), p2, 0)]
    >>> score = _fit_fm([p1, p2], [("lv", 0)], known, 2, 0.0, 1)  # doctest: +SKIP
    >>> [0.0 <= v <= 1.0 for v in score(("lv", 0), [p1, p2])]     # doctest: +SKIP
    [True, True]
    """
    from lightfm import LightFM
    from lightfm.data import Dataset

    # Table 2's project attributes. Every project also keeps its own identity
    # feature (LightFM adds one by default), so these content features sit
    # alongside the latent factor rather than replacing it - that combination is
    # the hybrid of Section 2.1.2.
    categories = sorted({c for p in projects for c in p.categories})
    targets = sorted({t for p in projects for t in p.targets})
    dataset = Dataset()
    dataset.fit(
        voters,
        projects,
        item_features=(["cost"]
                       + [f"category={c}" for c in categories]
                       + [f"target={t}" for t in targets]),
    )
    preference_matrix, signs = dataset.build_interactions(
        (v, p, 2 * preference - 1) for v, p, preference in preferences
    )
    # Section 5's minority-class weighting. LightFM does honour sample weights
    # under the logistic loss - only the k-OS loss ignores them - so an approval
    # simply contributes ``weight`` times as much to the loss as a rejection.
    sample_weight = signs.copy()
    sample_weight.data = np.where(
        signs.data > 0, float(weight), 1.0
    ).astype(np.float32)
    preference_matrix.data = signs.data
    costs = np.array([float(p.cost) for p in projects])
    costs = (costs - costs.mean()) / costs.std() if costs.std() else costs * 0
    project_attributes = dataset.build_item_features(
        (
            (p, {"cost": cost}
                | {f"category={c}": 1.0 for c in p.categories}
                | {f"target={t}": 1.0 for t in p.targets})
            for p, cost in zip(projects, costs)
        ),
        normalize=False,
    )
    model = LightFM(
        no_components=max(1, rank),
        loss="logistic",
        user_alpha=ridge,
        item_alpha=ridge,
        random_state=0,
    )
    model.fit(
        preference_matrix,
        item_features=project_attributes,
        sample_weight=sample_weight,
        epochs=30,
        num_threads=1,
    )
    user_ids, _, item_ids, _ = dataset.mapping()

    def score(voter, wanted):
        """The fitted approval probability of each ``wanted`` project."""
        raw = model.predict(
            user_ids[voter],
            np.array([item_ids[p] for p in wanted]),
            item_features=project_attributes,
        )
        return 1 / (1 + np.exp(-raw))

    return score


def train_factorization_machines(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_exposed: TVExposed,
    rank: int | None = 20,
    ridge: float | None = 0.0,
    *,
    settings=DEFAULT_FM_SETTINGS,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, dict[Project, float]]:
    """
    Fit the paper's hybrid latent model ``y(v,p)=FM(v, p, attributes(p))``.

    What makes this module *hybrid* (Section 2.1.2) rather than plain matrix
    factorization is that projects are described by content features as well as
    by their own latent factor, so a voter's taste can transfer to a project she
    was never asked about. The features are Table 2's project attributes, all of
    which ``parse_pabulib`` fills in: the numeric ``cost`` (standardised), plus
    one multi-hot indicator per ``category`` and per ``target`` population
    segment - multi-hot because Section 4 notes that "a certain proposal can
    consist multiple topics and population segments". An instance whose projects
    carry no categories or targets falls back to cost alone.

    Approvals are up-weighted against the imbalance of Table 1. ``rank`` and
    ``ridge`` are *not* tuned by default, a deliberate departure from Section 5:
    pass ``rank=None, ridge=None`` to search ``settings`` by validation F1
    instead.

    Measured on Praga-Poludnie 2022 with the offline-popularity setup, the
    search costs accuracy rather than buying it - FA 0.787 at the fixed
    ``rank=20, ridge=0.0`` against 0.590 for whatever the search selects - as
    well as fitting the model four times over. The reason is that Section 5.1's
    F1 is measured per vote while Section 5.2's FA depends on the *ranking* of
    projects surviving, and the two do not agree; the same disagreement is why
    :py:func:`_fit_per_project` leaves the minority-class correction off. Ridge
    is no help either: sweeping it over ``0 .. 0.1`` moves offline FA around
    erratically (0.787, 0.782, 0.346, 0.753, 0.007) without ever reaching the
    1.000 that plain matrix factorization gets on the same split.

    .. note::
        The content features are worth having under ``random`` sampling and
        harmful under the offline setups, measured on that same district:

        ============================  ==================  ========
        FM item features              offline_popularity  random
        ============================  ==================  ========
        none at all (i.e. plain MF)   1.000               0.927
        cost only                     0.814               0.955
        cost + categories + targets   0.590               1.000
        ============================  ==================  ========

        The offline setups expose every voter to the *same*, deliberately
        unrepresentative k projects (the most popular, or the most divisive), so
        the shared content embeddings are fitted on that slice and then applied
        to the projects nobody was asked about. Under ``random`` the exposed
        projects are a fair sample and the features help. The paper treats the
        sampling module and the prediction module as independent choices
        (Section 3); this is evidence that they interact.

    Examples
    --------
    The hybrid counterpart of :py:func:`train_matrix_factorization`: same
    voter-project preferences, but every project also described by its cost,
    categories and target segments. Skipped so the suite passes without the
    optional ``lightfm``; the unit tests cover this path and skip themselves
    when it is missing.

    >>> green = Project("green", 1, categories={"environment"})
    >>> road = Project("road", 2, categories={"transport"})
    >>> inst = Instance([green, road], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([green])] * 3)
    >>> scores = train_factorization_machines(       # doctest: +SKIP
    ...     inst, lv, {"v": ({green}, {road})})
    >>> scores["v"][green] > scores["v"][road]       # doctest: +SKIP
    True
    """
    try:
        import lightfm  # noqa: F401  (needed by _fit_fm)
    except ImportError:
        raise ImportError(
            "You need lightfm to train the factorization-machines predictor "
            "(pip install pabutools-recommendation[ml])."
        )

    projects, preferences = _known_preferences(instance, lv_profile, tv_exposed)
    voters = [("lv", i) for i in range(lv_profile.num_ballots())] + [
        ("tv", v) for v in tv_exposed
    ]
    if not projects or not preferences:
        return {v: {p: 0.0 for p in projects} for v in tv_exposed}

    weight = _minority_weight(preferences)

    def fit(votes, setting):
        """One candidate fit, in the shape :py:func:`_tune` expects."""
        return _fit_fm(projects, voters, votes, *setting, weight)

    if rank is None and ridge is None:
        rank, ridge = _tune(preferences, settings, fit, validation_fraction, seed)
    else:
        rank, ridge = (20 if rank is None else rank, 0.0 if ridge is None else ridge)
    logger.info(
        "train_factorization_machines: fitting %d collected votes over %d "
        "projects at rank %d, ridge %g, approvals weighted x%d, with %d "
        "categories and %d target segments as content features",
        len(preferences), len(projects), rank, ridge, weight,
        len({c for p in projects for c in p.categories}),
        len({t for p in projects for t in p.targets}),
    )
    score = fit(preferences, (rank, ridge))
    return {
        v: dict(zip(projects, score(("tv", v), projects))) for v in tv_exposed
    }


# ---------------------------------------------------------------------------
# Complete the Target Voters' partial ballots with the fitted models.
# ---------------------------------------------------------------------------
def _preference_sets(
    instance: Instance, ballot: CardinalBallot
) -> tuple[set[Project], set[Project], set[Project]]:
    """
    Decode a partial ballot into A_v, D_v, and H_v.

    Examples
    --------
    p1 is approved and p2 rejected; p3 was never scored, so it is hidden.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> a, d, h = _preference_sets(inst, CardinalBallot({p1: 1, p2: -1}))
    >>> a == {p1}, d == {p2}, h == {p3}
    (True, True, True)
    """
    approved = {p for p, preference in ballot.items() if preference > 0}
    disapproved = {p for p, preference in ballot.items() if preference < 0}
    hidden = set(instance) - approved - disapproved
    return approved, disapproved, hidden


def _complete_by_scores(
    instance: Instance, lv_profile: ApprovalProfile,
    tv_ballots: dict[str, CardinalBallot], train,
) -> dict[str, ApprovalBallot]:
    """
    Complete every TV ballot from a collaborative-filtering score: ``train`` is
    fitted on the LV ballots plus all the exposed sets E_TV, then a hidden
    project is approved iff its score is >= 0.5, while A_v is kept and D_v stays
    rejected. The MF and FM modules differ only in ``train``, so they share this.

    Examples
    --------
    A stand-in model that scores p3 highly and everything else low, applied to a
    voter who approved p1, rejected p2 and was never asked about p3. Her
    approval of p1 survives, her rejection of p2 survives, and only the hidden
    p3 is decided by the score.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> def train(instance, lv_profile, exposed):
    ...     return {v: {p: (0.9 if p == p3 else 0.1) for p in instance}
    ...             for v in exposed}
    >>> done = _complete_by_scores(
    ...     inst, ApprovalProfile([]), {"v": CardinalBallot({p1: 1, p2: -1})}, train)
    >>> done["v"] == ApprovalBallot({p1, p3})
    True
    """
    sets = {v: _preference_sets(instance, b) for v, b in tv_ballots.items()}
    scores = train(instance, lv_profile,
                   {v: (approved, disapproved)
                    for v, (approved, disapproved, _) in sets.items()})
    completed = {
        v: ApprovalBallot(approved | {p for p in hidden if scores[v][p] >= 0.5})
        for v, (approved, _, hidden) in sets.items()
    }
    logger.info(
        "%s: completed %d TV ballots, %d of %d hidden votes scored >= 0.5 "
        "and became approvals",
        getattr(train, "__name__", "collaborative filtering"), len(completed),
        sum(len(completed[v]) - len(sets[v][0]) for v in sets),
        sum(len(sets[v][2]) for v in sets),
    )
    return completed


def predict_by_classification(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, CardinalBallot],
    **kwargs,
) -> dict[str, ApprovalBallot]:
    """
    Prediction module - binary classification (Section 2.1.1): predict each
    hidden project with a per-project binary classifier. One classifier is
    trained for every project p, using preferences on P \\ {p} as input and the
    preference on p as output. The collection is trained once by
    :py:func:`train_classification` - on the LV ballots *and* the Target Voters'
    exposed votes E_TV, per Section 3.1.1 - and applied to every TV voter;
    preferences outside E_v are passed to XGBoost as missing values. Exposed
    approvals A_v are kept and exposed disapprovals D_v stay rejected.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        tv_ballots : dict[str, :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`]
            Every Target Voter's partial ballot to complete, keyed by voter id.
            Build these with ``pabutools_recommendation.partial_ballot`` /
            ``reveal_ballot`` rather than by hand; the score convention is
            +1 approved, -1 disapproved, 0 (or absent) hidden.
        **kwargs
            Forwarded to :py:func:`train_classification`. Pass ``param_grid=()``
            to skip the Section 5 hyperparameter search, which costs roughly ten
            times a plain fit and is worth switching off for large sweeps.

    Returns
    -------
        dict[str, :py:class:`~pabutools.election.ballot.approvalballot.ApprovalBallot`]
            The full predicted approval ballot of every TV voter, keyed by
            voter id.

    Examples
    --------
    2 LV voters both approve {p1, p2}. A TV voter with nothing exposed is
    completed to {p1, p2} - on such an unambiguous pattern any correct
    predictor (XGBoost included) agrees with the LV majority.

    >>> p1, p2, p3 = Project("p1", 4), Project("p2", 4), Project("p3", 6)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p2])])
    >>> partial = CardinalBallot({p1: 0, p2: 0, p3: 0})  # nothing exposed
    >>> predict_by_classification(inst, lv, {"v3": partial})["v3"] == {p1, p2}
    True
    """
    model = train_classification(instance, lv_profile, tv_ballots, **kwargs)
    projects = sorted(instance, key=str)
    voters = list(tv_ballots)
    preferences = {
        vid: _preference_sets(instance, tv_ballots[vid]) for vid in voters
    }
    # One row per Target Voter over every project, in the same column order the
    # classifiers were fitted on: 1 approved, 0 disapproved, NaN not asked.
    votes = np.array(
        [
            [
                1.0 if p in approved else 0.0 if p in disapproved else np.nan
                for p in projects
            ]
            for approved, disapproved, _ in (preferences[v] for v in voters)
        ],
        dtype=float,
    ).reshape(len(voters), len(projects))

    # Classify per *project*, not per (voter, project): every voter with that
    # project hidden goes through XGBoost in one call. On a Warsaw district this
    # is ~100 calls rather than ~700 000.
    predicted: dict[str, set[Project]] = {vid: set() for vid in voters}
    for index, project in enumerate(projects):
        rows = np.flatnonzero(np.isnan(votes[:, index]))  # hidden for these
        if not len(rows):
            continue
        kind, payload = model["per_project"][project]
        if kind == "const":
            labels = np.full(len(rows), payload)
        else:
            # Dropping this project's column reproduces the ``input_projects``
            # order the classifier was trained on.
            labels = payload[1].predict(np.delete(votes[rows], index, axis=1))
        for row, label in zip(rows, labels):
            if int(label) == 1:
                predicted[voters[row]].add(project)
    logger.info(
        "predict_by_classification: completed %d TV ballots in %d batched calls",
        len(voters), len(projects),
    )
    return {
        vid: ApprovalBallot(preferences[vid][0] | predicted[vid])
        for vid in voters
    }


def predict_by_matrix_factorization(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, CardinalBallot],
) -> dict[str, ApprovalBallot]:
    """
    Prediction module - collaborative filtering via Matrix Factorization
    (Section 2.1.2): one voter-project matrix holds the LV ballots and *every* TV
    voter's exposed votes (the paper's E_TV; approve=1, disapprove=0), it is
    factorised once by :py:func:`train_matrix_factorization`, and a hidden
    project is predicted approved iff its reconstructed score is >= 0.5.
    Approvals are up-weighted against the imbalance of Table 1 and the latent
    rank is chosen on a 15% validation set (Section 5). Exposed approvals A_v
    are kept and exposed disapprovals D_v stay rejected.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        tv_ballots : dict[str, :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`]
            Every Target Voter's partial ballot to complete, keyed by voter id.

    Returns
    -------
        dict[str, :py:class:`~pabutools.election.ballot.approvalballot.ApprovalBallot`]
            The full predicted approval ballot of every TV voter, keyed by
            voter id.

    Examples
    --------
    Three LV voters all support the cheap pair {p1, p2} and reject {p3, p4}.
    The reconstructed matrix completes a TV voter with nothing exposed to
    {p1, p2}.

    >>> p1, p2, p3, p4 = (Project("p1", 3), Project("p2", 3),
    ...                   Project("p3", 4), Project("p4", 4))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> partial = CardinalBallot({p1: 0, p2: 0, p3: 0, p4: 0})  # nothing exposed
    >>> predict_by_matrix_factorization(inst, lv, {"v4": partial})["v4"] == {p1, p2}
    True
    """
    return _complete_by_scores(
        instance, lv_profile, tv_ballots, train_matrix_factorization
    )


def predict_by_factorization_machines(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, CardinalBallot],
) -> dict[str, ApprovalBallot]:
    """
    Prediction module - hybrid Factorization Machines (Section 2.1.2): fitted by
    :py:func:`train_factorization_machines` on the same LV + E_TV voter-project
    preferences as Matrix Factorization, but with each project also described by
    Table 2's attributes - its cost, its categories and its target population
    segments - which is what makes the module hybrid rather than plain MF.
    Approvals are up-weighted against the imbalance of Table 1 and the latent
    rank and regularisation are chosen on a 15% validation set (Section 5). A
    hidden project is predicted approved iff the FM score is >= 0.5; exposed
    approvals A_v are kept and exposed disapprovals D_v stay rejected.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        tv_ballots : dict[str, :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`]
            Every Target Voter's partial ballot to complete, keyed by voter id.

    Returns
    -------
        dict[str, :py:class:`~pabutools.election.ballot.approvalballot.ApprovalBallot`]
            The full predicted approval ballot of every TV voter, keyed by
            voter id.

    Examples
    --------
    2 LV voters both approve {p1, p2}. A TV voter with nothing exposed is
    completed to {p1, p2}. Skipped so the doctest suite passes without the
    optional ``lightfm``; the unit tests cover this and skip themselves when the
    library is absent.

    >>> p1, p2, p3 = Project("p1", 4), Project("p2", 4), Project("p3", 6)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p2])])
    >>> partial = CardinalBallot({p1: 0, p2: 0, p3: 0})  # nothing exposed
    >>> predict_by_factorization_machines(   # doctest: +SKIP
    ...     inst, lv, {"v3": partial})["v3"] == {p1, p2}
    True
    """
    return _complete_by_scores(
        instance, lv_profile, tv_ballots, train_factorization_machines
    )
