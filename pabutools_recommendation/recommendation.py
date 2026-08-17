"""
An implementation of the algorithms in:
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

Programmer: Roei Yanku
Date: 2026-06-20.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    CardinalBallot,
    total_cost,
)
from pabutools.election.satisfaction import Cost_Sat
from pabutools.rules import BudgetAllocation, greedy_utilitarian_welfare

# All model fitting and prediction lives in ``model_training``; the three
# Section 2.1 prediction modules are used here as the ``predict`` argument of
# the pipeline and are listed in ``PREDICTORS``.
from pabutools_recommendation.model_training import (
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
)

# Log the steps of every algorithm at INFO/DEBUG level.
#  Configure a handler in your own script to see them, e.g.
# ``logging.basicConfig(level=logging.INFO)``.
logger = logging.getLogger(__name__)


# ===========================================================================
# Section 2.3 - Partial (three-state) ballots, encoded as a CardinalBallot.
# ===========================================================================
# A Target Voter answers only k projects, so "unknown" must be distinct from
# "disapproved" - which a plain approval set (just the approved projects) cannot
# express. On the pabutools maintainer's advice (Simon Rey) we do NOT add a new
# ballot type; instead we reuse the existing
# :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot` (a dict of
# project -> score) under this sign convention:
#
#     score > 0   ->  APPROVED     (A_v)   we store +1  (APPROVAL)
#     score < 0   ->  DISAPPROVED  (D_v)   we store -1  (DISAPPROVAL)
#     score == 0  ->  HIDDEN       (H_v)   we store  0  (HIDDEN)
#
# A project simply absent from the ballot is also HIDDEN. This convention is not
# self-evident, so never hand-write the scores: build ballots with
# ``partial_ballot`` / ``reveal_ballot`` and read them back with the
# ``*_projects`` accessors / ``as_approval_ballot``.

#: Score stored for an approved project (A_v). Any strictly positive score works.
APPROVAL = 1
#: Score stored for a disapproved project (D_v). Any strictly negative score works.
DISAPPROVAL = -1
#: Score stored for a hidden project (H_v); absent projects mean the same.
HIDDEN = 0


def partial_ballot(
    approved: Iterable[Project] = (),
    disapproved: Iterable[Project] = (),
    hidden: Iterable[Project] = (),
) -> CardinalBallot:
    """
    Build a partial ballot from the three explicit sets, hiding the convention:
    approved projects get +1, disapproved get -1, and hidden get 0. This is the
    intended way to create a partial ballot - callers name the three sets instead
    of writing raw scores.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> b[p1], b[p2], b[p3]
    (1, -1, 0)
    >>> exposed_projects(b) == {p1, p2}
    True
    """
    return CardinalBallot(
        {p: APPROVAL for p in approved}
        | {p: DISAPPROVAL for p in disapproved}
        | {p: HIDDEN for p in hidden}
    )


def reveal_ballot(
    instance: Instance,
    full_ballot: set[Project],
    exposed: set[Project],
) -> CardinalBallot:
    """
    Build the partial ballot exposing a set of projects of a voter whose full
    ballot in the ideal instance is known (Section 2.3): A_v = E_v ∩ full_ballot,
    D_v = E_v \\ full_ballot, H_v = P \\ E_v.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (used as the universe of projects P).
        full_ballot : set[:py:class:`~pabutools.election.instance.Project`]
            The voter's full ballot - her approval set A_v in the ideal instance.
        exposed : set[:py:class:`~pabutools.election.instance.Project`]
            The projects revealed for this voter (the exposed set E_v).

    Returns
    -------
        :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`
            The corresponding three-state ballot under the sign convention.

    Examples
    --------
    Example 2.4 from the paper: P = {p1, p2, p3, p4}, the voter's full ballot is
    {p1, p2}, and {p1, p3} is exposed. Then p1 is approved, p3 disapproved, and
    p2, p4 remain hidden.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> b = reveal_ballot(inst, {p1, p2}, {p1, p3})
    >>> approved_projects(b) == {p1}, disapproved_projects(b) == {p3}
    (True, True)
    >>> hidden_projects(b, inst) == {p2, p4}
    True
    """
    return partial_ballot(
        approved=exposed & full_ballot,       # A_v = E_v ∩ full ballot
        disapproved=exposed - full_ballot,    # D_v = E_v \ full ballot
        hidden=set(instance) - exposed,       # H_v = P \ E_v
    )


def approved_projects(ballot: CardinalBallot) -> set[Project]:
    """
    The approval set A_v: the projects with a strictly positive score.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> approved_projects(b) == {p1}
    True
    """
    return {project for project, score in ballot.items() if score > 0}


def disapproved_projects(ballot: CardinalBallot) -> set[Project]:
    """
    The disapproval set D_v: the projects with a strictly negative score.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> disapproved_projects(b) == {p2}
    True
    """
    return {project for project, score in ballot.items() if score < 0}


def exposed_projects(ballot: CardinalBallot) -> set[Project]:
    """
    The exposed set E_v = A_v ∪ D_v: the projects with a non-zero score.

    Examples
    --------
    The hidden p3 is left out, whichever way the other two went.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> exposed_projects(b) == {p1, p2}
    True
    """
    return {project for project, score in ballot.items() if score != 0}


def hidden_projects(ballot: CardinalBallot, instance: Instance) -> set[Project]:
    """
    The hidden set H_v = P \\ E_v: every project of the instance that the voter
    was not asked about (score 0 or absent from the ballot).

    Examples
    --------
    p3 scores 0 and p4 is absent from the ballot altogether; both count as
    hidden, so the voter is treated as never having been asked about either.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> hidden_projects(b, inst) == {p3, p4}
    True
    """
    return set(instance) - exposed_projects(ballot)


def as_approval_ballot(ballot: CardinalBallot) -> ApprovalBallot:
    """
    The pabutools approval ballot made of the approved projects A_v, so that a
    (completed) partial ballot can be fed to approval-based voting rules.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> as_approval_ballot(b) == ApprovalBallot({p1})
    True
    """
    return ApprovalBallot(approved_projects(ballot))


# ---------------------------------------------------------------------------
# Section 2.2.3 - Popularity and consensus (primitives used by every module).
# ---------------------------------------------------------------------------
# Definition 2.1 (Approval scores) needs no function of our own: it is exactly
# pabutools' ``profile.approval_scores()``. Callers below use the library method
# directly (``.get(p, 0)`` supplies the 0 for projects that nobody approved).
def consensus_levels(
    instance: Instance, profile: ApprovalProfile
) -> dict[Project, int]:
    """
    Definition 2.2 (Consensus levels): the absolute difference between the number
    of voters approving a project and the number disapproving it (so a unanimous
    approval or a unanimous rejection both reach the maximum n).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The approval profile.

    Returns
    -------
        dict[:py:class:`~pabutools.election.instance.Project`, int]
            A mapping from each project to its consensus level.

    Examples
    --------
    Four LV voters: p1 approved 4/0, p2 split 2/2, p3 rejected 0/4. Both p1
    and p3 are in full consensus, p2 in none.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                         ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> c = consensus_levels(inst, prof)
    >>> [c[p] for p in (p1, p2, p3)]
    [4, 0, 4]
    """
    # consensus(p) = |approvers(p) - disapprovers(p)|. With n voters and
    # score(p) approvers, disapprovers = n - score(p), so the difference is
    # |score - (n - score)| = |2*score - n|. Scores come from the library
    # (Definition 2.1 = pabutools' approval_scores()).
    n = profile.num_ballots()
    scores = profile.approval_scores()
    consensus = {
        project: abs(2 * scores.get(project, 0) - n) for project in instance
    }
    logger.debug("consensus_levels (n=%d): %s", n, consensus)
    return consensus


def most_consensual_projects(
    instance: Instance, profile: ApprovalProfile
) -> list[Project]:
    """
    The paper's ordering gamma (Section 2.2.3): the projects sorted by decreasing
    consensus level, ties broken lexicographically by project name. gamma_1 is the
    project most in consensus, gamma_m the "most controversial". Helper used by
    offline revealing-by-consensus and by-controversiality.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                         ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> most_consensual_projects(inst, prof)  # consensus p1=4, p3=4, p2=0
    [p1, p3, p2]
    """
    consensus = consensus_levels(instance, profile)
    gamma = sorted(instance, key=lambda project: (-consensus[project], str(project)))
    if gamma:
        logger.debug(
            "most_consensual_projects: gamma_1=%s (consensus %d), "
            "gamma_m=%s (consensus %d, the most controversial)",
            gamma[0], consensus[gamma[0]], gamma[-1], consensus[gamma[-1]],
        )
    return gamma


# ---------------------------------------------------------------------------
# Section 2.2.5 - The voting rule (greedy approval).
# ---------------------------------------------------------------------------
def greedy_approval(
    instance: Instance, profile: ApprovalProfile
) -> BudgetAllocation:
    """
    Greedy approval voting rule (Section 2.2.5). Projects are considered in
    decreasing order of their approval score and funded while the budget
    allows; a project that would exceed the remaining budget is skipped. Ties
    are broken lexicographically by project name.

    .. note::
        Delegates to pabutools'
        :py:func:`~pabutools.rules.greedywelfare.greedy_utilitarian_welfare` with
        :py:class:`~pabutools.election.satisfaction.additivesatisfaction.Cost_Sat`:
        the rule ranks by marginal satisfaction divided by cost, and under
        Cost_Sat a project's marginal satisfaction is score(p)*cost(p), so the
        ranking is by the **raw** approval score - exactly the paper's greedy
        approval. (With the default Cardinality_Sat the ranking would be
        score/cost, which is a different rule.)

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The (completed) approval profile.

    Returns
    -------
        :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            The winning bundle.

    Examples
    --------
    Example 2.3 from the paper: scores p1 = p2 = 2, p3 = 1, budget 3. p1 and p2
    are funded (cost 1 each); p3 (cost 2) no longer fits.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 2)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]),
    ...                         ApprovalBallot([p1, p3]),
    ...                         ApprovalBallot([p2])])
    >>> sorted(greedy_approval(inst, prof), key=str)
    [p1, p2]
    """
    chosen = greedy_utilitarian_welfare(
        instance, profile, sat_class=Cost_Sat, resoluteness=True
    )
    assert isinstance(chosen, BudgetAllocation)  # resolute mode: one allocation
    logger.info(
        "greedy_approval: funded %d/%d projects, cost %s of %s",
        len(chosen), len(instance), total_cost(chosen), instance.budget_limit,
    )
    return chosen


# ---------------------------------------------------------------------------
# Section 3.1.1 - Random setup.
# ---------------------------------------------------------------------------
def random_setup(
    instance: Instance,
    k: int,
    seed: int | None = None,
) -> set[Project]:
    """
    Random setup (Sections 2.4 / 3.1.1): the exposed set E_v of one Target
    Voter, made of k projects chosen uniformly at random from P (the rest is
    predicted later). Called once per TV voter with its own seed, so each
    voter gets an independent draw ("possibly different E for each voter");
    it needs no ballot, since the choice is random.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (the universe of projects P).
        k : int
            The number of projects to expose.
        seed : int, optional
            Seed for the random generator, for reproducibility.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E_v: k projects drawn uniformly at random from P.

    Examples
    --------
    With k = 2, whatever the draw, exactly two projects are exposed and they
    are real projects of the instance.

    >>> p1, p2, p3, p4 = (Project("p1", 2), Project("p2", 2),
    ...                   Project("p3", 3), Project("p4", 3))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> exposed = random_setup(inst, k=2, seed=0)
    >>> len(exposed) == 2 and exposed <= {p1, p2, p3, p4}
    True
    """
    rng = random.Random(seed)
    # Sort first so the sample is reproducible from the seed (a set has no
    # deterministic iteration order).
    population = sorted(instance, key=str)
    exposed = set(rng.sample(population, k))
    logger.info("random_setup: exposed %d of %d projects at random", k, len(population))
    return exposed


# ---------------------------------------------------------------------------
# Section 3.1.2 - Offline setup (popularity / consensus / controversiality).
# ---------------------------------------------------------------------------
def offline_popularity(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> set[Project]:
    """
    Offline revealing by popularity (Section 3.1.2). The exposed
    set E = {sigma_1, ..., sigma_k}: the k most approved projects among the LV
    voters. The same set is exposed to every Target Voter, so it depends only on
    the LV profile (the score ordering sigma is used only to pick the top k; the
    exposed set itself is unordered).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        k : int
            The number of projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E of the k most popular projects (ties broken by name
            when picking the top k).

    Examples
    --------
    LV scores p1 = 3, p2 = 1, p3 = 1, so the single most popular project is p1.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p3]),
    ...                       ApprovalBallot([p1, p2]),
    ...                       ApprovalBallot([p1])])
    >>> offline_popularity(inst, lv, k=1) == {p1}
    True
    """
    # The paper's sigma ordering: decreasing library approval score, ties by name.
    scores = lv_profile.approval_scores()
    sigma = sorted(instance, key=lambda p: (-scores.get(p, 0), str(p)))
    exposed = set(sigma[:k])
    logger.info("offline_popularity: exposing top-%d popular projects", k)
    return exposed


def offline_consensus(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> set[Project]:
    """
    Offline revealing by consensus (Section 3.1.2). The exposed set
    E = {gamma_1, ..., gamma_k}: the k projects with the highest consensus level
    among the LV voters. The same set is exposed to every Target Voter.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        k : int
            The number of projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E of the k projects most in consensus (ties broken by
            name when picking the top k).

    Examples
    --------
    p1 and p3 both have consensus 4 (p1 unanimously approved, p3 unanimously
    rejected); the tie is broken by name towards p1.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                       ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> offline_consensus(inst, lv, k=1) == {p1}
    True
    """
    exposed = set(most_consensual_projects(instance, lv_profile)[:k])
    logger.info("offline_consensus: exposing top-%d consensual projects", k)
    return exposed


def offline_controversiality(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> set[Project]:
    """
    Offline revealing by controversiality (Section 3.1.2). The exposed set
    E = {gamma_{m-k+1}, ..., gamma_m}: the k projects *least* in consensus
    among the LV voters (the hardest to predict, so asked directly). The same
    set is exposed to every Target Voter. (The paper writes
    {gamma_{m-k}, ..., gamma_m}, which is k+1 projects - an off-by-one; its
    prose says "the k projects least in consensus", implemented here.)

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        k : int
            The number of projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E of the k most controversial projects (ties broken
            by name when picking the bottom k).

    Examples
    --------
    Same data as the consensus example: p2 is split exactly in half
    (consensus 0) and is therefore the single most controversial project.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                       ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> offline_controversiality(inst, lv, k=1) == {p2}
    True
    """
    # The k *least* consensual projects are the last k of gamma,
    # {gamma_{m-k+1}, ..., gamma_m}. Slicing from m-k keeps k=0 correct.
    gamma = most_consensual_projects(instance, lv_profile)
    exposed = set(gamma[len(gamma) - k:])
    logger.info("offline_controversiality: exposing bottom-%d consensual projects", k)
    return exposed


# ---------------------------------------------------------------------------
# Section 3.1.3 - Online setup (adaptive controversial).
# ---------------------------------------------------------------------------
def online_adaptive_controversial(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    k: int,
) -> dict[str, set[Project]]:
    """
    Online adaptive-controversial setup (Section 3.1.3): the exposed set E_v of
    every Target Voter, keyed by voter id. The voters are queried one after the
    other (in ``tv_ballots`` order), each in k iterations. In every iteration
    the most controversial not-yet-asked project is recomputed over *everyone
    who has voted on it so far* - the LV voters plus every TV answer already
    collected - and the current voter's answer (simulated from her full ballot)
    is folded into that tally. Each answer thus shifts which projects later
    voters are asked about; this feedback is what makes the setup adaptive and
    distinguishes it from offline revealing-by-controversiality, per the
    paper's "choose project proposals based on all voters preferences
    iteratively" (Section 6).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        tv_ballots : dict[str, set[:py:class:`~pabutools.election.instance.Project`]]
            The full ballot of every Target Voter (her approval set A_v in the
            ideal instance), keyed by voter id - used to answer the adaptive
            questions. Voters are queried in iteration order.
        k : int
            The number of iterations / projects to expose per voter.

    Returns
    -------
        dict[str, set[:py:class:`~pabutools.election.instance.Project`]]
            The exposed set E_v of each Target Voter, keyed by voter id.

    Examples
    --------
    One voter, 4 LV voters, k = 2: p1, p2, p3 are tied as most controversial
    (consensus 0); the first question (tie broken by name) is p1, the second
    p2, so E_v = {p1, p2}.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p3]),
    ...                       ApprovalBallot([p2, p3]), ApprovalBallot([])])
    >>> online_adaptive_controversial(inst, lv, {"v5": {p1, p2}}, k=2)["v5"] == {p1, p2}
    True

    The adaptive feedback: p1 and p2 start tied as most controversial, so the
    first voter is asked p1. Her approval breaks the tie - p1 now leans
    approved and is *less* controversial - so the second voter is asked p2.

    >>> q1, q2, q3 = Project("q1", 1), Project("q2", 1), Project("q3", 1)
    >>> inst2 = Instance([q1, q2, q3], budget_limit=3)
    >>> lv2 = ApprovalProfile([ApprovalBallot([q1]), ApprovalBallot([q2])])
    >>> online_adaptive_controversial(inst2, lv2, {"v3": {q1}, "v4": {q1}}, k=1)
    {'v3': {q1}, 'v4': {q2}}
    """
    # Definition 2.2 over everyone who has voted on p so far: the LV electorate
    # first, then each collected TV answer. consensus(p) = |approvers - disapprovers|.
    scores = lv_profile.approval_scores()
    approvers = {p: scores.get(p, 0) for p in instance}
    disapprovers = {p: lv_profile.num_ballots() - approvers[p] for p in instance}
    exposed_by_voter: dict[str, set[Project]] = {}
    for vid, ballot in tv_ballots.items():
        exposed: set[Project] = set()
        for _ in range(k):
            # gamma_m of the not-yet-asked projects: least consensus, ties by name.
            asked = min(
                set(instance) - exposed,
                key=lambda p: (abs(approvers[p] - disapprovers[p]), str(p)),
            )
            logger.info(
                "online_adaptive_controversial: asks %s -> voter %s %s", asked,
                vid, "approves" if asked in ballot else "disapproves",
            )
            (approvers if asked in ballot else disapprovers)[asked] += 1
            exposed.add(asked)
        exposed_by_voter[vid] = exposed
    return exposed_by_voter


def next_adaptive_question(
    instance: Instance,
    lv_profile: ApprovalProfile,
    answers: CardinalBallot,
    collected: Iterable[CardinalBallot] = (),
) -> Project:
    """
    The single project to put to a voter next, under the online adaptive-
    controversial setup (Section 3.1.3): the most controversial project she has
    not been asked about yet, where controversiality is measured over everyone
    who has voted on it so far.

    :py:func:`online_adaptive_controversial` runs this rule as a closed loop by
    reading each answer off a known ballot, which a live process cannot do - the
    answers are exactly what it does not have yet. This is that loop's single
    step, exposed so a caller can drive it: ask the returned project, add the
    reply to ``answers``, and call again until ``answers`` holds k votes. Feeding
    the replies back is what makes the setup adaptive; it is the only setup of
    the five that cannot be planned in advance with :py:func:`exposed_sets`.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (the universe of projects P).
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        answers : :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`
            The replies this voter has given so far, as a partial ballot: her
            answered projects are the ones excluded from the next question.
            Pass an empty ``partial_ballot()`` for the first question.
        collected : Iterable[:py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`], optional
            The partial ballots already gathered from *other* voters. Their votes
            count towards controversiality but never towards what this voter is
            asked, exactly as in the closed loop.

    Returns
    -------
        :py:class:`~pabutools.election.instance.Project`
            The project to ask about, ties broken lexicographically by name.

    Raises
    ------
        ValueError
            If the voter has already been asked about every project.

    Examples
    --------
    Two LV voters split over q1 and q2, nobody approving q3: q1 and q2 are tied
    at consensus 0 and q3 is settled, so the first question is q1 (tie broken by
    name). Her approval of q1 tips it towards approved, so the second question
    is q2 - the same order the closed loop produces.

    >>> q1, q2, q3 = Project("q1", 1), Project("q2", 1), Project("q3", 1)
    >>> inst = Instance([q1, q2, q3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([q1]), ApprovalBallot([q2])])
    >>> first = next_adaptive_question(inst, lv, partial_ballot())
    >>> first
    q1
    >>> next_adaptive_question(inst, lv, partial_ballot(approved={first}))
    q2

    Once every project has been answered there is nothing left to ask:

    >>> answered = partial_ballot(approved={q1, q2}, disapproved={q3})
    >>> next_adaptive_question(inst, lv, answered)
    Traceback (most recent call last):
        ...
    ValueError: this voter has already been asked about all 3 projects
    """
    remaining = set(instance) - exposed_projects(answers)
    if not remaining:
        raise ValueError(
            f"this voter has already been asked about all {len(instance)} projects"
        )
    # Definition 2.2 over everyone who has voted on p so far: the LV electorate,
    # every answer collected from other voters, and this voter's own replies.
    scores = lv_profile.approval_scores()
    approvers = {p: scores.get(p, 0) for p in instance}
    disapprovers = {p: lv_profile.num_ballots() - approvers[p] for p in instance}
    for ballot in (*collected, answers):
        for project, score in ballot.items():
            if score > 0:
                approvers[project] += 1
            elif score < 0:
                disapprovers[project] += 1
    asked = min(
        remaining,
        key=lambda p: (abs(approvers[p] - disapprovers[p]), str(p)),
    )
    logger.info(
        "next_adaptive_question: asking %s (consensus %d of %d voters so far)",
        asked, abs(approvers[asked] - disapprovers[asked]),
        approvers[asked] + disapprovers[asked],
    )
    return asked




# ---------------------------------------------------------------------------
# Full pipeline (sampling -> prediction -> greedy approval).
# ---------------------------------------------------------------------------
# The paper's design space is a matrix: one setup (Section 3.1) times one
# predictor (Section 2.1). These registries name every choice so a caller can
# pick a cell (``run_pipeline``); the sweep over the whole matrix lives in
# :py:func:`~pabutools_recommendation.analytics.run_all_experiments`.
SETUPS = (
    "random",
    "offline_popularity",
    "offline_consensus",
    "offline_controversiality",
    "online_adaptive_controversial",
)

PREDICTORS = {
    "classification": predict_by_classification,
    "matrix_factorization": predict_by_matrix_factorization,
    "factorization_machines": predict_by_factorization_machines,
}


def exposed_sets(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    setup: str,
    k: int,
    seed: int | None = None,
) -> dict[str, set[Project]]:
    """
    The exposed set E_v of every Target Voter under one of the five Section 3.1
    setups, keyed by voter id. Helper that hides the setups' differing
    signatures behind one interface: the three offline samplers expose the
    *same* k projects to everyone, ``random`` draws independently per voter,
    and ``online_adaptive_controversial`` queries the voters sequentially,
    folding each answer into the controversiality tally that picks the later
    voters' questions.

    Examples
    --------
    Offline popularity exposes the single most popular LV project (p1) to every
    Target Voter.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> exposed_sets(inst, lv, {"v3": {p1}}, "offline_popularity", k=1)
    {'v3': {p1}}

    Invalid inputs raise a ValueError naming the problem:

    >>> exposed_sets(inst, lv, {"v3": {p1}}, "by_magic", k=1)
    Traceback (most recent call last):
        ...
    ValueError: unknown setup 'by_magic'; expected one of: random, offline_popularity, offline_consensus, offline_controversiality, online_adaptive_controversial
    >>> exposed_sets(inst, lv, {"v3": {p1}}, "random", k=99)
    Traceback (most recent call last):
        ...
    ValueError: k=99 must be between 0 and the number of projects (3)
    """
    if setup not in SETUPS:
        raise ValueError(
            f"unknown setup {setup!r}; expected one of: " + ", ".join(SETUPS)
        )
    if not 0 <= k <= len(instance):
        raise ValueError(
            f"k={k} must be between 0 and the number of projects ({len(instance)})"
        )
    logger.debug("exposed_sets: validated setup=%s, k=%d", setup, k)
    if setup == "random":
        # "possibly different E for each voter" (Section 2.4): one independent
        # uniform draw per voter, derived from the single seed.
        rng = random.Random(seed)
        return {vid: random_setup(instance, k, rng.randrange(2**32)) for vid in tv_ballots}
    if setup == "online_adaptive_controversial":
        return online_adaptive_controversial(instance, lv_profile, tv_ballots, k)
    offline = {
        "offline_popularity": offline_popularity,
        "offline_consensus": offline_consensus,
        "offline_controversiality": offline_controversiality,
    }[setup]
    shared = offline(instance, lv_profile, k)  # same set for every TV voter
    return {vid: shared for vid in tv_ballots}


def plan_sampling(
    n_voters: int,
    n_projects: int,
    sample_degree: float,
    lv_degree: float,
) -> tuple[int, int]:
    """
    Size a process from the two partiality knobs of Section 3.0.1, *before* any
    ballot exists: how many voters must give a full ballot, and how many projects
    each remaining voter is asked about.

    This is the arithmetic of Example 3.1 on its own. With n voters and m
    projects, ``sample_degree`` (s) fixes the total number of collected votes at
    s*n*m and ``lv_degree`` (l) is the share of those votes that come from full
    ballots, so ``|LV|`` = round(s*l*n) and the remaining s*(1-l)*n*m votes are
    divided equally among the s*n*m Target Voters. ``lv_degree == 1`` is the
    paper's naive sampling baseline: the sampled voters answer in full and
    nobody else is asked anything, so k = 0.

    Use it to plan a real election - "we have 5000 voters, 100 projects, and we
    are willing to ask for 30% of the votes; how many questions each?" -
    and :py:func:`split_lv_tv` to carve a simulated one out of a known profile.

    Parameters
    ----------
        n_voters : int
            The number n of voters in the electorate.
        n_projects : int
            The number m of projects on the ballot.
        sample_degree : float
            Fraction of all n*m votes that are collected, in [0, 1].
        lv_degree : float
            Fraction of the collected votes coming from full ballots, in [0, 1].

    Returns
    -------
        tuple[int, int]
            ``(number of Learning Voters, questions per Target Voter)``.

    Examples
    --------
    Four voters and two projects. Collecting everything with ``lv_degree == 1``
    makes all four Learning Voters and leaves nobody to question; collecting
    half the votes with half of them from full ballots gives one Learning Voter
    and asks the other three about 0.5*0.5*4*2/3 ~ 1 project each.

    >>> plan_sampling(4, 2, sample_degree=1.0, lv_degree=1.0)
    (4, 0)
    >>> plan_sampling(4, 2, sample_degree=0.5, lv_degree=0.5)
    (1, 1)

    A realistic district: 30% of the votes collected, a tenth of them as full
    ballots, so 150 people fill in all 100 projects and the other 4850 are asked
    about 28 each - instead of every one of the 5000 facing the whole ballot.

    >>> plan_sampling(5000, 100, sample_degree=0.3, lv_degree=0.1)
    (150, 28)

    >>> plan_sampling(4, 2, sample_degree=1.5, lv_degree=0.5)
    Traceback (most recent call last):
        ...
    ValueError: sample_degree=1.5 must be in [0, 1]
    """
    for name, degree in (("sample_degree", sample_degree), ("lv_degree", lv_degree)):
        if not 0 <= degree <= 1:
            raise ValueError(f"{name}={degree} must be in [0, 1]")
    n_lv = round(sample_degree * lv_degree * n_voters)
    n_tv = n_voters - n_lv
    if lv_degree == 1 or not n_tv:
        logger.info(
            "plan_sampling: sample=%.2f lv=%.2f -> all %d collected votes come "
            "from %d full ballots, nobody else is asked anything (k=0)",
            sample_degree, lv_degree, n_lv * n_projects, n_lv,
        )
        return n_lv, 0
    votes_left = sample_degree * (1 - lv_degree) * n_voters * n_projects
    k = round(votes_left / n_tv)
    logger.info(
        "plan_sampling: sample=%.2f lv=%.2f of %d voters x %d projects -> "
        "%d full ballots, the other %d asked k=%d each "
        "(%.0f of %d votes collected in total)",
        sample_degree, lv_degree, n_voters, n_projects, n_lv, n_tv, k,
        n_lv * n_projects + n_tv * k, n_voters * n_projects,
    )
    return n_lv, k


def split_lv_tv(
    instance: Instance,
    profile: ApprovalProfile,
    sample_degree: float,
    lv_degree: float,
    seed: int | None = None,
) -> tuple[ApprovalProfile, dict[str, set[Project]], int]:
    """
    Step 1 of the pipeline (Sections 2.4 / 3.0.1): partition the n voters of
    the *ideal* instance into Learning Voters (LV, full ballots) and Target
    Voters (TV, queried and completed), and derive k, the number of projects
    each TV voter is asked about.

    The two knobs are *vote* budgets (Example 3.1 of the paper). With n voters
    and m projects, ``sample_degree`` (s) fixes the total number of collected
    votes at s*n*m and ``lv_degree`` (l) is the share of those votes that come
    from full ballots:

    * ``|LV|`` = round(s*l*n) voters, drawn at random, keep their full ballots;
    * every *other* voter is a TV, and the remaining s*(1-l)*n*m votes are
      divided equally among them: k = round(s*(1-l)*n*m / ``|TV|``).

    ``lv_degree == 1`` is the paper's naive "sampling" baseline (Section 6):
    the sampled voters answer in full and the rest of the community does not
    vote at all - no TV voters, k = 0.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (m projects).
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The ideal instance's full ballots (all n voters).
        sample_degree : float
            Fraction of all n*m votes that are collected, in [0, 1].
        lv_degree : float
            Fraction of the collected votes coming from full ballots, in [0, 1].
        seed : int, optional
            Seed for the random partition, for reproducibility.

    Returns
    -------
        tuple[:py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`, dict[str, set[:py:class:`~pabutools.election.instance.Project`]], int]
            The LV profile (full ballots), the TV voters' full ballots keyed by
            voter id (their known ground truth, used to simulate the k answers),
            and k.

    Examples
    --------
    Four voters, two projects. Collecting everything with ``lv_degree == 1``
    makes everyone an LV and leaves no TV. Collecting half the votes with half
    of them from full ballots gives one LV voter; the other three are all TV,
    each asked k = 0.5*0.5*4*2/3 ~ 1 project.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> inst = Instance([p1, p2], budget_limit=2)
    >>> prof = ApprovalProfile([ApprovalBallot([p1]), ApprovalBallot([p2]),
    ...                         ApprovalBallot([p1, p2]), ApprovalBallot([])])
    >>> lv, tv, k = split_lv_tv(inst, prof, sample_degree=1.0, lv_degree=1.0)
    >>> lv.num_ballots(), len(tv), k
    (4, 0, 0)
    >>> lv, tv, k = split_lv_tv(inst, prof, sample_degree=0.5, lv_degree=0.5, seed=0)
    >>> lv.num_ballots(), len(tv), k
    (1, 3, 1)
    >>> split_lv_tv(inst, prof, sample_degree=1.5, lv_degree=0.5)
    Traceback (most recent call last):
        ...
    ValueError: sample_degree=1.5 must be in [0, 1]
    """
    voters = list(profile)
    n, m = len(voters), len(instance)
    n_lv, k = plan_sampling(n, m, sample_degree, lv_degree)
    order = random.Random(seed).sample(range(n), n)  # a random permutation
    lv_profile = ApprovalProfile([voters[i] for i in order[:n_lv]])
    # lv_degree == 1 is the naive sampling baseline: nobody else votes at all.
    tv_ballots = (
        {} if lv_degree == 1
        else {f"v{i}": set(voters[i]) for i in order[n_lv:]}
    )
    logger.info(
        "split_lv_tv: sample=%.2f lv=%.2f -> %d LV, %d TV asked k=%d each (of %d voters)",
        sample_degree, lv_degree, lv_profile.num_ballots(), len(tv_ballots), k, n,
    )
    return lv_profile, tv_ballots, k


def complete_ballots(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    k: int,
    *,
    setup: str,
    predict,
    seed: int | None = None,
) -> tuple[dict[str, ApprovalBallot], dict[str, set[Project]]]:
    """
    Steps 2 and 3 of the pipeline: choose every Target Voter's exposed set under
    ``setup``, then complete her ballot with the ``predict`` module.

    Returns the completed ballots *and* the exposed sets, both keyed by voter
    id. The exposed sets come back because a caller needs them to tell which
    votes were predicted rather than answered - Section 5.1 scores only the
    hidden ones. Shared by :py:func:`run_pipeline`, which goes on to apply the
    voting rule, and
    :py:func:`~pabutools_recommendation.analytics.run_all_experiments`,
    which also scores the ballots themselves.

    Examples
    --------
    Three LV voters approve {p1, p2}; the single TV voter is shown p1 only, and
    her hidden vote on p2 is filled in from theirs.

    >>> p1, p2, p3 = Project("p1", 3), Project("p2", 3), Project("p3", 4)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> done, shown = complete_ballots(inst, lv, {"v": {p1, p2}}, k=1,
    ...                                setup="offline_popularity",
    ...                                predict=predict_by_classification)
    >>> shown["v"] == {p1}, done["v"] == {p1, p2}
    (True, True)
    """
    exposed = exposed_sets(instance, lv_profile, tv_ballots, setup, k, seed)
    partial = {
        vid: reveal_ballot(instance, tv_ballots[vid], exposed[vid])
        for vid in tv_ballots
    }
    hidden = sum(len(instance) - len(exposed[vid]) for vid in tv_ballots)
    logger.info(
        "complete_ballots: setup=%s exposed %d of %d votes across %d TV "
        "voters, asking %s to predict the remaining %d",
        setup, sum(len(e) for e in exposed.values()),
        len(tv_ballots) * len(instance), len(tv_ballots),
        getattr(predict, "__name__", predict), hidden,
    )
    return predict(instance, lv_profile, partial), exposed


def run_pipeline(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    k: int,
    *,
    setup: str,
    predict,
    seed: int | None = None,
) -> BudgetAllocation:
    """
    The complete Section 3 pipeline for one (setup, predictor) choice: expose k
    projects to each TV voter with ``setup``, complete the hidden votes with the
    ``predict`` module, then run greedy approval on the LV plus completed TV
    ballots. ``setup`` and ``predict`` are required - the pipeline runs exactly
    one cell of the design matrix, so the caller must name both explicitly (the
    sweep over every cell is
    :py:func:`~pabutools_recommendation.analytics.run_all_experiments`).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters.
        tv_ballots : dict[str, set[:py:class:`~pabutools.election.instance.Project`]]
            The full ballot of every Target Voter (TV) - her approval set A_v in
            the ideal instance - keyed by voter id.
        k : int
            The number of projects to expose per TV voter.
        setup : str
            The name of the Section 3.1 setup to use (one of :py:data:`SETUPS`).
        predict : callable
            The Section 2.1 prediction module (one of :py:data:`PREDICTORS`'
            values): it takes the TV voters' partial ballots, keyed by voter
            id, and returns their completed approval ballots.
        seed : int, optional
            Seed for the ``random`` setup, ignored by the others.

    Returns
    -------
        :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            The predicted winning bundle.

    Examples
    --------
    The "perfect" case: three LV voters and one TV voter all support the two
    cheap projects {p1, p2}; the predicted bundle coincides with the real one.

    >>> p1, p2, p3, p4 = (Project("p1", 3), Project("p2", 3),
    ...                   Project("p3", 4), Project("p4", 4))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> sorted(run_pipeline(inst, lv, {"v4": {p1, p2}}, k=1,
    ...                     setup="offline_popularity",
    ...                     predict=predict_by_matrix_factorization), key=str)
    [p1, p2]
    """
    logger.info(
        "run_pipeline: starting with setup=%s predict=%s k=%d on %d LV + %d TV ballots",
        setup, predict.__name__, k, lv_profile.num_ballots(), len(tv_ballots),
    )
    # Step 1 (the LV/TV split) is done by the caller / :py:func:`split_lv_tv`.
    # Steps 2-3 - sampling + prediction.
    completed, _ = complete_ballots(
        instance, lv_profile, tv_ballots, k,
        setup=setup, predict=predict, seed=seed,
    )
    # Step 4 - combine with the known LV ballots.
    combined = ApprovalProfile(list(lv_profile) + [completed[vid] for vid in tv_ballots])
    logger.info(
        "run_pipeline: all %d TV ballots completed, running the voting rule",
        len(tv_ballots),
    )
    # Steps 5-6 - voting rule: greedy approval on the LV + completed TV ballots.
    return greedy_approval(instance, combined)


def run_experiment(
    instance: Instance,
    profile: ApprovalProfile,
    sample_degree: float,
    lv_degree: float,
    *,
    setup: str,
    predictor="classification",
    seed: int | None = None,
) -> BudgetAllocation:
    """
    One experiment end to end, straight from the two partiality knobs: split the
    ideal ``profile`` into LV and TV with :py:func:`split_lv_tv` (which derives k
    from the knobs) and run :py:func:`run_pipeline` on the result. A convenience
    composition of the two, for when you want a single cell of the design matrix
    without holding the split yourself.

    Like :py:func:`run_pipeline` this is a *simulation*: it needs the ideal
    profile, because the Target Voters' answers are read off their true ballots.
    To run a process whose answers you actually collected, use :py:func:`elect`.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The ideal instance's full ballots (all n voters).
        sample_degree, lv_degree : float
            The partiality knobs of Section 3.0.1, both in [0, 1].
        setup : str
            The Section 3.1 setup to sample with (one of :py:data:`SETUPS`).
        predictor : str or callable, optional
            A name from :py:data:`PREDICTORS` or a prediction module itself.
            Defaults to ``"classification"``, the paper's best performer.
        seed : int, optional
            Seed for the split and the sampling, for reproducibility.

    Returns
    -------
        :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            The predicted winning bundle.

    Examples
    --------
    Four voters who all approve the two cheap projects; collecting half the votes
    still reproduces the real bundle.

    >>> p1, p2, p3 = Project("p1", 3), Project("p2", 3), Project("p3", 4)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2])] * 4)
    >>> sorted(run_experiment(inst, prof, 0.5, 0.5, setup="offline_popularity",
    ...                       seed=0), key=str)
    [p1, p2]
    """
    logger.info(
        "run_experiment: simulating sample_degree=%.2f lv_degree=%.2f with "
        "setup=%s and the %s predictor",
        sample_degree, lv_degree, setup, predictor,
    )
    lv_profile, tv_ballots, k = split_lv_tv(
        instance, profile, sample_degree, lv_degree, seed=seed
    )
    return run_pipeline(
        instance, lv_profile, tv_ballots, k,
        setup=setup, predict=as_predictor(predictor), seed=seed,
    )


# ---------------------------------------------------------------------------
# Running a real process, as opposed to reproducing the paper's experiments.
# ---------------------------------------------------------------------------
#: A Section 2.1 prediction module: it completes the Target Voters' partial
#: ballots, keyed by voter id. Named in :py:data:`PREDICTORS`.
Predictor = Callable[
    [Instance, ApprovalProfile, dict[str, CardinalBallot]],
    dict[str, ApprovalBallot],
]


def as_predictor(predictor: str | Predictor) -> Predictor:
    """
    The prediction module named by ``predictor``, which may be a key of
    :py:data:`PREDICTORS` or the function itself, so callers can say
    ``"classification"`` instead of importing it.

    Examples
    --------
    >>> as_predictor("classification") is predict_by_classification
    True
    >>> as_predictor(predict_by_classification) is predict_by_classification
    True
    >>> as_predictor("by_magic")
    Traceback (most recent call last):
        ...
    ValueError: unknown predictor 'by_magic'; expected one of: classification, matrix_factorization, factorization_machines
    """
    if callable(predictor):
        return predictor
    if predictor not in PREDICTORS:
        raise ValueError(
            f"unknown predictor {predictor!r}; expected one of: "
            + ", ".join(PREDICTORS)
        )
    return PREDICTORS[predictor]


def elect(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, CardinalBallot],
    predictor: str | Predictor = "classification",
) -> BudgetAllocation:
    """
    Decide a real PB process from the ballots that were actually collected.

    This is the deployment counterpart of :py:func:`run_pipeline`. The pipeline
    is a simulation: it takes the Target Voters' *true* ballots and reads their
    answers off them, which you cannot do in a live process because those
    answers are exactly what you do not have. ``elect`` instead takes the
    answers themselves - each Target Voter's partial ballot, saying which of the
    projects she was asked about she approved and which she rejected - completes
    the gaps with a Section 2.1 prediction module, and applies greedy approval
    to the LV ballots plus the completed TV ballots.

    A live process therefore runs in three steps: :py:func:`plan_sampling` to
    size it, :py:func:`exposed_sets` to choose which projects to put in front of
    each voter, and ``elect`` once the answers are in.

    .. note::
        Of the five setups, ``random`` and the three ``offline_*`` ones need
        nothing but the voter ids to choose their questions, so they work in a
        live process as they are. ``online_adaptive_controversial`` picks each
        question from the answers gathered so far, so deploying it needs an
        interactive loop rather than a single call.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots that were submitted in full (the Learning Voters).
            May be empty if every voter answered only part of the ballot.
        tv_ballots : dict[str, :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`]
            The partial ballots that came back, keyed by voter id: build each one
            with :py:func:`partial_ballot` from the projects that voter approved
            and rejected. Everything else is treated as unanswered and predicted.
        predictor : str or callable, optional
            A name from :py:data:`PREDICTORS` or a prediction module itself.
            Defaults to ``"classification"``, the paper's best performer.

    Returns
    -------
        :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            The winning bundle, respecting the instance's budget limit.

    Examples
    --------
    Three voters filled in the whole ballot and approved {p1, p2}. A fourth was
    asked about only p1 and p3: she approved p1 and rejected p3, and was never
    asked about p2. Her opinion on p2 is predicted from the others, and the
    budget of 6 funds the two projects costing 3.

    >>> p1, p2, p3 = Project("p1", 3), Project("p2", 3), Project("p3", 4)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> answers = {"v4": partial_ballot(approved={p1}, disapproved={p3})}
    >>> sorted(elect(inst, lv, answers), key=str)
    [p1, p2]

    An election where nobody filled in a full ballot still works, as long as the
    answers between them cover the projects.

    >>> answers = {
    ...     "a": partial_ballot(approved={p1}, disapproved={p3}),
    ...     "b": partial_ballot(approved={p2}, disapproved={p3}),
    ... }
    >>> sorted(elect(inst, ApprovalProfile([]), answers), key=str)
    [p1, p2]
    """
    completed = as_predictor(predictor)(instance, lv_profile, tv_ballots)
    combined = ApprovalProfile(
        list(lv_profile) + [completed[vid] for vid in tv_ballots]
    )
    logger.info(
        "elect: %d full ballots + %d partial ballots completed, running the rule",
        lv_profile.num_ballots(), len(tv_ballots),
    )
    return greedy_approval(instance, combined)
