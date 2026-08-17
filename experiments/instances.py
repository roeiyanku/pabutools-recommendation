"""
Random input generation for the recommendation-system experiments.

The assignment asks for *random inputs of increasing size*. pabutools already
ships :py:func:`~pabutools.election.instance.get_random_instance` and
:py:func:`~pabutools.election.profile.approvalprofile.get_random_approval_profile`,
but a profile of independent uniform approvals is the wrong input for this
particular algorithm: every prediction module here (classification, matrix
factorization, factorization machines) works by finding *correlations between
voters*. If preferences are independent coin flips there is nothing to learn,
every predictor degrades to guessing the base rate, and the comparison graph
comes out flat - which would say nothing about the algorithms.

So the default generator draws voters from a small number of latent groups
("clustered"), which is the standard synthetic model for recommender input and
matches the paper's assumption that voters have shared, learnable taste. The
uniform generator is kept as ``"uniform"`` for a null-model sanity check: the
predictors *should* look indistinguishable there, and confirming that is a
useful control.

Programmer: Roei Yanku
"""

from __future__ import annotations

import random

from pabutools.election import (
    ApprovalBallot,
    ApprovalProfile,
    Instance,
    Project,
    total_cost,
)

#: Fraction of the total project cost available as budget. Fixed here on
#: purpose: ``get_random_instance`` samples the budget uniformly between the
#: cheapest project and the total cost, which swings from "nothing fits" to
#: "everything fits" and would drown the differences between setups in noise.
BUDGET_FRACTION = 0.3

#: Number of latent taste groups in the clustered model.
NUM_GROUPS = 4

#: Probability that a voter agrees with their group's opinion on a project.
#: Below 1.0 so voters are correlated but not identical.
AGREEMENT = 0.85


def random_instance(
    num_projects: int,
    *,
    min_cost: int = 100,
    max_cost: int = 1000,
    budget_fraction: float = BUDGET_FRACTION,
    rng: random.Random | None = None,
) -> Instance:
    """
    A random PB instance with ``num_projects`` projects, costs drawn uniformly
    from ``[min_cost, max_cost]`` and a budget limit of ``budget_fraction`` of
    the total cost.
    """
    rng = rng or random.Random()
    instance = Instance(
        Project(name=str(p), cost=rng.randint(min_cost, max_cost))
        for p in range(num_projects)
    )
    instance.budget_limit = round(budget_fraction * total_cost(instance))
    return instance


def clustered_profile(
    instance: Instance,
    num_voters: int,
    *,
    num_groups: int = NUM_GROUPS,
    agreement: float = AGREEMENT,
    rng: random.Random | None = None,
) -> ApprovalProfile:
    """
    Approval profile drawn from ``num_groups`` latent taste groups. Each group
    has a hidden approval vector over the projects; each voter belongs to one
    group and copies its opinion on each project with probability
    ``agreement``, flipping it otherwise. Voters within a group are therefore
    similar but not identical - exactly the structure the prediction modules
    are meant to exploit.
    """
    rng = rng or random.Random()
    projects = list(instance)
    group_opinions = [
        {p: rng.random() < 0.5 for p in projects} for _ in range(num_groups)
    ]
    ballots = []
    for _ in range(num_voters):
        opinion = group_opinions[rng.randrange(num_groups)]
        ballots.append(
            ApprovalBallot(
                p
                for p in projects
                if (opinion[p] if rng.random() < agreement else not opinion[p])
            )
        )
    return ApprovalProfile(ballots)


def uniform_profile(
    instance: Instance,
    num_voters: int,
    *,
    approval_probability: float = 0.5,
    rng: random.Random | None = None,
) -> ApprovalProfile:
    """
    Null model: every voter approves every project independently with
    probability ``approval_probability``. No structure, hence nothing for a
    predictor to learn - used as a control, not as the main input.
    """
    rng = rng or random.Random()
    projects = list(instance)
    return ApprovalProfile(
        ApprovalBallot(p for p in projects if rng.random() < approval_probability)
        for _ in range(num_voters)
    )


def random_setting(
    num_voters: int,
    num_projects: int,
    *,
    model: str = "clustered",
    seed: int | None = None,
) -> tuple[Instance, ApprovalProfile]:
    """
    One random (instance, ideal profile) pair. ``model`` is ``"clustered"``
    (default) or ``"uniform"``; ``seed`` makes the draw reproducible, which the
    experiment harness relies on to re-run a row identically.
    """
    rng = random.Random(seed)
    instance = random_instance(num_projects, rng=rng)
    if model == "clustered":
        profile = clustered_profile(instance, num_voters, rng=rng)
    elif model == "uniform":
        profile = uniform_profile(instance, num_voters, rng=rng)
    else:
        raise ValueError(f"unknown model {model!r}; expected 'clustered' or 'uniform'")
    return instance, profile
