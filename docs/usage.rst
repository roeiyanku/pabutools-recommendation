Recommendation
==============

For reference, see the module :py:mod:`~pabutools_recommendation`.

When a participatory budgeting instance has many projects, asking every voter about every
project is a burden, and ballots are often left unfinished. This module implements the
approach of Leibiker and Talmon (2023): ask each voter about only a few projects, estimate
the votes that were never asked for, and run a voting rule on the completed profile.

There are two ways to use it. If you are running a process and only hold partial ballots,
:py:func:`~pabutools_recommendation.recommendation.elect` returns an outcome. If you hold
complete ballots and want to know what asking less would have cost you,
:py:func:`~pabutools_recommendation.recommendation.run_experiment` simulates the loss.

Partial Ballots
---------------

A voter who was asked about only some of the projects has three kinds of opinion: projects
she approved, projects she rejected, and projects she was never shown. This is stored in a
:py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot` under the convention
*positive means approved, negative means disapproved, zero or absent means not asked*.
:py:func:`~pabutools_recommendation.recommendation.partial_ballot` builds one:

.. code-block:: python

    from pabutools.election import Instance, Project, ApprovalProfile, ApprovalBallot
    from pabutools_recommendation import partial_ballot

    garden = Project("Garden", 18)
    crossings = Project("Crossings", 24)
    library = Project("Library", 16)
    shade = Project("Shade", 12)
    instance = Instance([garden, crossings, library, shade], budget_limit=40)

    # A voter who approved the garden, rejected the crossings, and was asked nothing else.
    ballot = partial_ballot(approved={garden}, disapproved={crossings})

Running a Process
-----------------

:py:func:`~pabutools_recommendation.recommendation.elect`

Some voters do provide a complete ballot; they are the *learning voters*, and their profile
is what the estimation learns from. Every other voter is asked about a limited number of
projects.

Which projects to ask about is decided by a *sampling setup*. The offline setups choose one
set for everybody, out of the complete ballots already collected:

.. code-block:: python

    from pabutools_recommendation import offline_controversiality

    lv_profile = ApprovalProfile([
        ApprovalBallot([garden, library]),
        ApprovalBallot([crossings, shade]),
        ApprovalBallot([garden, shade]),
        ApprovalBallot([garden, library, shade]),
    ])

    offline_controversiality(instance, lv_profile, k=2)
    # {Library, Shade}  - the two projects the learning voters most disagree about

The alternatives are :py:func:`~pabutools_recommendation.recommendation.random_setup`,
:py:func:`~pabutools_recommendation.recommendation.offline_popularity` and
:py:func:`~pabutools_recommendation.recommendation.offline_consensus`.

Collect an answer for those projects from each remaining voter, and elect:

.. code-block:: python

    from pabutools_recommendation import elect

    answers = {
        "v5": partial_ballot(approved={garden}, disapproved={crossings}),
        "v6": partial_ballot(approved={crossings}, disapproved={garden}),
    }

    elect(instance, lv_profile, answers, predictor="classification")
    # BudgetAllocation([Garden, Shade])

:py:func:`~pabutools_recommendation.recommendation.elect` returns a
:py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`, like any other rule. An
answer a voter actually gave is never overwritten: the estimation only fills in the projects
she was not asked about.

To decide how many questions to ask in the first place,
:py:func:`~pabutools_recommendation.recommendation.plan_sampling` turns a vote budget into a
plan. Collecting 30% of the votes from 5000 voters over 100 projects, a tenth of them as
complete ballots, means 150 people fill in the whole ballot and the rest answer 28 questions
each:

.. code-block:: python

    from pabutools_recommendation import plan_sampling

    plan_sampling(5000, 100, sample_degree=0.3, lv_degree=0.1)
    # (150, 28)

Asking Adaptively
-----------------

:py:func:`~pabutools_recommendation.recommendation.next_adaptive_question`

The offline setups fix their questions in advance. The online setup cannot: it re-reads the
tally after every answer, so each reply changes what is asked next. A live process drives it
one question at a time:

.. code-block:: python

    from pabutools_recommendation import next_adaptive_question, partial_ballot

    answers = partial_ballot()
    for _ in range(2):
        project = next_adaptive_question(instance, lv_profile, answers)
        # ... put `project` to the voter, then fold her reply into `answers` ...

Estimating the Missing Votes
----------------------------

Three prediction modules are provided:
:py:func:`~pabutools_recommendation.model_training.predict_by_classification` (one binary
classifier per project),
:py:func:`~pabutools_recommendation.model_training.predict_by_matrix_factorization` and
:py:func:`~pabutools_recommendation.model_training.predict_by_factorization_machines`
(collaborative filtering).

They are selected by name, as in the ``predictor="classification"`` argument above. Any
callable with the same signature works, so you can supply your own:

.. code-block:: python

    def approve_everything(instance, lv_profile, partial_ballots):
        return {voter: ApprovalBallot(instance) for voter in partial_ballots}

    elect(instance, lv_profile, answers, predictor=approve_everything)

These three modules rely on ``xgboost``, ``scikit-surprise`` and ``lightfm``, which are
**not** installed by default. They come with the optional ``ml`` extra::

    pip install pabutools-recommendation[ml]

The libraries are imported only when a model is actually fitted, so everything else in this
module works without them.

Measuring the Loss
------------------

:py:func:`~pabutools_recommendation.recommendation.run_experiment`

If you already hold complete ballots, you can measure what asking less would have cost. This
is a simulation: the profile is split into the voters who answer in full and the ones who do
not, each of the latter has her answers to k projects read off her real ballot, and the rest
are withheld for the prediction module to estimate. The allocation that comes back can then
be compared with the one complete information produces:

.. code-block:: python

    from pabutools_recommendation import run_experiment, greedy_approval
    from pabutools_recommendation import fractional_allocation_score

    profile = ApprovalProfile([
        ApprovalBallot([garden, library]), ApprovalBallot([crossings, shade]),
        ApprovalBallot([garden, shade]), ApprovalBallot([garden, library, shade]),
        ApprovalBallot([crossings, shade]), ApprovalBallot([garden, library]),
    ])

    real = set(greedy_approval(instance, profile))
    predicted = set(run_experiment(
        instance, profile, 0.5, 0.5, setup="offline_controversiality", seed=17
    ))

    fractional_allocation_score(real, predicted, instance.budget_limit)  # 0.75
    len(real ^ predicted)                                                # 0

The two arguments ``0.5, 0.5`` are the *sample degree* — the share of all voter-project
votes that is collected — and the *LV degree*, the share of those votes coming from complete
ballots. :py:func:`~pabutools_recommendation.analytics.fractional_allocation_score`
reports the cost of the correctly predicted projects as a share of the budget, and the
symmetric difference counts the projects on which the two allocations disagree.

:py:func:`~pabutools_recommendation.analytics.run_all_experiments` repeats this over
a grid of both degrees, for every setup and every prediction module at once.

To score the predicted votes themselves rather than the allocation, use
:py:func:`~pabutools_recommendation.analytics.classification_metrics`, which reports
precision, recall and F1 over the projects a voter was never asked about.

.. note::
    The evaluation tools live in :py:mod:`~pabutools_recommendation.analytics`, separate
    from :py:mod:`~pabutools_recommendation.recommendation`: one module runs the system,
    the other measures how well it did. They complement Pabutools' own
    :py:mod:`pabutools.analysis`, which scores a completed election rather than the
    prediction that produced it.
