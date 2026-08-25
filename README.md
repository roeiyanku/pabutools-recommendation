# pabutools-recommendation

Vote prediction for participatory budgeting — an extension of
[pabutools](https://github.com/comsoc-community/pabutools).

**Paper:** *"A Recommendation System for Participatory Budgeting"*
([PDF](https://optlearnmas23.github.io/files/p17.pdf))
**Authors:** Gil Leibiker and Nimrod Talmon (2023)
**Programmer:** Roei Yanku

## What it adds

Pabutools takes a **complete** profile — every voter has an opinion on every
project — and elects a bundle with MES, Phragmén, greedy welfare, and so on.
When an instance has hundreds of projects, that assumption breaks: asking every
voter about every project is a burden, and real ballots come back unfinished.

This package adds the option that pabutools does not offer: ask each voter about
only *part* of the projects, then **predict** the rest of their ballot before the
rule runs.

```
  partial ballots  ──►  sampling setup  ──►  prediction module  ──►  pabutools rule
  (some answers)        (which projects       (fill in the rest)      (elect a bundle)
                         to ask about)
```

It plugs into pabutools rather than replacing any of it: the inputs and outputs
are pabutools' own `Instance`, `Project`, `ApprovalProfile` and
`BudgetAllocation`, and the election itself is run by `pabutools.rules`.

- **Sampling setups** (paper §3.1) — `random_setup`, `offline_popularity`,
  `offline_consensus`, `offline_controversiality`, and the adaptive
  `online_adaptive_controversial`.
- **Prediction modules** (§2.1) — `predict_by_classification` (one binary
  classifier per project), `predict_by_matrix_factorization` and
  `predict_by_factorization_machines` (collaborative filtering). Any callable
  with the same signature works, so you can supply your own.
- **Evaluation** (§5) — `classification_metrics` (precision / recall / F1 on the
  votes never asked for) and `fractional_allocation_score` (how much of the
  budget the predicted bundle got right).

## Install

The package is not on PyPI yet, so install it from this repository. Pip clones
it, builds it, and pulls in `pabutools` from PyPI as a dependency:

```bash
pip install "pabutools-recommendation @ git+https://github.com/roeiyanku/pabutools-recommendation"
```

That exact line also works as a row in a `requirements.txt`.

The three prediction modules need `xgboost`, `scikit-surprise` and `lightfm`,
which are **not** installed by default. They come with the `ml` extra:

```bash
pip install "pabutools-recommendation[ml]"
```

Those libraries are imported only when a model is actually fitted, so the
sampling setups, the voting rule and the pipeline all work without them.

## Quickstart

A worked example, one step at a time. The snippets run as written, in order,
and every output shown is the real one.

The setting: a town is choosing between four projects. Together they cost 70,
but only 40 can be spent. Normally you would ask every resident about all four —
but with hundreds of projects that becomes a questionnaire nobody finishes. So
we ask most people about **two** projects only, and predict the rest.

### 1. Describe the election

```python
from pabutools.election import Instance, Project, ApprovalProfile, ApprovalBallot

# Project(name, cost)
garden    = Project("Garden", 18)
crossings = Project("Crossings", 24)
library   = Project("Library", 16)
shade     = Project("Shade", 12)

# An Instance bundles the projects with the money available to fund them.
instance = Instance([garden, crossings, library, shade], budget_limit=40)
```

`Instance` and `Project` are pabutools' own classes. This package does not
replace them — it consumes them, and hands the result back to a pabutools rule
at the end.

### 2. Collect full ballots from a few voters

Prediction needs something to learn from, so a minority of voters — the paper
calls them **learning voters** — are asked about everything. Here, four of them:

```python
lv_profile = ApprovalProfile([
    ApprovalBallot([garden, library]),
    ApprovalBallot([garden, shade]),
    ApprovalBallot([garden, library, shade]),
    ApprovalBallot([garden, crossings]),
])
```

An `ApprovalBallot` lists the projects a voter approved; anything absent is a
project they saw and rejected. These four ballots are complete — every voter had
an opinion on all four projects. Tallied up:

| project | approvals | |
|---|---|---|
| Garden | 4 / 4 | everyone wants it |
| Crossings | 1 / 4 | almost nobody does |
| Library | 2 / 4 | evenly split |
| Shade | 2 / 4 | evenly split |

### 3. Decide which two projects to ask everyone else about

This is the interesting choice, and a **sampling setup** makes it. Asking about
Garden is close to wasted breath — all four learning voters approved it, so a
fifth voter's answer is largely predictable. The projects worth spending a
question on are the ones opinion actually divides:

```python
from pabutools_recommendation import offline_controversiality

offline_controversiality(instance, lv_profile, k=2)
# {Library, Shade}
```

Library and Shade each split the learning voters 2–2, which makes them the least
predictable and so the most informative to ask about. The alternatives are
`random_setup`, `offline_popularity` (ask about the most-approved),
`offline_consensus` (the least divisive), and `online_adaptive_controversial`,
which re-picks after every single answer instead of fixing the questions up
front.

### 4. Collect the partial answers

Everyone else — the **target voters** — is asked only about those two projects.
Their reply has three possible states per project, and `partial_ballot` records
it:

```python
from pabutools_recommendation import partial_ballot

answers = {
    "v5": partial_ballot(approved={library}, disapproved={shade}),
    "v6": partial_ballot(approved={shade},   disapproved={library}),
}
# Neither voter was asked about Garden or Crossings - those stay unknown.
```

That third state is the whole point. A project a voter *rejected* and a project
they were *never shown* look identical in an ordinary approval ballot — both are
simply absent — but they mean opposite things to a prediction model. Under the
hood this is a pabutools `CardinalBallot` scoring +1 / −1 / 0, so no new ballot
type had to be invented to carry it.

### 5. Predict the missing votes, then elect

```python
from pabutools_recommendation import elect

elect(instance, lv_profile, answers, predictor="classification")
# [Garden, Library]
```

`elect` does three things in order:

1. fits a model on the learning voters' complete ballots,
2. uses it to fill in each target voter's unanswered projects — v5 and v6 get a
   predicted opinion on Garden and Crossings,
3. runs greedy approval over the now-complete profile, taking projects in order
   of approval until the money runs out.

Answers a voter actually gave are never overwritten; only the gaps are filled.

The winning bundle costs 18 + 16 = 34 of the 40 available. Shade cannot join it —
that would come to 46.

Swap in `predictor="matrix_factorization"` or `"factorization_machines"` for the
other two modules, or pass your own function with the same signature.

### Sizing a real process

`plan_sampling` turns a vote budget into a plan. Say you can afford 30% of all
possible votes from 5000 voters over 100 projects, and want a tenth of those
votes to come from complete ballots:

```python
from pabutools_recommendation import plan_sampling

plan_sampling(5000, 100, sample_degree=0.3, lv_degree=0.1)
# (150, 28)   ->  150 people fill in the whole ballot,
#                 each of the other 4850 answers 28 questions
```

### Measuring what asking less costs you

If you already hold complete ballots, `run_experiment` hides part of them,
predicts them back, and lets you compare the result against what full
information would have produced:

```python
from pabutools_recommendation import (
    run_experiment, greedy_approval, fractional_allocation_score,
)

full_profile = ApprovalProfile([
    ApprovalBallot([garden, library]),  ApprovalBallot([garden, shade]),
    ApprovalBallot([garden, library, shade]), ApprovalBallot([garden, crossings]),
    ApprovalBallot([garden, library]),  ApprovalBallot([garden, shade]),
])

real      = set(greedy_approval(instance, full_profile))
predicted = set(run_experiment(instance, full_profile, 0.5, 0.5,
                               setup="offline_controversiality", seed=17))

real == predicted                                              # True
fractional_allocation_score(real, predicted, instance.budget_limit)  # 0.85
```

Here half the votes were collected instead of all of them, and the outcome came
out identical. The score is the cost of the correctly predicted projects as a
share of the budget: 34 of 40.

`classification_metrics` scores the predicted *votes* rather than the outcome,
reporting precision, recall and F1 over the projects a voter was never asked
about. `run_all_experiments` sweeps the whole grid — every setup against every
prediction module, across a range of sampling levels.

Every public function carries a docstring with worked examples, so `help()` is
the reference: `help(pabutools_recommendation)` for the tour, or `help` on any
individual function.

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

Tests that fit a real model skip themselves when the corresponding optional
library is absent, so a bare install still runs a green suite. Install the `ml`
extra to run all of them. The doctests in every module are collected by
`tests/test_doctests.py`, so the worked examples in the docstrings are checked
too.

## Experiments

[`experiments/`](experiments/) holds the performance study — how the setups and
prediction modules compare, how they scale to 480 projects, and what the
optimisation work bought. Results (CSVs and plots) are committed alongside the
scripts, and [experiments/README.md](experiments/README.md) writes up the
findings.

```bash
pip install -e ".[ml,experiments]"
python -m experiments.compare_setups     # algorithm comparison
python -m experiments.compare_improvement  # before vs after optimisation
```

## License

GPLv3, matching pabutools. See [LICENSE.md](LICENSE.md).
