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

```bash
pip install pabutools-recommendation
```

Or, until it is on PyPI, straight from the repository — this line also works in
a `requirements.txt`:

```bash
pip install "pabutools-recommendation @ git+https://github.com/roeiyanku/pabutools-recommendation"
```

The three prediction modules need `xgboost`, `scikit-surprise` and `lightfm`,
which are **not** installed by default. They come with the `ml` extra:

```bash
pip install "pabutools-recommendation[ml]"
```

Those libraries are imported only when a model is actually fitted, so the
sampling setups, the voting rule and the pipeline all work without them.

## Quickstart

```python
from pabutools.election import Instance, Project, ApprovalProfile, ApprovalBallot
from pabutools_recommendation import elect, offline_controversiality, partial_ballot

garden    = Project("Garden", 18)
crossings = Project("Crossings", 24)
library   = Project("Library", 16)
shade     = Project("Shade", 12)
instance  = Instance([garden, crossings, library, shade], budget_limit=40)

# The voters who did fill in a complete ballot: what the prediction learns from.
lv_profile = ApprovalProfile([
    ApprovalBallot([garden, library]),
    ApprovalBallot([crossings, shade]),
    ApprovalBallot([garden, shade]),
    ApprovalBallot([garden, library, shade]),
])

# Which two projects to ask everybody else about.
offline_controversiality(instance, lv_profile, k=2)
# {Library, Shade}  - the two the learning voters most disagree about

# Their answers: approved, disapproved, and everything else left unasked.
answers = {
    "v5": partial_ballot(approved={garden}, disapproved={crossings}),
    "v6": partial_ballot(approved={crossings}, disapproved={garden}),
}

elect(instance, lv_profile, answers, predictor="classification")
# BudgetAllocation([Garden, Shade])
```

Already holding complete ballots and want to know what asking less would have
cost? `run_experiment` simulates it, and `fractional_allocation_score` scores the
result against the full-information outcome. See [docs/usage.rst](docs/usage.rst)
for the whole walkthrough, and [docs/reference.rst](docs/reference.rst) for the
API.

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
