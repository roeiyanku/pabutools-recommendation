"""
Unit tests for `pabutools_recommendation.analytics`, the Section 5 evaluation of
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

Run with:  python -m unittest tests.test_analytics -v

Programmer: Roei Yanku
"""

from unittest import TestCase

from pabutools.election import Project

from pabutools_recommendation import (
    classification_metrics,
    fractional_allocation_score,
)


class TestClassificationMetrics(TestCase):
    """Section 5.1: precision / recall / F1 over one voter's hidden projects."""

    def test_mixed_hit_miss_false_alarm(self):
        p = {name: Project(name, 1) for name in ("p1", "p2", "p3", "p4")}
        hidden = set(p.values())
        # really approves {p1, p2}, predicted {p1, p3}: hit p1, miss p2, false p3.
        m = classification_metrics({p["p1"], p["p2"]}, {p["p1"], p["p3"]}, hidden)
        assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def test_exposed_votes_excluded(self):
        # A wrong prediction on an *exposed* project must not affect the metrics:
        # only the hidden set is scored. Hidden = {p2}, predicted perfectly there.
        p = {name: Project(name, 1) for name in ("p1", "p2")}
        m = classification_metrics(
            real_approved={p["p2"]},          # p1 exposed, p2 hidden+approved
            predicted_approved={p["p1"], p["p2"]},
            hidden={p["p2"]},
        )
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


class TestFractionalAllocation(TestCase):
    """Section 5.2: the Fractional Allocation score of a predicted bundle."""

    def test_partial_overlap(self):
        p = {name: Project(name, cost)
             for name, cost in (("p1", 2), ("p2", 3), ("p3", 5))}
        # overlap is {p2}, cost 3, budget 10 -> 0.3
        score = fractional_allocation_score(
            {p["p1"], p["p2"]}, {p["p2"], p["p3"]}, budget_limit=10
        )
        self.assertAlmostEqual(float(score), 0.3)
