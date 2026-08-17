"""
Unit tests for `pabutools_recommendation`, the implementation of the algorithms in
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

Run with:  python -m unittest tests.test_recommendation -v

Each test is the specification its function must satisfy. They come in three
kinds:
    * small, hand-checked instances (the examples from the paper);
    * edge cases (partial ballots and invalid setup/k values);
    * large inputs, checked against an independent reference or via invariants.

Programmer: Roei Yanku
Date: 2026-06-20.
"""

import random
from collections import Counter
from importlib.util import find_spec
from unittest import SkipTest, TestCase, skipUnless

from parameterized import parameterized

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    total_cost,
)

from pabutools_recommendation import (
    # Partial-ballot helpers + the +1/-1/0 convention constants.
    partial_ballot,
    reveal_ballot,
    approved_projects,
    disapproved_projects,
    exposed_projects,
    hidden_projects,
    consensus_levels,
    greedy_approval,
    random_setup,
    offline_popularity,
    offline_consensus,
    offline_controversiality,
    online_adaptive_controversial,
    next_adaptive_question,
    exposed_sets,
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
    run_pipeline,
    run_experiment,
    split_lv_tv,
    plan_sampling,
    elect,
)
from pabutools_recommendation.model_training import (
    train_factorization_machines,
    train_matrix_factorization,
)

# Evaluating the system lives with the rest of pabutools' analysis tools; the
# end-to-end test below scores its own result with the paper's FA metric.
from pabutools_recommendation import fractional_allocation_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_projects(specs):
    """specs: list of (name, cost) -> dict name -> Project."""
    return {name: Project(name, cost) for name, cost in specs}


# The three prediction modules are backed by the optional ``recommendation``
# extra, which the library's own CI does not install, so a test that fits a
# model reports "skipped" there instead of failing. Everything that does not
# need a model - the samplers, the ballot algebra, the pipeline's validation,
# the metrics and the voting rule - runs unconditionally.
def _installed(module):
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


PREDICTOR_LIBRARY = {
    "predict_by_classification": "xgboost",
    "predict_by_matrix_factorization": "surprise",
    "predict_by_factorization_machines": "lightfm",
}


def requires(*modules):
    """Skip the decorated test unless every named library is installed."""
    missing = [module for module in modules if not _installed(module)]
    return skipUnless(
        not missing,
        f"needs {', '.join(missing)} (pip install pabutools-recommendation[ml])",
    )


def require_predictor(predictor):
    """Skip the running test when this predictor's library is absent.

    The per-predictor counterpart of :py:func:`requires`, for the tests
    ``parameterized`` expands over :py:data:`LIBRARY_PREDICTORS`: which
    library a case needs is only known once the case is running.
    """
    module = PREDICTOR_LIBRARY[predictor.__name__]
    if not _installed(module):
        raise SkipTest(
            f"needs {module} (pip install pabutools-recommendation[ml])"
        )


def consensus_data():
    """4 LV voters: p1 approved 4/0, p2 split 2/2, p3 rejected 0/4."""
    p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
    inst = Instance(p.values(), budget_limit=3)
    lv = ApprovalProfile(
        [
            ApprovalBallot([p["p1"], p["p2"]]),
            ApprovalBallot([p["p1"]]),
            ApprovalBallot([p["p1"], p["p2"]]),
            ApprovalBallot([p["p1"]]),
        ]
    )
    return p, inst, lv


def random_instance(n_projects, n_voters, budget, seed):
    """Build a random instance + approval profile for property testing."""
    rng = random.Random(seed)
    projects = [Project(f"p{i}", rng.randint(1, 10)) for i in range(n_projects)]
    inst = Instance(projects, budget_limit=budget)
    ballots = []
    for _ in range(n_voters):
        approved = [p for p in projects if rng.random() < 0.3]
        ballots.append(ApprovalBallot(approved))
    return projects, inst, ApprovalProfile(ballots)


def padded_projects(m, cost):
    """m unit-named projects p000, p001, ... so str-order == index-order."""
    width = len(str(m - 1))
    return [Project(f"p{i:0{width}d}", cost) for i in range(m)]


def manual_scores(projects, profile):
    """Independent (Counter-based) reimplementation of the approval scores."""
    counts = Counter()
    for ballot in profile:
        for proj in ballot:
            counts[proj] += 1
    return {proj: counts[proj] for proj in projects}


def reference_greedy_approval(projects, profile, budget):
    """
    Independent reference implementation of greedy approval, used to cross-check
    `greedy_approval` on random inputs (the "compare to another algorithm"
    strategy). Projects are taken in decreasing score, ties broken by name, and
    funded while they fit the budget.
    """
    counts = manual_scores(projects, profile)
    ordered = sorted(projects, key=lambda p: (-counts[p], str(p)))
    chosen, spent = [], 0
    for p in ordered:
        if spent + p.cost <= budget:
            chosen.append(p)
            spent += p.cost
    return chosen


# ---------------------------------------------------------------------------
# consensus_levels
# (Approval scores, Definition 2.1, have no function of our own - the code uses
# pabutools' profile.approval_scores() directly, which the library itself tests.)
# ---------------------------------------------------------------------------
class TestConsensusLevels(TestCase):
    def test_hand_checked_split(self):
        p, inst, lv = consensus_data()
        c = consensus_levels(inst, lv)
        assert c[p["p1"]] == 4
        assert c[p["p2"]] == 0
        assert c[p["p3"]] == 4

    def test_large_structured_even_split_is_zero(self):
        # Exactly half approve and half reject each project -> consensus 0.
        projects = padded_projects(20, cost=1)
        inst = Instance(projects, budget_limit=10)
        prof = ApprovalProfile(
            [ApprovalBallot(projects)] * 50 + [ApprovalBallot([])] * 50
        )
        c = consensus_levels(inst, prof)
        assert all(c[p] == 0 for p in projects)


# ---------------------------------------------------------------------------
# greedy_approval (voting rule)
# ---------------------------------------------------------------------------
class TestGreedyApproval(TestCase):
    def test_skips_unaffordable_but_funds_affordable(self):
        # A project that exceeds the budget is skipped, while a cheaper one that
        # fits is funded.
        p = make_projects([("cheap", 10), ("dear", 100)])
        inst = Instance(p.values(), budget_limit=10)
        prof = ApprovalProfile([ApprovalBallot([p["cheap"], p["dear"]])])
        assert set(greedy_approval(inst, prof)) == {p["cheap"]}

    def test_matches_reference_random(self):
        # Cross-check against the independent greedy reference, several seeds.
        for seed in range(5):
            projects, inst, prof = random_instance(30, 50, 40, seed=200 + seed)
            got = set(greedy_approval(inst, prof))
            expected = set(reference_greedy_approval(projects, prof, inst.budget_limit))
            assert got == expected


# ---------------------------------------------------------------------------
# random_setup
# ---------------------------------------------------------------------------
class TestRandomSetup(TestCase):
    def test_exposes_k_independently_per_voter(self):
        # One voter's draw: exactly k real projects.
        p = make_projects([("p1", 2), ("p2", 2), ("p3", 3), ("p4", 3)])
        inst = Instance(p.values(), budget_limit=6)
        exposed = random_setup(inst, k=2, seed=0)
        assert len(exposed) == 2 and exposed <= set(p.values())

        # Section 2.4: "possibly different E for each voter" - one seed must
        # not collapse the whole population onto a single shared random set,
        # yet the same seed must reproduce the same draws.
        projects = padded_projects(8, cost=1)
        inst = Instance(projects, budget_limit=8)
        lv = ApprovalProfile([ApprovalBallot(projects[:2])])
        tv = {f"v{i}": set() for i in range(12)}
        exposed = exposed_sets(inst, lv, tv, "random", k=2, seed=123)
        assert all(len(e) == 2 for e in exposed.values())
        assert len({frozenset(e) for e in exposed.values()}) > 1
        assert exposed == exposed_sets(inst, lv, tv, "random", k=2, seed=123)

# ---------------------------------------------------------------------------
# offline_popularity / consensus / controversiality
# ---------------------------------------------------------------------------
class TestOfflineSamplers(TestCase):
    def test_popularity_top_k(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile(
            [
                ApprovalBallot([p["p1"], p["p3"]]),
                ApprovalBallot([p["p1"], p["p2"]]),
                ApprovalBallot([p["p1"]]),
            ]
        )
        assert offline_popularity(inst, lv, k=1) == {p["p1"]}

    def test_consensus_top_k(self):
        p, inst, lv = consensus_data()
        assert offline_consensus(inst, lv, k=1) == {p["p1"]}

    def test_controversiality_bottom_k(self):
        p, inst, lv = consensus_data()
        assert offline_controversiality(inst, lv, k=1) == {p["p2"]}

# ---------------------------------------------------------------------------
# online_adaptive_controversial
# ---------------------------------------------------------------------------
class TestOnlineAdaptive(TestCase):
    def test_adaptive_feedback_shifts_the_questions(self):
        # One TV voter: p1, p2, p3 are tied as most controversial (consensus 0),
        # ties broken by name, so the two questions are p1 then p2.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        inst = Instance(p.values(), budget_limit=4)
        lv = ApprovalProfile(
            [
                ApprovalBallot([p["p1"], p["p2"]]),
                ApprovalBallot([p["p1"], p["p3"]]),
                ApprovalBallot([p["p2"], p["p3"]]),
                ApprovalBallot([]),
            ]
        )
        assert online_adaptive_controversial(
            inst, lv, {"v5": {p["p1"], p["p2"]}}, k=2
        ) == {"v5": {p["p1"], p["p2"]}}

        # The feedback loop across voters. p1 and p2 start tied as most
        # controversial (both split 1-1 among LV), so the first voter is
        # always asked p1, and her answer - approval *or* disapproval - breaks
        # the tie, steering the second voter to p2.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]]), ApprovalBallot([p["p2"]])])
        assert online_adaptive_controversial(
            inst, lv, {"v3": {p["p1"]}, "v4": {p["p1"]}}, k=1
        ) == {"v3": {p["p1"]}, "v4": {p["p2"]}}
        assert online_adaptive_controversial(
            inst, lv, {"v3": set(), "v4": {p["p1"]}}, k=1
        ) == {"v3": {p["p1"]}, "v4": {p["p2"]}}

        # And across more than one step: after v3 approves p1 (2-1) and v4
        # approves p2 (2-1), both are equally off-split again for v5 and the
        # name tie-break returns to p1.
        assert online_adaptive_controversial(
            inst, lv, {"v3": {p["p1"]}, "v4": {p["p2"]}, "v5": set()}, k=1
        ) == {"v3": {p["p1"]}, "v4": {p["p2"]}, "v5": {p["p1"]}}

    def test_step_function_reproduces_the_closed_loop(self):
        # next_adaptive_question is the open-loop form of the same rule, for a
        # live process where the answers arrive one at a time. Driving it with
        # a voter's ballot must ask exactly what the closed loop asks.
        rng = random.Random(0)
        for _ in range(60):
            n_projects, n_lv = rng.randint(3, 7), rng.randint(2, 6)
            k = rng.randint(1, n_projects)
            p = make_projects([(f"p{i}", 1) for i in range(n_projects)])
            projects = list(p.values())
            inst = Instance(projects, budget_limit=n_projects)
            lv = ApprovalProfile([
                ApprovalBallot([q for q in projects if rng.random() < 0.5])
                for _ in range(n_lv)
            ])
            truth = {q for q in projects if rng.random() < 0.5}

            closed = online_adaptive_controversial(inst, lv, {"v": truth}, k)["v"]

            answers = partial_ballot()
            for _ in range(k):
                asked = next_adaptive_question(inst, lv, answers)
                approved = approved_projects(answers)
                disapproved = disapproved_projects(answers)
                if asked in truth:
                    approved = approved | {asked}
                else:
                    disapproved = disapproved | {asked}
                answers = partial_ballot(
                    approved=approved, disapproved=disapproved
                )
            assert exposed_projects(answers) == closed

    def test_step_function_rejects_a_fully_answered_ballot(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])])
        answered = partial_ballot(approved={p["p1"]}, disapproved={p["p2"]})
        with self.assertRaisesRegex(ValueError, "already been asked"):
            next_adaptive_question(inst, lv, answered)

# ---------------------------------------------------------------------------
# Partial (three-state) ballots, Section 2.3. Represented as a CardinalBallot
# under the convention: +1 approved, -1 disapproved, 0 (or absent) hidden.
# ---------------------------------------------------------------------------
class TestPartialBallot(TestCase):
    def test_scores_follow_convention(self):
        # The whole point of the encoding: +1 = approve, -1 = disapprove,
        # 0 = unknown. Pin it down so the convention can't silently drift.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        b = partial_ballot(
            approved={p["p1"]}, disapproved={p["p2"]}, hidden={p["p3"]}
        )
        assert b[p["p1"]] == 1
        assert b[p["p2"]] == -1
        assert b[p["p3"]] == 0

    def test_absent_project_is_hidden(self):
        # A project never mentioned in the ballot is hidden, just like an
        # explicit 0 - both belong to H_v.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        b = partial_ballot(approved={p["p1"]}, disapproved={p["p2"]})  # p3 absent
        assert hidden_projects(b, inst) == {p["p3"]}

    def test_reveal_example2_4(self):
        # Example 2.4: full ballot {p1,p2}, exposed set {p1,p3}.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        inst = Instance(p.values(), budget_limit=4)
        b = reveal_ballot(inst, {p["p1"], p["p2"]}, {p["p1"], p["p3"]})
        assert approved_projects(b) == {p["p1"]}
        assert disapproved_projects(b) == {p["p3"]}
        assert hidden_projects(b, inst) == {p["p2"], p["p4"]}
        assert exposed_projects(b) == {p["p1"], p["p3"]}


# ---------------------------------------------------------------------------
# The three prediction modules of the paper: classification (XGBoost), MF, FM
# (Section 2.1). They share one contract - keep the exposed votes, predict the
# hidden ones - so the same behavioural invariants are checked for each.
# ---------------------------------------------------------------------------
LIBRARY_PREDICTORS = [
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
]


class TestLibraryPredictors(TestCase):
    @parameterized.expand([(p.__name__, p) for p in LIBRARY_PREDICTORS])
    def test_exposed_votes_are_respected(self, _name, predictor):
        # The exposed approval p3 is kept and the exposed disapproval p1 stays
        # rejected, regardless of what the model would otherwise predict.
        require_predictor(predictor)
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])] * 3)
        partial = partial_ballot(
            approved={p["p3"]}, disapproved={p["p1"]}, hidden={p["p2"]}
        )
        pred = predictor(inst, lv, {"v": partial})["v"]
        assert p["p3"] in pred       # exposed approval kept
        assert p["p1"] not in pred   # exposed disapproval rejected

    @requires("surprise", "lightfm")
    def test_cf_pools_e_tv_and_fm_uses_the_cost_feature(self):
        # Section 3.1: prediction is based on LV *and* E_TV, the exposed votes
        # of all TV voters. The lone LV voter rejects p2, but five TV voters
        # expose approvals of it; a sixth TV voter with nothing exposed must
        # pick p2 up from their pooled answers.
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])])
        tv = {
            f"b{i}": partial_ballot(approved={p["p2"]}, hidden={p["p1"]})
            for i in range(5)
        }
        tv["a"] = partial_ballot(hidden=set(p.values()))
        completed = predict_by_matrix_factorization(inst, lv, tv)
        assert p["p2"] in completed["a"]

        # The FM hybrid ingredient: the voter approves her exposed cheap
        # projects and rejects the exposed expensive ones; the fitted cost
        # slope carries that pattern to the hidden pair. Plain MF has no cost
        # feature and can only reach the same answer through the e1/e2/e3
        # correlation in the LV ballots, so it separates the two far more
        # weakly - which is exactly how FM and MF must differ.
        p = make_projects(
            [("c1", 1), ("c2", 1), ("c3", 1), ("e1", 10), ("e2", 10), ("e3", 10)]
        )
        inst = Instance(p.values(), budget_limit=12)
        cheap = {p["c1"], p["c2"], p["c3"]}
        lv = ApprovalProfile([ApprovalBallot(p.values())] * 3
                             + [ApprovalBallot(cheap)] * 2)
        partial = partial_ballot(
            approved={p["c1"], p["c2"]},
            disapproved={p["e1"], p["e2"]},
            hidden={p["c3"], p["e3"]},
        )
        fm = predict_by_factorization_machines(inst, lv, {"v": partial})["v"]
        assert p["c3"] in fm and p["e3"] not in fm
        # Compare the cheap/expensive score gaps rather than the completed
        # ballots: whether MF's e3 lands either side of the 0.5 cut-off depends
        # on its latent rank, but its gap is always the weaker of the two.
        exposed = {"v": ({p["c1"], p["c2"]}, {p["e1"], p["e2"]})}
        mf_scores = train_matrix_factorization(inst, lv, exposed)["v"]
        fm_scores = train_factorization_machines(inst, lv, exposed)["v"]
        gap = lambda s: s[p["c3"]] - s[p["e3"]]  # noqa: E731
        assert gap(fm_scores) > gap(mf_scores)

    @requires("lightfm")
    def test_fm_uses_the_pabulib_category_and_target_features(self):
        # The hybrid ingredient of Section 2.1.2, on Table 2's *set-valued*
        # project attributes rather than cost. Every project costs the same and
        # every LV voter approves everything, so neither cost nor the approval
        # scores can separate them: the only thing that can carry this voter's
        # exposed pattern to the projects she was never asked about is the
        # category / target metadata that ``parse_pabulib`` fills in.
        p = make_projects([(name, 1) for name in
                           ("g1", "g2", "g3", "r1", "r2", "r3")])
        for name in ("g1", "g2", "g3"):
            p[name].categories, p[name].targets = {"greenery"}, {"children"}
        for name in ("r1", "r2", "r3"):
            p[name].categories, p[name].targets = {"roads"}, {"drivers"}
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile([ApprovalBallot(p.values())] * 4)

        # She approves the greenery she was shown and rejects the roads.
        exposed = {"v": ({p["g1"], p["g2"]}, {p["r1"], p["r2"]})}
        scores = train_factorization_machines(inst, lv, exposed)
        # So the unseen greenery project must outscore the unseen road one.
        assert scores["v"][p["g3"]] > scores["v"][p["r3"]]


# ---------------------------------------------------------------------------
# run_pipeline (full pipeline)
# (split_lv_tv's Section 3.0.1 vote-budget arithmetic - TV = all non-LV
# voters, derived k, the lv_degree=1 baseline, degree validation - is pinned
# by its doctests in pabutools/recommendation/recommendation.py.)
# ---------------------------------------------------------------------------
class TestRunPipeline(TestCase):
    def test_invalid_setup_or_k_raises_value_error(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])] * 2)
        with self.assertRaisesRegex(ValueError, "unknown setup 'by_magic'"):
            run_pipeline(inst, lv, {"v1": {p["p1"]}}, k=1,
                         setup="by_magic",
                         predict=predict_by_matrix_factorization)
        for bad_k in (-1, 3):  # below 0 and above the number of projects
            with self.assertRaisesRegex(ValueError, "must be between 0 and"):
                run_pipeline(inst, lv, {"v1": {p["p1"]}}, k=bad_k,
                             setup="random",
                             predict=predict_by_matrix_factorization)

    @parameterized.expand([
        (f"{setup}_{predictor.__name__}", setup, predictor)
        for setup in (
            "random",
            "offline_popularity",
            "offline_consensus",
            "offline_controversiality",
            "online_adaptive_controversial",
        )
        for predictor in LIBRARY_PREDICTORS
    ])
    def test_two_camp_electorate_recovers_real_bundle(self, _name, setup, predictor):
        """
        The end-to-end criterion of the paper (Section 2.4): the pipeline must
        estimate the *ideal* outcome. A structured two-camp electorate - a 60%
        majority camp approving {p1, p2, p3} and a 40% minority camp approving
        {p4, p5, p6} - fixes the real winning bundle at {p1, p2, p3}. A third
        of the voters become TV with only k=2 of the 6 projects exposed, and
        every setup x predictor combination must still reconstruct the exact
        real bundle (FA = 1.0, SD = 0).
        """
        require_predictor(predictor)
        p = make_projects([(f"p{i}", 1) for i in range(1, 7)])
        camp_a = {p["p1"], p["p2"], p["p3"]}   # 18 of 30 voters (60%)
        camp_b = {p["p4"], p["p5"], p["p6"]}   # 12 of 30 voters (40%)
        inst = Instance(p.values(), budget_limit=3)

        # The real bundle, from all 30 full ballots: camp A's projects score
        # 18 > 12, and the budget funds exactly three unit-cost projects.
        full_profile = ApprovalProfile(
            [ApprovalBallot(camp_a)] * 18 + [ApprovalBallot(camp_b)] * 12
        )
        real_bundle = set(greedy_approval(inst, full_profile))
        assert real_bundle == camp_a  # sanity: the ground truth is as designed

        # LV/TV split preserving the 60/40 mix: 20 LV, 10 TV.
        lv = ApprovalProfile([ApprovalBallot(camp_a)] * 12
                             + [ApprovalBallot(camp_b)] * 8)
        tv = {f"a{i}": set(camp_a) for i in range(6)}
        tv.update({f"b{i}": set(camp_b) for i in range(4)})

        predicted = set(run_pipeline(inst, lv, tv, k=2,
                                     setup=setup, predict=predictor, seed=7))
        assert predicted == real_bundle
        assert fractional_allocation_score(
            real_bundle, predicted, inst.budget_limit) == 1.0
        assert len(real_bundle ^ predicted) == 0  # symmetric distance


# ---------------------------------------------------------------------------
# Running a real process: plan_sampling / elect / run_experiment
# ---------------------------------------------------------------------------
class TestRealProcess(TestCase):
    def test_plan_sampling_matches_the_split_it_describes(self):
        # plan_sampling is the arithmetic split_lv_tv performs, minus the
        # ballots, so the two must agree on every point of the paper's grid.
        p = make_projects([(f"p{i}", 1) for i in range(4)])
        inst = Instance(p.values(), budget_limit=4)
        prof = ApprovalProfile([ApprovalBallot(set(p.values()))] * 10)
        for sample_degree in (0.1, 0.3, 0.5, 0.9):
            for lv_degree in (0.1, 0.5, 0.9, 1.0):
                n_lv, k = plan_sampling(10, 4, sample_degree, lv_degree)
                lv, tv, split_k = split_lv_tv(
                    inst, prof, sample_degree, lv_degree, seed=1
                )
                assert (lv.num_ballots(), split_k) == (n_lv, k)
                assert len(tv) == (0 if lv_degree == 1 else 10 - n_lv)

    @requires("xgboost")
    def test_elect_needs_no_ground_truth(self):
        # The deployment path: nobody's full ballot is known to the algorithm -
        # there are only the answers voters actually gave. Two camps disagree
        # about p1 against p3, every voter was asked about just those two of
        # the four projects, and the budget funds exactly one of them.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        inst = Instance(p.values(), budget_limit=1)
        answers = {}
        for i in range(12):  # majority camp: approve p1/p2, reject p3/p4
            answers[f"a{i}"] = partial_ballot(
                approved={p["p1"]}, disapproved={p["p3"]}
            )
        for i in range(5):   # minority camp: the other way round
            answers[f"b{i}"] = partial_ballot(
                approved={p["p3"]}, disapproved={p["p1"]}
            )
        bundle = elect(inst, ApprovalProfile([]), answers)
        assert total_cost(bundle) <= inst.budget_limit
        assert p["p1"] in bundle and p["p3"] not in bundle

    @requires("xgboost")
    def test_elect_keeps_every_answer_that_was_given(self):
        # Whatever the model thinks, an answer a voter actually gave must
        # survive into her counted ballot (Section 2.4: the completed instance
        # has to agree with the partial one).
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 4)
        # She rejects p1 although every full ballot approves it.
        answers = {"v": partial_ballot(approved={p["p3"]}, disapproved={p["p1"]})}
        completed = predict_by_classification(inst, lv, answers)["v"]
        assert p["p3"] in completed and p["p1"] not in completed

    @requires("xgboost")
    def test_run_experiment_equals_split_then_pipeline(self):
        # The convenience wrapper must be exactly its two parts, same seed.
        p = make_projects([("p1", 2), ("p2", 2), ("p3", 3), ("p4", 3)])
        inst = Instance(p.values(), budget_limit=4)
        prof = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 6)
        lv, tv, k = split_lv_tv(inst, prof, 0.5, 0.3, seed=11)
        expected = set(run_pipeline(inst, lv, tv, k, setup="offline_popularity",
                                    predict=predict_by_classification, seed=11))
        got = set(run_experiment(inst, prof, 0.5, 0.3,
                                 setup="offline_popularity", seed=11))
        assert got == expected

    def test_unknown_predictor_name_is_rejected(self):
        p = make_projects([("p1", 1)])
        inst = Instance(p.values(), budget_limit=1)
        with self.assertRaisesRegex(ValueError, "unknown predictor"):
            elect(inst, ApprovalProfile([]), {}, predictor="by_magic")


if __name__ == "__main__":
    import unittest

    unittest.main()
