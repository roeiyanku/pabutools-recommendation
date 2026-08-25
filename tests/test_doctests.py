"""
Runs the doctests of the recommendation modules as part of the normal test
suite, so the worked examples in the docstrings stay honest.

Run with:  python -m unittest tests.test_doctests -v

These used to run only from a ``if __name__ == "__main__": doctest.testmod()``
block at the bottom of each module, i.e. only when someone remembered to
execute the module by hand. Collecting them here means ``python -m unittest``
and ``pytest`` both check them. (They are wrapped in an ordinary ``TestCase``
rather than handed over with a ``load_tests`` hook, because ``pytest`` does not
implement that protocol and would silently collect none of them.)

A handful of examples fit an actual model and therefore need one of the
optional ``ml`` libraries, which a bare install does not bring in; those are
skipped when their library is absent, exactly as the equivalent unit tests are.
"""

import doctest
from importlib.util import find_spec
from unittest import TestCase

from parameterized import parameterized

from pabutools_recommendation import analytics, model_training, recommendation

#: The modules whose doctests are collected.
MODULES = (recommendation, model_training, analytics)

#: Doctests that fit a model, and the optional library each one needs. Anything
#: not listed here is pure Python and always runs. (The factorization-machines
#: examples carry their own ``# doctest: +SKIP``, so they need no entry.)
DOCTEST_LIBRARY = {
    # xgboost - the classification predictor, and the examples defaulting to it.
    "_xgboost": "xgboost",
    "_fit_one_project": "xgboost",
    "train_classification": "xgboost",
    "predict_by_classification": "xgboost",
    "complete_ballots": "xgboost",
    "run_experiment": "xgboost",
    "elect": "xgboost",
    # scikit-surprise - the matrix-factorization predictor.
    "_fit_mf": "surprise",
    "train_matrix_factorization": "surprise",
    "predict_by_matrix_factorization": "surprise",
    "run_pipeline": "surprise",
    "run_all_experiments": "surprise",
}


def _installed(module):
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _collect():
    """Every non-empty doctest of :py:data:`MODULES`, as (name, test) pairs."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    return [
        (test.name, test)
        for module in MODULES
        for test in sorted(finder.find(module), key=lambda t: t.name)
    ]


class TestDoctests(TestCase):
    @parameterized.expand(_collect())
    def test_doctest(self, name, test):
        library = DOCTEST_LIBRARY.get(name.rsplit(".", 1)[-1])
        if library is not None and not _installed(library):
            self.skipTest(
                f"needs {library} (pip install pabutools-recommendation[ml])"
            )
        failures = []
        # Default flags, i.e. exactly what the removed ``testmod()`` calls used.
        runner = doctest.DocTestRunner(optionflags=0)
        runner.run(test, out=failures.append)
        assert not runner.failures, (
            f"{runner.failures} of {runner.tries} examples failed in {name}:\n"
            + "".join(failures)
        )
