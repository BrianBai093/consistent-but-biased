"""The pre-specified statistics, checked against values worked out by hand."""

from fractions import Fraction

import pytest

from tools.analyze_text_experiment import (
    benjamini_hochberg,
    exact_mcnemar,
    holm,
    sign_flip_p,
)


def test_eight_one_way_differences_and_twelve_ties():
    differences = [Fraction(1)] * 8 + [Fraction(0)] * 12
    assert sign_flip_p(differences, "greater") == pytest.approx(1 / 256)
    assert sign_flip_p(differences, "two-sided") == pytest.approx(2 / 256)
    assert sign_flip_p(differences, "less") == pytest.approx(1.0)


def test_shares_of_three_runs_are_handled_exactly():
    differences = [Fraction(1, 3), Fraction(-1, 3), Fraction(2, 3)]
    # Sums of +/-1, +/-1, +/-2 (in thirds): 8 sign patterns, observed 2.
    # Patterns with sum >= 2: (+,+,+)=4, (+,-,+)=2, (-,+,+)=2 -> 3 of 8.
    assert sign_flip_p(differences, "greater") == pytest.approx(3 / 8)


def test_no_difference_at_all_is_not_evidence():
    assert sign_flip_p([Fraction(0)] * 5, "two-sided") == 1.0


def test_mcnemar_matches_the_binomial():
    assert exact_mcnemar(8, 0) == pytest.approx(2 / 256)
    assert exact_mcnemar(0, 10) == pytest.approx(2 / 1024)
    assert exact_mcnemar(2, 3) == 1.0


def test_holm_and_bh():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "c": 0.06, "b": 0.06}
    )
    assert benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "c": 0.04, "b": 0.04}
    )


def test_a_share_of_an_effect_is_a_ratio_of_case_means():
    from tools.analyze_chair_isolation import ratio_ci

    assert ratio_ci([0.5, 1.0], [1.0, 2.0]) == pytest.approx((0.5, 0.5))
    # Where the effect itself can vanish in a resample, its share is not a number.
    assert ratio_ci([1.0, 1.0], [1.0, 0.0]) is None
